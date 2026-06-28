import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from numpy import ndarray
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .callbacks import Callback, NormalizationCallback, DebugCallback
from .data import EvenetTensorDataset, build_sampler, DistributedWeightedSampler
from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import (
    calculate_physics_metrics,
    classification_auc_from_score_histograms,
    classification_metrics_from_confusion_matrix,
    compute_sic_from_scores,
    compute_sic_from_score_histograms,
    compute_accuracy,
    compute_classification_metrics,
    compute_loss,
    summarize_metrics,
)
from .optim import (
    build_optimizers_and_schedulers,
    DEFAULT_LR_GROUPS,
    DEFAULT_MODULE_GROUPS,
    DEFAULT_WEIGHT_DECAY,
    set_peft_trainable,
    print_trainable
)


def format_metrics_for_logging(
        metrics: Dict[str, Any],
        *,
        exclude_keys: Optional[Iterable[str]] = None,
        float_fmt: str = ".5f",
) -> str:
    """Format evaluation metrics for clean logging.

    - Excludes selected keys
    - Nicely formats scalars
    - Summarizes non-scalars
    """
    exclude_keys = set(exclude_keys or [])

    lines = []
    for key in sorted(metrics.keys()):
        if key in exclude_keys:
            continue

        val = metrics[key]

        # Scalars
        if isinstance(val, (int, float)):
            lines.append(f"{key:>24s} : {val:{float_fmt}}")

        # 0-dim tensors / numpy scalars
        elif hasattr(val, "item") and callable(val.item):
            try:
                lines.append(f"{key:>24s} : {val.item():{float_fmt}}")
            except Exception:
                lines.append(f"{key:>24s} : <tensor>")

        # Arrays / tensors
        elif hasattr(val, "shape"):
            lines.append(f"{key:>24s} : array{tuple(val.shape)}")

        # Everything else
        else:
            lines.append(f"{key:>24s} : {type(val).__name__}")

    return "\n".join(lines)


@dataclass
class TrainerConfig:
    device: str = "auto"
    lr: List[float] = field(default_factory=lambda: list(DEFAULT_LR_GROUPS))
    weight_decay: List[float] = field(default_factory=lambda: [DEFAULT_WEIGHT_DECAY] * len(DEFAULT_LR_GROUPS))
    module_lists: List[List[str]] = field(default_factory=lambda: [list(group) for group in DEFAULT_MODULE_GROUPS])
    grad_clip: Optional[float] = None
    num_workers: int = 2
    scheduler_fn: Optional[Any] = None
    optimizer_fn: Optional[Any] = None
    warmup_epochs: Optional[int] = 1
    warmup_ratio: float = 0.1
    warmup_start_factor: float = 0.1
    min_lr: float = 0.0
    checkpoint_path: Optional[str] = None
    checkpoint_every: int = 1
    resume_from: Optional[str] = None
    use_wandb: bool = False
    wandb: Optional[Dict[str, Any]] = None
    compute_physics_metrics: bool = True
    physics_bins: int = 1000
    physics_metric_config: Dict[str, Any] = field(default_factory=dict)
    classification_score_bins: int = 100
    loss_gamma: float = 0.0
    eval_batch_size: Optional[int] = None
    eval_output_path: Optional[str] = None
    save_top_k: int = 0
    monitor_metric: str = "val_loss"
    minimize_metric: bool = True
    early_stop_metric: str = "val_loss"
    early_stop_minimize: bool = True
    early_stop_patience: int = 0
    find_unused_parameters: bool = True
    use_peft: bool = False  # new field to indicate whether to use PEFT


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            feature_names: Dict[str, Iterable[str]],
            config: TrainerConfig,
            callbacks: Optional[List[Callback]] = None,
            class_labels: Optional[List[str]] = None,
            debug: bool = False,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.callbacks: List[Callback] = callbacks or []
        self.debug = debug
        if self.debug and not any(isinstance(cb, DebugCallback) for cb in self.callbacks):
            self.callbacks.append(DebugCallback())
        self._init_distributed()
        self.device = self._resolve_device(config.device)

        self.global_step = 0

        self.class_labels = class_labels
        self.num_classes = len(class_labels) if class_labels is not None else self._infer_num_classes(model)
        self.train_accuracy = None
        self.val_accuracy = None
        self._init_metrics()
        self.wandb_run = None
        self._maybe_init_wandb()
        self.train_dataset: EvenetTensorDataset
        self.val_dataset: Optional[EvenetTensorDataset] = None
        self.test_dataset: Optional[EvenetTensorDataset] = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.optimizers: List[torch.optim.Optimizer] = []
        self.optimizer_tags: List[str] = []
        self.scheduler: Optional[Any] = None
        self.schedulers: List[Any] = []
        self._best_checkpoints: List[Tuple[float, str]] = []
        self.train_sampler: Optional[Any] = None
        self.val_sampler: Optional[Any] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def _init_distributed(self) -> None:
        if dist.is_available() and not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend, init_method="env://")
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                if dist.is_available() and dist.is_initialized():
                    torch.cuda.set_device(self.local_rank)
                    return torch.device(f"cuda:{self.local_rank}")
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def _infer_num_classes(self, model: torch.nn.Module) -> Optional[int]:
        if hasattr(model, "num_classes"):
            classes = getattr(model, "num_classes")
            if isinstance(classes, dict):
                return next(iter(classes.values()), None)
            if isinstance(classes, int):
                return classes
        return None

    def _init_metrics(self) -> None:
        if self.num_classes is None:
            logging.info("Skipping torchmetrics initialization because num_classes could not be inferred.")
            return
        try:
            from torchmetrics import Accuracy  # type: ignore

            self.train_accuracy = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                compute_on_cpu=True,
                sync_on_compute=True,
            ).to(self.device)
            self.val_accuracy = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                compute_on_cpu=True,
                sync_on_compute=True,
            ).to(self.device)
            logging.info("Initialized torchmetrics Accuracy with num_classes=%s", self.num_classes)
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("torchmetrics is unavailable; falling back to manual accuracy. Error: %s", exc)

    def _maybe_init_wandb(self) -> None:
        if not self.config.use_wandb:
            return
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("Weights & Biases requested but import failed: %s", exc)
            return

        if not self.is_rank_zero():
            logging.info("Skipping Weights & Biases init on non-zero rank %s", self.rank)
            return

        wandb_settings = self.config.wandb or {}
        self.wandb_run = wandb.init(
            project=wandb_settings.get("project"),
            name=wandb_settings.get("name"),
            config=wandb_settings.get("config", {}),
            entity=wandb_settings.get("entity"),
            mode=wandb_settings.get("mode"),
            group=wandb_settings.get("group"),
            job_type=wandb_settings.get("job_type"),
            tags=wandb_settings.get("tags"),
            notes=wandb_settings.get("notes"),
            dir=wandb_settings.get("dir", "./"),
            reinit=True,
        )
        logging.info("Initialized Weights & Biases run: %s", self.wandb_run.name if self.wandb_run else "<none>")

    def attach_normalizer(self, normalizer: Any) -> None:
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            self.train_dataset.set_normalizer(normalizer)
        if getattr(self, "val_dataset", None):
            self.val_dataset.set_normalizer(normalizer)
        if getattr(self, "test_dataset", None):
            self.test_dataset.set_normalizer(normalizer)

    def attach_normalizer_state(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            if isinstance(cb, NormalizationCallback) and cb.normalizer is not None:
                cb.normalizer.load_state_dict(state)
                self.attach_normalizer(cb.normalizer)
                break

    def _current_normalizer_state(self) -> Optional[Dict[str, Any]]:
        for cb in self.callbacks:
            if isinstance(cb, NormalizationCallback) and cb.normalizer is not None:
                return cb.normalizer.state_dict()
        return None

    def _all_gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return tensor
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=0)

    def _all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _distributed_barrier(self) -> None:
        if self.world_size <= 1:
            return
        if dist.get_backend() == "nccl" and self.device.type == "cuda":
            device_id = self.device.index if self.device.index is not None else self.local_rank
            dist.barrier(device_ids=[device_id])
        else:
            dist.barrier()

    @staticmethod
    def _classification_probabilities_tensor(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            sig = torch.sigmoid(logits)
            return torch.stack([1.0 - sig, sig], dim=1)
        if logits.shape[1] == 1:
            sig = torch.sigmoid(logits[:, 0])
            return torch.stack([1.0 - sig, sig], dim=1)
        return torch.softmax(logits, dim=1)

    def _reduce_mean_scalar(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.device)
        tensor = self._all_reduce_tensor(tensor)
        tensor = tensor / max(1, self.world_size)
        return tensor.item()

    def setup_datasets(
            self,
            train_data: Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
            val_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
            test_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
    ) -> None:
        X_train, y_train, w_train = train_data
        self.train_dataset = EvenetTensorDataset(X_train, y_train, w_train)
        self.val_dataset = EvenetTensorDataset(*val_data) if val_data is not None else None
        self.test_dataset = EvenetTensorDataset(*test_data) if test_data is not None else None

    def _maybe_wrap_ddp(self) -> torch.nn.Module:
        if isinstance(self.model, DDP):
            return self.model

        model = self.model.to(self.device)
        if self.world_size > 1:
            if self.device.type == "cuda":
                device_id = self.device.index if self.device.index is not None else self.local_rank
                model = DDP(
                    model,
                    device_ids=[device_id],
                    find_unused_parameters=self.config.find_unused_parameters,
                )
            else:
                model = DDP(model, find_unused_parameters=self.config.find_unused_parameters)
        return model

    def _unwrap_model(self) -> torch.nn.Module:
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _load_model_state(self, state: Optional[Dict[str, torch.Tensor]]) -> None:
        if self.world_size > 1:
            # payload = [state]
            # dist.broadcast_object_list(payload, src=0)
            # state = payload[0]

            payload = [None]
            if dist.get_rank() == 0:
                payload[0] = state  # must be CPU objects for object broadcast

            dist.broadcast_object_list(payload, src=0)
            state = payload[0]
        if state is None:
            return
        self._unwrap_model().load_state_dict(state)

    def _describe_sampler(self, sampler_obj: Optional[Any]) -> str:
        if sampler_obj is None:
            return "None (shuffled)"
        details: List[str] = [sampler_obj.__class__.__name__]
        if isinstance(sampler_obj, DistributedWeightedSampler):
            details.append(f"epoch_size={sampler_obj.epoch_size}")
            details.append(f"replacement={sampler_obj.replacement}")
        if isinstance(sampler_obj, DistributedSampler):
            details.append(f"num_replicas={sampler_obj.num_replicas}")
            details.append(f"rank={sampler_obj.rank}")
        return ", ".join(details)

    def _class_stats(self, dataset: EvenetTensorDataset) -> Tuple[int, torch.Tensor, torch.Tensor]:
        labels = dataset.labels.long()
        inferred_classes = self.num_classes if self.num_classes is not None else int(labels.max().item() + 1)
        num_classes = max(inferred_classes, int(labels.max().item() + 1))
        counts = torch.bincount(labels, minlength=num_classes)

        if dataset.sample_weights is None:
            weights = torch.ones_like(labels, dtype=torch.float32)
        else:
            weights = torch.as_tensor(dataset.sample_weights, dtype=torch.float32)
            finite_mask = torch.isfinite(weights)
            weights = torch.where(finite_mask, weights, torch.zeros_like(weights))

        weight_sums = torch.zeros(num_classes, dtype=torch.float32)
        for cls_idx in range(num_classes):
            mask = labels == cls_idx
            if mask.any():
                weight_sums[cls_idx] = weights[mask].sum()

        return num_classes, counts, weight_sums

    def _log_class_distribution(self, name: str, dataset: EvenetTensorDataset) -> None:
        if not self.is_rank_zero():
            return
        num_classes, counts, weight_sums = self._class_stats(dataset)
        total = counts.sum().item() or 1.0
        total_weight = weight_sums.sum().item() or 1.0
        class_names = self.class_labels or [str(i) for i in range(num_classes)]
        parts = []
        for idx in range(num_classes):
            label = class_names[idx] if idx < len(class_names) else str(idx)
            frac = counts[idx].item() / total
            w_frac = weight_sums[idx].item() / total_weight
            parts.append(
                f"{label}: count={counts[idx].item()} (frac={frac:.3f}), weight_sum={weight_sums[idx].item():.3f} (frac={w_frac:.3f})"
            )
        logging.info("Class distribution for %s -> %s", name, " | ".join(parts))

    def _log_training_overview(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            train_sampler: Optional[Any],
            val_sampler: Optional[Any],
            epochs: int,
    ) -> None:
        if not self.is_rank_zero():
            return
        logging.info(
            "Training setup: epochs=%d, batch_size=%d, world_size=%d, device=%s",
            epochs,
            train_loader.batch_size,
            self.world_size,
            self.device,
        )
        logging.info(
            "Train loader: size=%d, steps_per_epoch=%d, sampler=%s",
            len(train_loader.dataset),
            len(train_loader),
            self._describe_sampler(train_sampler),
        )
        self._log_class_distribution("train", train_loader.dataset)
        if val_loader is not None:
            logging.info(
                "Val loader: size=%d, steps_per_epoch=%d, sampler=%s",
                len(val_loader.dataset),
                len(val_loader),
                self._describe_sampler(val_sampler),
            )
            self._log_class_distribution("val", val_loader.dataset)
        else:
            logging.info("Validation loader: None")

    def _setup_optimizers_and_schedulers(self, epochs: int, steps_per_epoch: int) -> None:
        self.optimizers, self.schedulers, self.optimizer_tags = build_optimizers_and_schedulers(
            self.model,
            self.config,
            epochs,
            world_size=self.world_size,
            steps_per_epoch=steps_per_epoch,
        )
        self.optimizer = self.optimizers[0] if self.optimizers else None
        self.scheduler = self.schedulers[0] if self.schedulers else None

    def train(
            self,
            train_data: Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]],
            val_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
            test_data: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
            epochs: int,
            batch_size: int,
            sampler: Optional[str],
            epoch_size: Optional[int] = None,
    ) -> None:
        try:
            self.global_step = 0
            self.setup_datasets(train_data, val_data, test_data)
            # Insert normalization callback by default
            if not any(isinstance(cb, NormalizationCallback) for cb in self.callbacks):
                self.callbacks.insert(0, NormalizationCallback())

            self.model = self._maybe_wrap_ddp()
            sampler_obj = build_sampler(
                sampler,
                self.train_dataset,
                self.train_dataset.sample_weights,
                epoch_size,
            )
            if sampler_obj is None and self.world_size > 1:
                sampler_obj = DistributedSampler(self.train_dataset)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler_obj,
                shuffle=sampler_obj is None,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=False,
            )

            val_loader = None
            val_sampler = None
            if self.val_dataset is not None:
                val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.world_size > 1 else None
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=batch_size,
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                )

            self.train_sampler = sampler_obj
            self.val_sampler = val_loader.sampler if val_loader else None
            self.train_loader = train_loader
            self.val_loader = val_loader

            steps_per_epoch = max(1, len(train_loader))

            # set peft
            if getattr(self.config, "use_peft", True):
                set_peft_trainable(self.model, train_layernorm=True)
                if self.is_rank_zero():
                    print_trainable(self.model)

            self._setup_optimizers_and_schedulers(epochs, steps_per_epoch)

            if self.config.resume_from:
                self.restore_checkpoint(self.config.resume_from)

            for cb in self.callbacks:
                cb.on_train_start(self)

            self._log_training_overview(train_loader, val_loader, sampler_obj, val_sampler, epochs)

            best_metric: Optional[float] = None
            best_model_state: Optional[Dict[str, torch.Tensor]] = None
            best_epoch: Optional[int] = None
            epochs_since_improve = 0

            for epoch in range(epochs):
                if isinstance(sampler_obj, (DistributedSampler, DistributedWeightedSampler)):
                    sampler_obj.set_epoch(epoch)
                if isinstance(val_loader, DataLoader) and isinstance(val_loader.sampler, DistributedSampler):
                    val_loader.sampler.set_epoch(epoch)

                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)

                train_metrics = self._run_epoch(self.model, train_loader, epoch, training=True)
                if val_loader is not None:
                    val_metrics = self._run_epoch(self.model, val_loader, epoch, training=False)
                else:
                    val_metrics = {}

                if self.is_rank_zero():
                    merged = {f"train_{k}": v for k, v in train_metrics.items()}
                    merged.update({f"val_{k}": v for k, v in val_metrics.items()})
                    self._log_epoch_stdout(epoch, epochs, merged)
                    if self.wandb_run is not None:
                        wandb_payload = {
                            "epoch": epoch + 1,
                            **self._format_wandb_epoch_metrics(train_metrics, val_metrics),
                        }
                        # self.wandb_run.log(wandb_payload)
                        self.wandb_run.log(wandb_payload, step=self.global_step)
                else:
                    merged = {}

                if self.config.checkpoint_path and self.is_rank_zero():
                    self._save_epoch_checkpoint(merged, epoch)

                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, merged)

                stop_training = False
                metric_value = merged.get(self.config.early_stop_metric) if self.is_rank_zero() else None
                improved = False
                if metric_value is not None:
                    if best_metric is None:
                        improved = True
                    elif self.config.early_stop_minimize:
                        improved = metric_value < best_metric
                    else:
                        improved = metric_value > best_metric

                if improved:
                    best_metric = metric_value
                    best_epoch = epoch
                    epochs_since_improve = 0
                    if self.is_rank_zero():
                        best_model_state = {
                            k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                            for k, v in self._unwrap_model().state_dict().items()
                        }
                elif (
                        self.config.early_stop_patience > 0
                        and best_metric is not None
                        and metric_value is not None
                ):
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.config.early_stop_patience:
                        stop_training = True

                stop_tensor = torch.tensor(1 if stop_training else 0, device=self.device)
                if self.world_size > 1:
                    dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    logging.info(
                        "Early stopping triggered after %d epochs without improvement on %s",
                        self.config.early_stop_patience,
                        self.config.early_stop_metric,
                    )
                    break

            # ALL ranks must participate in the broadcast inside _load_model_state
            self._load_model_state(best_model_state)

            if self.is_rank_zero() and best_model_state is not None:
                logging.info(
                    "Restored best model from epoch %d based on %s=%.4f",
                    (best_epoch or 0) + 1,
                    self.config.early_stop_metric,
                    best_metric if best_metric is not None else float("nan"),
                )

            for cb in self.callbacks:
                cb.on_train_end(self)

            logging.info("Training finished")

            if self.test_dataset is not None:
                eval_batch_size = self.config.eval_batch_size or batch_size
                logging.info("Evaluating on test set")
                eval_metrics = self.evaluate(
                    self.test_dataset,
                    batch_size=eval_batch_size,
                    output_path=self.config.eval_output_path,
                )
                if self.is_rank_zero():
                    logging.info(
                        "Evaluation finished for test split; metrics available under keys: %s",
                        ", ".join(sorted(eval_metrics.keys())),
                    )

                    allowed_keys = {
                        "auc",
                        "max_sic",
                        "max_sic_unc",
                        "accuracy",
                        "loss",
                    }

                    logging.info(
                        "Evaluation metrics (test split):\n%s",
                        format_metrics_for_logging(
                            {k: eval_metrics[k] for k in allowed_keys if k in eval_metrics},
                        ),
                    )
        finally:
            self._finalize_training()

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        normalizer_state = self._current_normalizer_state()
        extra_payload = dict(extra or {})
        if self.schedulers:
            scheduler_states = [s.state_dict() for s in self.schedulers if hasattr(s, "state_dict")]
            if scheduler_states:
                extra_payload["schedulers"] = scheduler_states if len(scheduler_states) > 1 else scheduler_states[0]
        logging.info(
            "Saving checkpoint to %s (includes normalizer=%s, extra keys=%s)",
            path,
            normalizer_state is not None,
            list(extra_payload.keys()),
        )
        if len(self.optimizers) > 1:
            optimizer_state = [opt.state_dict() for opt in self.optimizers]
        elif self.optimizer is not None:
            optimizer_state = self.optimizer.state_dict()
        else:
            optimizer_state = {}
        save_checkpoint(
            path,
            model_state=self._unwrap_model().state_dict(),
            optimizer_state=optimizer_state,
            normalizer_state=normalizer_state,
            extra=extra_payload,
        )

    def _extract_monitored_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        if not metrics:
            return None
        value = metrics.get(self.config.monitor_metric)
        if value is None:
            return None
        if isinstance(value, float):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _finalize_training(self) -> None:
        if self.wandb_run is not None and self.is_rank_zero():
            try:
                self.wandb_run.finish()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logging.warning("Failed to close Weights & Biases run cleanly: %s", exc)
            finally:
                self.wandb_run = None

        if dist.is_available() and dist.is_initialized():
            try:
                self._distributed_barrier()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logging.warning("Distributed barrier failed during shutdown: %s", exc)
            try:
                dist.destroy_process_group()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logging.warning("Failed to destroy process group cleanly: %s", exc)

    def _should_replace_worst(self, metric: float) -> bool:
        if len(self._best_checkpoints) < self.config.save_top_k:
            return True
        worst_metric, _ = self._worst_checkpoint()
        if self.config.minimize_metric:
            return metric < worst_metric
        return metric > worst_metric

    def _worst_checkpoint(self) -> Tuple[float, str]:
        key_fn = (lambda item: item[0]) if self.config.minimize_metric else (lambda item: -item[0])
        return max(self._best_checkpoints, key=key_fn)

    def _ensure_dir_like_base(self, base: Path) -> Path:
        if base.exists() and base.is_file():
            raise ValueError(
                f"Checkpoint path {base} is a file but is treated as a directory; "
                "please provide a filename with an extension or remove the file."
            )
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _checkpoint_filename(self, base: Path, epoch: int, metric: Optional[float]) -> Path:
        suffix = base.suffix or ".pt"
        stem = base.stem if base.suffix else (base.name or "checkpoint")
        base_is_dir = base.is_dir() or base.suffix == ""

        if base_is_dir:
            base = self._ensure_dir_like_base(base)

        if metric is None:
            filename = f"{stem}-epoch{epoch + 1:04d}{suffix}"
            return base / filename if base_is_dir else base.with_name(filename)

        safe_metric = self.config.monitor_metric.replace("/", "_")
        filename = f"{stem}-{safe_metric}-epoch{epoch + 1:04d}-{metric:.4f}{suffix}"
        if base_is_dir:
            return base / filename
        return base.with_name(filename)

    def _save_epoch_checkpoint(self, metrics: Dict[str, float], epoch: int) -> None:
        if self.config.save_top_k > 0:
            self._maybe_save_best_checkpoint(metrics, epoch)
            return

        if (epoch + 1) % max(1, self.config.checkpoint_every) != 0:
            return

        extra = {"epoch": epoch}
        periodic_path = self._checkpoint_filename(Path(self.config.checkpoint_path), epoch, metric=None)
        self.save_checkpoint(str(periodic_path), extra)

    def _maybe_save_best_checkpoint(self, metrics: Dict[str, float], epoch: int) -> bool:
        metric_value = self._extract_monitored_metric(metrics)
        if metric_value is None or not self.config.checkpoint_path:
            return False

        if not self._should_replace_worst(metric_value):
            return False

        checkpoint_base = Path(self.config.checkpoint_path)
        checkpoint_path = self._checkpoint_filename(checkpoint_base, epoch, metric_value)

        extra = {"epoch": epoch, "monitored_metric": metric_value}
        self.save_checkpoint(str(checkpoint_path), extra)
        logging.info(
            "Saved top-k checkpoint at %s for %s=%.4f (k=%d/%d)",
            checkpoint_path,
            self.config.monitor_metric,
            metric_value,
            len(self._best_checkpoints) + 1,
            self.config.save_top_k,
        )

        self._best_checkpoints.append((metric_value, str(checkpoint_path)))
        if len(self._best_checkpoints) > self.config.save_top_k:
            worst_metric, worst_path = self._worst_checkpoint()
            try:
                Path(worst_path).unlink(missing_ok=True)
            except OSError:
                pass
            logging.info(
                "Removed checkpoint %s to maintain top-k=%d (dropped metric %.4f)",
                worst_path,
                self.config.save_top_k,
                worst_metric,
            )
            self._best_checkpoints = [(m, p) for m, p in self._best_checkpoints if (m, p) != (worst_metric, worst_path)]

        return True

    def restore_checkpoint(self, path: str, map_location: Optional[str] = None) -> None:
        logging.info("Restoring checkpoint from %s", path)
        checkpoint = load_checkpoint(path, map_location=map_location)
        self._unwrap_model().load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            opt_state = checkpoint["optimizer"]
            if isinstance(opt_state, list) and self.optimizers:
                for optimizer, state in zip(self.optimizers, opt_state):
                    optimizer.load_state_dict(state)
            elif self.optimizer is not None and isinstance(opt_state, dict):
                self.optimizer.load_state_dict(opt_state)
        if "normalizer" in checkpoint and checkpoint["normalizer"]:
            self.attach_normalizer_state(checkpoint["normalizer"])
            logging.info("Restored normalizer state from checkpoint")
        if "extra" in checkpoint:
            scheduler_state = checkpoint["extra"].get("schedulers") if checkpoint.get("extra") else None
            if scheduler_state:
                states = scheduler_state if isinstance(scheduler_state, list) else [scheduler_state]
                for scheduler, state in zip(self.schedulers, states):
                    if scheduler:
                        scheduler.load_state_dict(state)

    def _forward(self, model: torch.nn.Module, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            return model(**features)
        except TypeError:
            return model(features)

    def _maybe_concat_parameters(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "params" not in features:
            return features

        merged = dict(features)
        params = merged.pop("params")
        globals_tensor = merged.get("globals")
        if globals_tensor is None:
            merged["globals"] = params
            return merged

        if globals_tensor.shape[0] != params.shape[0]:
            logging.warning(
                "Batch globals and params have mismatched batch sizes (%d vs %d); skipping parameter concatenation.",
                globals_tensor.shape[0],
                params.shape[0],
            )
            return merged

        merged["globals"] = torch.cat([globals_tensor, params], dim=-1)
        return merged

    def _warn_if_global_dim_mismatch(self, globals_tensor: torch.Tensor) -> None:
        model = self._unwrap_model()
        expected_dim = getattr(model, "global_input_dim", None)
        if expected_dim is None or hasattr(self, "_warned_global_dim") and self._warned_global_dim:
            return
        actual_dim = globals_tensor.shape[-1]
        if actual_dim != expected_dim:
            logging.error(
                "Global feature dimension (%d) does not match model expectation (%d). "
                "If you added parameterized inputs, update global_input_dim accordingly.",
                actual_dim,
                expected_dim,
            )
            self._warned_global_dim = True

            exit(1)

    def _prepare_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        merged_features = self._maybe_concat_parameters(features)
        prepared: Dict[str, torch.Tensor] = {}
        for name, tensor in merged_features.items():
            tensor = tensor.to(self.device)
            if name in {"x", "globals"}:
                tensor = tensor.float()
            prepared[name] = tensor

        if "globals" in prepared:
            self._warn_if_global_dim_mismatch(prepared["globals"])
        return prepared

    def _set_model_mode(self, model: torch.nn.Module, training: bool) -> None:
        if not training:
            model.eval()
            return

        model.train()
        return
        #
        # if not getattr(self.config, "use_peft", False):
        #     return
        #
        # # PEFT: backbone eval, adapters/head train
        #
        #
        # m = self._unwrap_model()
        #
        #
        # if hasattr(m, "PET"):
        #     m.PET.eval()
        #     if hasattr(m.PET, "adapters"):
        #         m.PET.adapters.train()
        # for mod in m.modules():
        #     if isinstance(mod, torch.nn.LayerNorm):
        #         mod.train()

    def _run_epoch(
            self,
            model: torch.nn.Module,
            loader: DataLoader,
            epoch: int,
            training: bool,
    ) -> Dict[str, float]:
        # if training:
        #     model.train()
        # else:
        #     model.eval()
        self._set_model_mode(model, training)

        metric_sum: Dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
        metric_count: Dict[str, int] = {"loss": 0, "accuracy": 0}
        physics_probs: List[torch.Tensor] = []
        physics_targets: List[torch.Tensor] = []
        physics_weights: List[torch.Tensor] = []
        collect_physics = self.config.compute_physics_metrics and self.num_classes == 2
        classification_confusion: Optional[torch.Tensor] = None
        classification_entries: Optional[torch.Tensor] = None
        score_histograms: Optional[torch.Tensor] = None
        score_histograms_w2: Optional[torch.Tensor] = None
        score_bins = max(1, int(self.config.classification_score_bins))
        if self.num_classes is not None:
            classification_confusion = torch.zeros(
                self.num_classes * self.num_classes,
                dtype=torch.float64,
                device=self.device,
            )
            classification_entries = torch.zeros_like(classification_confusion)
            score_histograms = torch.zeros(
                self.num_classes * self.num_classes * score_bins,
                dtype=torch.float64,
                device=self.device,
            )
            score_histograms_w2 = torch.zeros_like(score_histograms)

        metric_tracker = self.train_accuracy if training else self.val_accuracy
        if metric_tracker is not None:
            metric_tracker.reset()

        progress = None
        if self.is_rank_zero():
            try:
                from tqdm.auto import tqdm  # type: ignore

                progress = tqdm(
                    total=len(loader),
                    desc=f"{'Train' if training else 'Val'} Epoch {epoch + 1}",
                    leave=False,
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.debug("Progress bar unavailable: %s", exc)

        for batch_idx, (features, targets, weights) in enumerate(loader):
            batch_payload = {"features": features, "targets": targets, "weights": weights}
            for cb in self.callbacks:
                cb.on_batch_start(self, epoch, batch_idx, batch_payload, training)

            features = self._prepare_features(batch_payload["features"])
            targets = batch_payload["targets"].long().to(self.device)
            weights = batch_payload["weights"]
            weight_tensor: Optional[torch.Tensor] = None
            if weights is not None:
                weights = weights.to(self.device)
                finite_mask = torch.isfinite(weights)
                # Zero-out any non-finite weights to avoid NaNs in the loss
                weight_tensor = torch.where(finite_mask, weights, torch.zeros_like(weights))
                if not torch.all(finite_mask):
                    logging.debug("Non-finite weights detected; treating them as zero during loss computation.")
            with torch.set_grad_enabled(training):
                outputs = self._forward(model, features)
                if not torch.isfinite(outputs).all():
                    logging.debug("Non-finite outputs detected; treating them as zero."
                                  f"[Rank {self.rank}] Non-finite logits detected\n"
                                  f"min={outputs.min().item()}, max={outputs.max().item()}"
                                  )

                weighted_sampler = isinstance(loader.sampler, DistributedWeightedSampler)
                weight_tensor_input = weight_tensor if not weighted_sampler else torch.ones_like(weight_tensor)
                loss = compute_loss(outputs, targets, weight_tensor_input, gamma=self.config.loss_gamma)
                if training:
                    optimizers = self.optimizers or ([self.optimizer] if self.optimizer else [])
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    if not torch.isfinite(loss).all().item():
                        print("[Rank %d] Non-finite loss detected; skipping backward step." % self.rank)
                        return
                    loss.backward()
                    if self.config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        if scheduler:
                            scheduler.step()

            loss_value = loss.detach()

            if torch.isfinite(loss_value):
                metric_sum["loss"] += loss_value.item() * targets.size(0)
                metric_count["loss"] += targets.size(0)
            # else:
            #     # Optional: track how often this happens
            #     metric_sum["nan_loss"] += 1
            # metric_count["loss"] += targets.size(0)

            logits_for_metrics = outputs.mean(dim=0) if outputs.dim() == 3 else outputs
            preds = torch.argmax(logits_for_metrics, dim=1)
            batch_accuracy = compute_accuracy(outputs, targets)
            if metric_tracker is not None:
                metric_tracker.update(preds, targets)
            else:
                metric_sum["accuracy"] += batch_accuracy * targets.size(0)
                metric_count["accuracy"] += targets.size(0)

            reduced_loss = self._reduce_mean_scalar(loss.item())
            reduced_accuracy = self._reduce_mean_scalar(batch_accuracy)

            metric_weights = (
                weight_tensor.detach()
                if weight_tensor is not None
                else torch.ones_like(targets, dtype=torch.float32, device=self.device)
            )
            if classification_confusion is not None:
                valid = (
                    (targets >= 0)
                    & (targets < self.num_classes)
                    & (preds >= 0)
                    & (preds < self.num_classes)
                    & torch.isfinite(metric_weights)
                    & (metric_weights >= 0)
                )
                if torch.any(valid):
                    encoded = targets[valid] * self.num_classes + preds[valid]
                    classification_confusion += torch.bincount(
                        encoded,
                        weights=metric_weights[valid].to(torch.float64),
                        minlength=self.num_classes * self.num_classes,
                    )
                    classification_entries += torch.bincount(
                        encoded,
                        minlength=self.num_classes * self.num_classes,
                    ).to(torch.float64)

            if score_histograms is not None and score_histograms_w2 is not None:
                probabilities = self._classification_probabilities_tensor(logits_for_metrics.detach().float())
                score_classes = min(self.num_classes, probabilities.shape[1])
                valid = (
                    (targets >= 0)
                    & (targets < self.num_classes)
                    & torch.isfinite(metric_weights)
                    & (metric_weights >= 0)
                )
                if score_classes > 0 and torch.any(valid):
                    scores = probabilities[valid, :score_classes].clamp(0.0, 1.0)
                    bins = torch.clamp((scores * score_bins).long(), max=score_bins - 1)
                    true_offset = targets[valid].view(-1, 1) * self.num_classes * score_bins
                    score_offset = torch.arange(score_classes, device=self.device).view(1, -1) * score_bins
                    encoded = true_offset + score_offset + bins
                    hist_weights = metric_weights[valid].to(torch.float64).view(-1, 1).expand_as(scores)
                    score_histograms += torch.bincount(
                        encoded.reshape(-1),
                        weights=hist_weights.reshape(-1),
                        minlength=self.num_classes * self.num_classes * score_bins,
                    )
                    score_histograms_w2 += torch.bincount(
                        encoded.reshape(-1),
                        weights=(hist_weights ** 2).reshape(-1),
                        minlength=self.num_classes * self.num_classes * score_bins,
                    )

            if collect_physics:
                physics_probs.append(logits_for_metrics.detach())
                physics_targets.append(targets.detach())
                physics_weights.append(metric_weights.detach())

            batch_metrics = {"loss": loss.item(), "accuracy": batch_accuracy}
            for cb in self.callbacks:
                cb.on_batch_end(
                    self, epoch, batch_idx, {"features": features, "targets": targets}, loss.item(), batch_metrics
                )

            if training:
                self.global_step += 1
                self._log_train_step(self.global_step, reduced_loss, reduced_accuracy, epoch)

            if progress is not None:
                progress.set_postfix({"loss": f"{reduced_loss:.4f}"}, refresh=False)
                progress.update(1)

        if progress is not None:
            progress.close()
            progress = None
        if self.is_rank_zero():
            stage = "train" if training else "validation"
            logging.info("Computing %s epoch metrics from full entries", stage)

        metric_sum_tensor = torch.tensor([metric_sum["loss"], metric_sum["accuracy"]], device=self.device)
        metric_count_tensor = torch.tensor([metric_count["loss"], metric_count["accuracy"]], device=self.device)
        metric_sum_tensor = self._all_reduce_tensor(metric_sum_tensor)
        metric_count_tensor = self._all_reduce_tensor(metric_count_tensor)

        metric_sum["loss"], metric_sum["accuracy"] = metric_sum_tensor.tolist()
        metric_count["loss"] = int(metric_count_tensor[0].item())
        metric_count["accuracy"] = int(metric_count_tensor[1].item())

        metrics = summarize_metrics(metric_sum, metric_count)
        if metric_tracker is not None:
            metrics["accuracy"] = float(metric_tracker.compute().item())

        confusion_matrix = None
        entries_matrix = None
        if classification_confusion is not None:
            classification_confusion = self._all_reduce_tensor(classification_confusion)
            if self.is_rank_zero():
                confusion_matrix = classification_confusion.reshape(self.num_classes, self.num_classes).cpu().numpy()
        if classification_entries is not None:
            classification_entries = self._all_reduce_tensor(classification_entries)
            if self.is_rank_zero():
                entries_matrix = classification_entries.reshape(self.num_classes, self.num_classes).cpu().numpy()

        score_histograms_np = None
        score_histograms_w2_np = None
        if score_histograms is not None and score_histograms_w2 is not None:
            score_histograms = self._all_reduce_tensor(score_histograms)
            score_histograms_w2 = self._all_reduce_tensor(score_histograms_w2)
            if self.is_rank_zero():
                score_histograms_np = score_histograms.reshape(
                    self.num_classes,
                    self.num_classes,
                    score_bins,
                ).cpu().numpy()
                score_histograms_w2_np = score_histograms_w2.reshape(
                    self.num_classes,
                    self.num_classes,
                    score_bins,
                ).cpu().numpy()

        if confusion_matrix is not None:
            metrics.update(
                self._compute_epoch_classification_metrics(
                    confusion_matrix=confusion_matrix,
                    entries_matrix=entries_matrix,
                    score_histograms=score_histograms_np,
                    score_bins=score_bins,
                    training=training,
                )
            )

        if self.config.compute_physics_metrics and self.num_classes and self.num_classes > 2 and score_histograms_np is not None:
            metrics.update(
                self._compute_epoch_multiclass_physics_metrics(
                    score_histograms_np,
                    score_histograms_w2_np,
                    training=training,
                )
            )

        if self.config.compute_physics_metrics and physics_probs and self.num_classes == 2:
            metrics.update(
                self._compute_epoch_physics_metrics(physics_probs, physics_targets, physics_weights, training=training)
            )
        return metrics

    def _log_train_step(self, step: int, loss: float, accuracy: float, epoch: int) -> None:
        if self.wandb_run is None or not self.is_rank_zero():
            return
        self.wandb_run.log(
            {
                "train/loss": loss,
                "metric-Accuracy/train_step": accuracy,
                "epoch": epoch + 1,
                **self._optimizer_learning_rates(),
            },
            step=step,
        )

    @staticmethod
    def _wandb_metric_key(prefix: str, name: str) -> str:
        if name == "loss":
            return f"{prefix}/loss"
        groups = [
            ("weighted_auc", "metric-AUC", "weighted"),
            ("macro_auc", "metric-AUC", "macro"),
            ("auc", "metric-AUC", "overall"),
            ("auc_", "metric-AUC", None),
            ("weighted_recall", "metric-Recall", "weighted"),
            ("macro_recall", "metric-Recall", "macro"),
            ("recall_", "metric-Recall", None),
            ("weighted_precision", "metric-Precision", "weighted"),
            ("macro_precision", "metric-Precision", "macro"),
            ("precision_", "metric-Precision", None),
            ("weighted_f1", "metric-F1", "weighted"),
            ("macro_f1", "metric-F1", "macro"),
            ("f1_", "metric-F1", None),
            ("support_", "metric-Support", None),
            ("balanced_accuracy", "metric-Accuracy", "balanced"),
            ("accuracy", "metric-Accuracy", "overall"),
            ("max_sic_unc_", "metric-SIC-unc", None),
            ("max_sic_unc", "metric-SIC-unc", "best"),
            ("max_sic_", "metric-SIC", None),
            ("max_sic", "metric-SIC", "best"),
            ("trafo_bin_sig_", "metric-Trafo", None),
            ("trafo_bin_sig", "metric-Trafo", "best"),
        ]
        for token, group, fixed_suffix in groups:
            if name == token:
                return f"{group}/{prefix}_{fixed_suffix}"
            if token.endswith("_") and name.startswith(token):
                return f"{group}/{prefix}_{name[len(token):]}"
        return f"metric-Other/{prefix}_{name}"

    def _format_metric_group(self, metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
        return {self._wandb_metric_key(prefix, name): value for name, value in metrics.items()}

    def _format_wandb_epoch_metrics(
            self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        payload.update(self._format_metric_group(train_metrics, "train"))
        if val_metrics:
            payload.update(self._format_metric_group(val_metrics, "val"))
        return payload

    def _optimizer_learning_rates(self) -> Dict[str, float]:
        lr_logs: Dict[str, float] = {}
        for tag, optimizer in zip(self.optimizer_tags, self.optimizers):
            if optimizer is None:
                continue
            lrs = [group.get("lr") for group in optimizer.param_groups if "lr" in group]
            for idx, lr in enumerate(lrs):
                if lr is None:
                    continue
                key = f"Optimizer/{tag}-lr" if len(lrs) == 1 else f"Optimizer/{tag}-lr-{idx}"
                lr_logs[key] = float(lr)
        return lr_logs

    def _physics_metric_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "bins": self.config.physics_bins,
            "min_bkg_events": 100,
        }
        for key, value in (self.config.physics_metric_config or {}).items():
            if key not in {"SIC_base", "sic_base"}:
                kwargs[key] = value
        return kwargs

    def _sic_base_indices(self) -> List[int]:
        raw = (self.config.physics_metric_config or {}).get("SIC_base")
        if raw is None:
            raw = (self.config.physics_metric_config or {}).get("sic_base")
        if not raw:
            return []
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        label_to_idx = {label: idx for idx, label in enumerate(self.class_labels or [])}
        indices: List[int] = []
        for item in raw:
            idx = label_to_idx[item] if isinstance(item, str) else int(item)
            if self.num_classes is None or idx < 0 or idx >= self.num_classes:
                raise ValueError(f"SIC_base class {item!r} is outside class_labels")
            indices.append(idx)
        return sorted(set(indices))

    def _metric_label(self, label: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "class"

    def _classification_scalar_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        scalar_keys = [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "macro_auc",
            "weighted_auc",
        ]
        scalars = {key: float(metrics[key]) for key in scalar_keys if key in metrics}
        class_names = self.class_labels or [str(i) for i in range(len(metrics.get("class_support", [])))]
        for idx, label in enumerate(class_names):
            suffix = self._metric_label(label)
            for metric_key, prefix in [
                ("class_precision", "precision"),
                ("class_recall", "recall"),
                ("class_f1", "f1"),
                ("class_support", "support"),
                ("class_auc", "auc"),
            ]:
                values = metrics.get(metric_key)
                if values is None or idx >= len(values) or not np.isfinite(values[idx]):
                    continue
                scalars[f"{prefix}_{suffix}"] = float(values[idx])
        return scalars

    def _log_classification_plots(
            self,
            metric_arrays: Dict[str, Any],
            stage: str,
            probs: Optional[np.ndarray] = None,
            targets: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None,
    ) -> None:
        if self.wandb_run is None or not self.is_rank_zero():
            return
        try:
            import wandb
            from .plots import (
                close_figure,
                plot_confusion_matrix,
                plot_rejection_curves,
                plot_rejection_curves_from_histograms,
                plot_score_distributions,
                plot_score_distributions_from_histograms,
            )
        except Exception as exc:  # pragma: no cover - optional plotting dependency
            logging.warning("Skipping classification plots: %s", exc)
            return

        figures = {
            f"Classification/{stage}-confusion": plot_confusion_matrix(
                metric_arrays["confusion_matrix"],
                self.class_labels,
                entries_matrix=metric_arrays.get("confusion_entries"),
                normalize=True,
                title=f"{stage.title()} confusion matrix",
            ),
        }
        if "score_histograms" in metric_arrays:
            figures[f"Classification/{stage}-background-rejection"] = plot_rejection_curves_from_histograms(
                metric_arrays["score_histograms"],
                self.class_labels,
                title=f"{stage.title()} one-vs-rest background rejection",
            )
            figures[f"Classification/{stage}-score-distributions"] = plot_score_distributions_from_histograms(
                metric_arrays["score_histograms"],
                self.class_labels,
                bin_edges=metric_arrays.get("score_bin_edges"),
                title=f"{stage.title()} score distributions by true class",
            )
        elif probs is not None and targets is not None and weights is not None:
            figures[f"Classification/{stage}-background-rejection"] = plot_rejection_curves(
                probs,
                targets,
                weights,
                self.class_labels,
                title=f"{stage.title()} one-vs-rest background rejection",
            )
            figures[f"Classification/{stage}-score-distributions"] = plot_score_distributions(
                probs,
                targets,
                weights,
                self.class_labels,
                title=f"{stage.title()} score distributions by true class",
            )
        else:
            logging.warning("Skipping classification score plots for %s: no score data available", stage)
        try:
            self.wandb_run.log(
                {name: wandb.Image(fig) for name, fig in figures.items()},
                step=self.global_step,
            )
        finally:
            for fig in figures.values():
                close_figure(fig)

    def _compute_epoch_classification_metrics(
            self,
            confusion_matrix: Optional[np.ndarray],
            entries_matrix: Optional[np.ndarray],
            score_histograms: Optional[np.ndarray],
            score_bins: int,
            training: bool,
    ) -> Dict[str, float]:
        if not self.is_rank_zero():
            return {}
        if confusion_matrix is None:
            return {}

        metric_arrays = classification_metrics_from_confusion_matrix(confusion_matrix, entries_matrix)
        if score_histograms is not None:
            class_auc = classification_auc_from_score_histograms(score_histograms)
            finite_auc = np.isfinite(class_auc)
            support = metric_arrays["class_support"]
            metric_arrays.update({
                "class_auc": class_auc,
                "macro_auc": float(np.mean(class_auc[finite_auc])) if np.any(finite_auc) else 0.5,
                "weighted_auc": float(np.sum(class_auc[finite_auc] * support[finite_auc]) / support[finite_auc].sum())
                if np.any(finite_auc) and support[finite_auc].sum() > 0 else 0.5,
                "score_histograms": score_histograms,
                "score_bin_edges": np.linspace(0.0, 1.0, score_bins + 1),
            })
        else:
            metric_arrays.update({
                "class_auc": np.full(confusion_matrix.shape[0], np.nan, dtype=float),
                "macro_auc": 0.5,
                "weighted_auc": 0.5,
            })

        self._log_classification_plots(metric_arrays, "train" if training else "valid")
        return self._classification_scalar_metrics(metric_arrays)

    def _write_sic_summary_plot(
            self,
            sic_result: Dict[str, np.ndarray],
            stage: str,
            label: str,
            save_path: Optional[Path] = None,
    ) -> None:
        if (self.wandb_run is None and save_path is None) or not self.is_rank_zero():
            return
        try:
            from .plots import close_figure, plot_sic_summary
            wandb = None
            if self.wandb_run is not None:
                import wandb as wandb_module
                wandb = wandb_module
        except Exception as exc:  # pragma: no cover - optional plotting dependency
            logging.warning("Skipping SIC plot for %s: %s", label, exc)
            return
        fig = plot_sic_summary(sic_result, title=f"{stage.title()} SIC: {label} vs SIC_base")
        try:
            if save_path is not None:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"Physics/{stage}-SIC-{self._metric_label(label)}": wandb.Image(fig)},
                    step=self.global_step,
                )
        finally:
            close_figure(fig)

    def _compute_epoch_multiclass_physics_metrics(
            self,
            score_histograms: np.ndarray,
            score_histograms_w2: Optional[np.ndarray],
            training: bool,
    ) -> Dict[str, float]:
        base_indices = self._sic_base_indices()
        if not base_indices or self.num_classes is None:
            return {}
        kwargs = self._physics_metric_kwargs()
        min_bkg_events = int(kwargs["min_bkg_events"])
        min_bkg_ratio = kwargs.get("min_bkg_ratio")
        class_names = self.class_labels or [str(i) for i in range(self.num_classes)]
        stage = "train" if training else "valid"
        metrics: Dict[str, float] = {}
        best_sic = 0.0
        best_sic_unc = 0.0
        for sig_idx in range(self.num_classes):
            if sig_idx in base_indices:
                continue
            sig_hist = score_histograms[sig_idx, sig_idx]
            bkg_hist = score_histograms[base_indices, sig_idx].sum(axis=0)
            bkg_w2_hist = (
                score_histograms_w2[base_indices, sig_idx].sum(axis=0)
                if score_histograms_w2 is not None
                else bkg_hist
            )
            sic_result = compute_sic_from_score_histograms(
                sig_hist,
                bkg_hist,
                bkg_w2_hist,
                min_bkg_events=min_bkg_events,
                min_bkg_ratio=min_bkg_ratio,
            )
            label = class_names[sig_idx] if sig_idx < len(class_names) else str(sig_idx)
            suffix = self._metric_label(label)
            metrics[f"max_sic_{suffix}"] = float(sic_result["max_sic"])
            metrics[f"max_sic_unc_{suffix}"] = float(sic_result["max_sic_unc"])
            auc = float(np.trapz(sic_result["sig_eff"], sic_result["bkg_eff"]))
            metrics[f"auc_{suffix}"] = auc
            if sic_result["max_sic"] > best_sic:
                best_sic = float(sic_result["max_sic"])
                best_sic_unc = float(sic_result["max_sic_unc"])
            self._write_sic_summary_plot(sic_result, stage, label)
        if metrics:
            metrics["max_sic"] = best_sic
            metrics["max_sic_unc"] = best_sic_unc
        return metrics

    def _compute_multiclass_physics_metrics_from_arrays(
            self,
            probs: np.ndarray,
            labels: np.ndarray,
            weights: np.ndarray,
            training: bool,
            output_base: Optional[Path] = None,
    ) -> Dict[str, float]:
        base_indices = self._sic_base_indices()
        if not base_indices or self.num_classes is None:
            return {}
        kwargs = self._physics_metric_kwargs()
        bins = int(kwargs.get("bins", self.config.physics_bins))
        min_bkg_events = int(kwargs["min_bkg_events"])
        min_bkg_ratio = kwargs.get("min_bkg_ratio")
        edges = np.linspace(0.0, 1.0, bins + 1)
        class_names = self.class_labels or [str(i) for i in range(self.num_classes)]
        stage = "train" if training else "test"
        base_mask = np.isin(labels, np.asarray(base_indices))
        metrics: Dict[str, float] = {}
        best_sic = 0.0
        best_sic_unc = 0.0
        for sig_idx in range(self.num_classes):
            if sig_idx in base_indices:
                continue
            mask = base_mask | (labels == sig_idx)
            if not np.any(mask):
                continue
            binary_targets = (labels[mask] == sig_idx).astype(int)
            if binary_targets.sum() == 0 or (binary_targets == 0).sum() == 0:
                continue
            scores = probs[mask, sig_idx]
            selected_weights = weights[mask]
            result = calculate_physics_metrics(
                logits=scores,
                targets=binary_targets,
                weights=selected_weights,
                training=training,
                log_plots=False,
                **kwargs,
            )
            sic_result = compute_sic_from_scores(
                binary_targets,
                scores,
                selected_weights,
                edges,
                min_bkg_events=min_bkg_events,
                min_bkg_ratio=min_bkg_ratio,
            )
            label = class_names[sig_idx] if sig_idx < len(class_names) else str(sig_idx)
            suffix = self._metric_label(label)
            metrics[f"auc_{suffix}"] = float(result["auc"])
            metrics[f"max_sic_{suffix}"] = float(result["max_sic"])
            metrics[f"max_sic_unc_{suffix}"] = float(result["max_sic_unc"])
            metrics[f"trafo_bin_sig_{suffix}"] = float(result["trafo_bin_sig"])
            if result["max_sic"] > best_sic:
                best_sic = float(result["max_sic"])
                best_sic_unc = float(result["max_sic_unc"])
            save_path = (
                output_base.with_name(f"{output_base.stem}-sic-{suffix}.png")
                if output_base is not None
                else None
            )
            self._write_sic_summary_plot(sic_result, stage, label, save_path=save_path)
        if metrics:
            metrics["max_sic"] = best_sic
            metrics["max_sic_unc"] = best_sic_unc
        return metrics

    def _compute_epoch_physics_metrics(
            self,
            probs_list: List[torch.Tensor],
            targets_list: List[torch.Tensor],
            weights_list: List[torch.Tensor],
            training: bool,
    ) -> dict[Any, Any] | dict[str, ndarray]:
        probs = self._all_gather_tensor(torch.cat(probs_list).detach())
        targets = self._all_gather_tensor(torch.cat(targets_list).detach())
        weights = self._all_gather_tensor(torch.cat(weights_list).detach())

        if not self.is_rank_zero():
            return {}

        metrics = calculate_physics_metrics(
            logits=probs.cpu().numpy(),
            targets=targets.cpu().numpy(),
            weights=weights.cpu().numpy(),
            training=training,
            log_plots=self.wandb_run is not None,
            wandb_run=self.wandb_run,
            log_step=self.global_step,
            **self._physics_metric_kwargs(),
        )
        return {
            "auc": metrics["auc"],
            "max_sic": metrics["max_sic"],
            "max_sic_unc": metrics["max_sic_unc"],
            "trafo_bin_sig": metrics["trafo_bin_sig"],
        }

    def _save_classification_plots(
            self,
            base_path: Path,
            probs: np.ndarray,
            targets: np.ndarray,
            weights: np.ndarray,
            metric_arrays: Dict[str, Any],
            stage: str,
    ) -> None:
        try:
            from .plots import close_figure, plot_confusion_matrix, plot_rejection_curves, plot_score_distributions
        except Exception as exc:  # pragma: no cover - optional plotting dependency
            logging.warning("Skipping saved classification plots: %s", exc)
            return

        base_path.parent.mkdir(parents=True, exist_ok=True)
        figures = {
            "confusion": plot_confusion_matrix(
                metric_arrays["confusion_matrix"],
                self.class_labels,
                entries_matrix=metric_arrays.get("confusion_entries"),
                normalize=True,
                title=f"{stage.title()} confusion matrix",
            ),
            "background-rejection": plot_rejection_curves(
                probs,
                targets,
                weights,
                self.class_labels,
                title=f"{stage.title()} one-vs-rest background rejection",
            ),
            "score-distributions": plot_score_distributions(
                probs,
                targets,
                weights,
                self.class_labels,
                title=f"{stage.title()} score distributions by true class",
            ),
        }
        try:
            for name, fig in figures.items():
                fig.savefig(base_path.with_name(f"{base_path.stem}-{name}.png"), dpi=300, bbox_inches="tight")
        finally:
            for fig in figures.values():
                close_figure(fig)

    def _log_epoch_stdout(self, epoch: int, total_epochs: int, metrics: Dict[str, float]) -> None:
        msg_parts = [f"Epoch {epoch + 1}/{total_epochs}"]
        for key in [
            "train_loss",
            "train_accuracy",
            "train_balanced_accuracy",
            "train_macro_f1",
            "train_weighted_auc",
            "train_auc",
            "train_max_sic",
            "val_loss",
            "val_accuracy",
            "val_balanced_accuracy",
            "val_macro_f1",
            "val_weighted_auc",
            "val_auc",
            "val_max_sic",

            "train_trafo_bin_sig",
            "val_trafo_bin_sig",
        ]:
            if key in metrics:
                msg_parts.append(f"{key}={metrics[key]:.4f}")
        logging.info(" | ".join(msg_parts))

    def _collect_predictions(
            self, dataset: EvenetTensorDataset, batch_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        logging.info("Starting _collect_predictions")
        logging.info("Dataset size = %d", len(dataset))
        logging.info("Batch size = %d", batch_size)

        original_flag = getattr(dataset, "include_indices", False)
        dataset.include_indices = True

        sampler = DistributedSampler(dataset, shuffle=False) if self.world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        if self.debug:
            logging.info(
                "world_size=%d, sampler=%s, num_workers=%d",
                self.world_size,
                "DistributedSampler" if sampler else "None",
                self.config.num_workers,
            )

        self.model.to(self.device)
        self.model.eval()

        local_outputs: List[torch.Tensor] = []
        local_indices: List[torch.Tensor] = []

        with torch.no_grad():
            for step, batch in enumerate(loader):
                features, _, _, *maybe_idx = batch
                batch_indices = maybe_idx[0] if maybe_idx else None

                if step == 0 and self.debug:
                    logging.info("First batch received")
                    if batch_indices is not None:
                        if self.debug:
                            logging.info(
                                "batch_indices shape=%s, min=%d, max=%d",
                                tuple(batch_indices.shape),
                                batch_indices.min().item(),
                                batch_indices.max().item(),
                            )

                features = self._prepare_features(features)
                outputs = self._forward(self.model, features)

                if step == 0 and self.debug:
                    logging.info("Raw outputs shape = %s", tuple(outputs.shape))

                # Ensemble case: [E, B, C] → [B, C]
                outputs = outputs.mean(dim=0) if outputs.dim() == 3 else outputs
                outputs = outputs.detach().cpu()

                local_outputs.append(outputs)
                if batch_indices is not None:
                    local_indices.append(batch_indices.cpu())

        dataset.include_indices = original_flag

        preds_tensor = torch.cat(local_outputs, dim=0) if local_outputs else torch.empty((0,))
        index_tensor = (
            torch.cat(local_indices, dim=0)
            if local_indices
            else torch.empty((0,), dtype=torch.long)
        )

        if self.debug:
            logging.info(
                "Local outputs: preds=%s, indices=%s",
                tuple(preds_tensor.shape),
                tuple(index_tensor.shape),
            )

        # ------------------------
        # DDP gather
        # ------------------------
        if self.world_size > 1:
            gathered_indices: List[Optional[torch.Tensor]] = [None for _ in range(self.world_size)]
            gathered_preds: List[Optional[torch.Tensor]] = [None for _ in range(self.world_size)]

            if self.debug:
                if dist.get_rank() == 0:
                    logging.info("default pg backend=%s world_size=%d", dist.get_backend(), dist.get_world_size())

                r = dist.get_rank()
                logging.info("[Rank %d] Before barrier", r)
                torch.cuda.synchronize()
                self._distributed_barrier()
                logging.info("[Rank %d] After barrier", r)

                logging.info("rank=%s backend=%s preds_device=%s idx_device=%s",
                             dist.get_rank(), dist.get_backend(),
                             preds_tensor.device, index_tensor.device)

            dist.all_gather_object(gathered_indices, index_tensor)
            dist.all_gather_object(gathered_preds, preds_tensor)

            if self.debug:
                sizes = [
                    gi.shape if gi is not None else None
                    for gi in gathered_indices
                ]
                logging.info("Gathered index shapes per rank = %s", sizes)

            index_tensor = torch.cat([g for g in gathered_indices if g is not None], dim=0)
            preds_tensor = torch.cat([g for g in gathered_preds if g is not None], dim=0)

            if self.debug:
                logging.info(
                    "After gather: preds=%s, indices=%s",
                    tuple(preds_tensor.shape),
                    tuple(index_tensor.shape),
                )

            if index_tensor.numel() > 0:
                order = torch.argsort(index_tensor)
                index_tensor = index_tensor[order]
                preds_tensor = preds_tensor[order]

                unique_mask = torch.ones_like(index_tensor, dtype=torch.bool)
                unique_mask[1:] = index_tensor[1:] != index_tensor[:-1]
                unique_positions = torch.nonzero(unique_mask, as_tuple=False).squeeze(1)

                removed = index_tensor.numel() - unique_positions.numel()
                logging.info("Removed %d duplicated entries", removed)

                index_tensor = index_tensor[unique_positions]
                preds_tensor = preds_tensor[unique_positions]

        logging.info(
            "Finished prediction collection: preds=%s, indices=%s",
            tuple(preds_tensor.shape),
            tuple(index_tensor.shape),
        )

        return preds_tensor, index_tensor

    def predict(self, dataset: EvenetTensorDataset, batch_size: int = 256) -> torch.Tensor:
        preds, _ = self._collect_predictions(dataset, batch_size)
        return preds

    def evaluate(
            self,
            dataset: EvenetTensorDataset,
            batch_size: int = 256,
            output_path: Optional[str] = None,
    ) -> Dict[str, float]:
        preds, indices = self._collect_predictions(dataset, batch_size)

        if not self.is_rank_zero():
            return {}

        labels = dataset.labels[indices] if indices.numel() > 0 else dataset.labels
        weights = dataset.sample_weights[
            indices] if dataset.sample_weights is not None and indices.numel() > 0 else dataset.sample_weights
        raw_features = (
            {name: tensor[indices] for name, tensor in dataset.raw_features.items()}
            if indices.numel() > 0
            else dataset.raw_features
        )

        valid_weight_tensor = None
        if weights is not None:
            finite_mask = torch.isfinite(weights)
            valid_weight_tensor = weights if torch.all(finite_mask) else None

        loss = compute_loss(preds, labels, valid_weight_tensor, gamma=self.config.loss_gamma)
        accuracy = compute_accuracy(preds, labels)
        metrics: Dict[str, Any] = {"loss": float(loss.item()), "accuracy": float(accuracy)}
        weights_np = (
            weights.numpy()
            if weights is not None
            else torch.ones_like(labels, dtype=torch.float32).numpy()
        )
        preds_np = preds.numpy()
        labels_np = labels.numpy()
        class_metric_arrays = compute_classification_metrics(
            logits=preds_np,
            targets=labels_np,
            weights=weights_np,
            class_labels=self.class_labels,
        )
        metrics.update(self._classification_scalar_metrics(class_metric_arrays))
        probs = class_metric_arrays["probabilities"]
        self._log_classification_plots(
            class_metric_arrays,
            "test",
            probs=probs,
            targets=labels_np,
            weights=weights_np,
        )

        resolved_base = self._resolve_eval_base_path(Path(output_path)) if output_path else None
        if resolved_base is not None:
            self._save_classification_plots(resolved_base, probs, labels_np, weights_np, class_metric_arrays, "test")

        if self.config.compute_physics_metrics and preds.numel() > 0 and self.num_classes == 2:
            metrics.update(
                calculate_physics_metrics(
                    logits=preds_np,
                    targets=labels_np,
                    weights=weights_np,
                    training=False,
                    log_plots=self.wandb_run is not None,
                    wandb_run=self.wandb_run,
                    f_name=str(resolved_base.with_name(f"{resolved_base.stem}-sic.png")) if resolved_base else None,
                    **self._physics_metric_kwargs(),
                )
            )
        elif self.config.compute_physics_metrics and preds.numel() > 0 and self.num_classes and self.num_classes > 2:
            metrics.update(
                self._compute_multiclass_physics_metrics_from_arrays(
                    probs,
                    labels_np,
                    weights_np,
                    training=False,
                    output_base=resolved_base,
                )
            )

        saved_description = None
        if resolved_base is not None:
            class_names = self.class_labels or [str(i) for i in range(self.num_classes or 0)]
            suffixes = "/".join(f"{resolved_base.stem}-{self._metric_label(name)}{resolved_base.suffix}" for name in class_names)
            saved_description = f"{resolved_base.parent} ({suffixes})"

            self._export_evaluation(
                base_path=resolved_base,
                preds=preds,
                labels=labels,
                weights=weights,
                raw_features=raw_features,
                metrics=metrics,
            )

        total_entries = int(labels.shape[0])
        class_names = self.class_labels or [str(i) for i in range(self.num_classes or 0)]
        class_counts = ", ".join(
            f"{name}={int((labels == idx).sum().item())}"
            for idx, name in enumerate(class_names)
        )

        logging.info(
            "Evaluation completed on %d entries (%s). Metrics saved%s",
            total_entries,
            class_counts,
            f" to {saved_description}" if saved_description else " in-memory",
        )

        return metrics

    def _ensure_np_suffix(self, path: Path) -> Path:
        if path.suffix.lower() not in {".npz", ".npy"}:
            return path.with_suffix(path.suffix + ".npz") if path.suffix else path.with_suffix(".npz")
        return path

    def _resolve_eval_base_path(self, provided: Path) -> Path:
        """Resolve eval output path to a base NPZ path.

        If ``provided`` includes a numpy suffix, use it directly. Otherwise, treat it as a
        directory and drop files named ``eval_output-<class>.npz`` inside.
        """

        if provided.suffix.lower() in {".npz", ".npy"}:
            return self._ensure_np_suffix(provided)
        return provided.joinpath("eval_output.npz")

    def _export_evaluation(
            self,
            *,
            base_path: Path,
            preds: torch.Tensor,
            labels: torch.Tensor,
            weights: Optional[torch.Tensor],
            raw_features: Dict[str, torch.Tensor],
            metrics: Dict[str, Any],
    ) -> None:
        base_path = self._ensure_np_suffix(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        weights_np = weights.cpu().numpy() if weights is not None else None
        features_np = {k: v.cpu().numpy() for k, v in raw_features.items()}

        expected_len = labels_np.shape[0]
        for name, arr in features_np.items():
            if arr.shape[0] != expected_len:
                raise ValueError(
                    f"Feature '{name}' has length {arr.shape[0]} but expected {expected_len} to match labels"
                )

        metric_arrays = {f"metric_{k}": np.array(v, dtype=np.float32) for k, v in metrics.items()}

        num_classes = self.num_classes or (int(labels_np.max() + 1) if labels_np.size else 0)
        class_names = self.class_labels or [str(i) for i in range(num_classes)]
        for idx, class_name in enumerate(class_names):
            mask = labels_np == idx
            if mask.any():
                class_payload: Dict[str, np.ndarray] = {
                    **{k: v[mask] for k, v in features_np.items()},
                    "predictions": preds_np[mask],
                    "labels": labels_np[mask],
                    **({"sample_weights": weights_np[mask]} if weights_np is not None else {}),
                    "num_entries": np.array(mask.sum(), dtype=np.int64),
                }
                class_payload.update(metric_arrays)

                class_path = base_path.with_name(f"{base_path.stem}-{self._metric_label(class_name)}{base_path.suffix}")
                np.savez(class_path, **class_payload)
