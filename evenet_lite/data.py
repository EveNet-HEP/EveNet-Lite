from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

from .callbacks import EvenetLiteNormalizer
from .heads import DEFAULT_HEAD


class EvenetTensorDataset(Dataset):
    """Dataset wrapper around in-memory tensors.

    Args:
        features: Mapping of feature group name to tensor with leading batch dimension.
        labels: Target labels tensor, or ``head -> labels`` for multi-head classifiers.
        sample_weights: Optional per-sample weights tensor.
        normalizer: Normalizer applied on-the-fly when retrieving items.
    """

    def __init__(
            self,
            features: Dict[str, torch.Tensor],
            labels: torch.Tensor | Mapping[str, torch.Tensor],
            sample_weights: Optional[torch.Tensor] = None,
            normalizer: Optional[EvenetLiteNormalizer] = None,
            include_indices: bool = False,
    ) -> None:
        self.raw_features = {}
        for name, tensor in features.items():
            tensor = torch.as_tensor(tensor)
            if name in {"x", "globals", "params"}:
                tensor = tensor.to(dtype=torch.float32)
            self.raw_features[name] = tensor

        if isinstance(labels, Mapping):
            if not labels:
                raise ValueError("labels dictionary must not be empty")
            self.labels = {head: torch.as_tensor(value).long() for head, value in labels.items()}
        else:
            self.labels = torch.as_tensor(labels).long()
        self.sample_weights = (
            torch.as_tensor(sample_weights, dtype=torch.float32) if sample_weights is not None else None
        )
        self.normalizer = normalizer
        self.include_indices = include_indices

        self.features: Dict[str, torch.Tensor] = {}
        self._prepared_normalizer: Optional[EvenetLiteNormalizer] = None
        self._length = self._validate_lengths()
        self._prepare_features()

    @property
    def label_heads(self) -> list[str]:
        return list(self.labels.keys()) if isinstance(self.labels, dict) else [DEFAULT_HEAD]

    def labels_for(self, head: Optional[str] = None) -> torch.Tensor:
        if isinstance(self.labels, dict):
            resolved = head if head is not None else self.label_heads[0]
            return self.labels[resolved]
        return self.labels

    def _validate_lengths(self) -> int:
        label_tensors = self.labels.values() if isinstance(self.labels, dict) else [self.labels]
        lengths = {int(tensor.shape[0]) for tensor in label_tensors}
        if len(lengths) != 1:
            raise ValueError("all label tensors must have the same leading dimension")
        length = lengths.pop()

        if self.sample_weights is not None and self.sample_weights.shape[0] != length:
            raise ValueError("sample_weights length must match labels")

        for name, tensor in self.raw_features.items():
            if tensor.shape[0] != length:
                raise ValueError(f"feature {name!r} length {tensor.shape[0]} does not match labels length {length}")
        return length

    def set_normalizer(self, normalizer: EvenetLiteNormalizer) -> None:
        if normalizer is self._prepared_normalizer:
            return

        self.normalizer = normalizer
        self._prepare_features()

    def _prepare_features(self) -> None:
        """Precompute normalized features to avoid per-sample allocations."""

        if self.normalizer is self._prepared_normalizer and self.features:
            return

        if self.normalizer is None:
            self.features = self.raw_features
            self._prepared_normalizer = self.normalizer
            return

        with torch.no_grad():
            transformed = self.normalizer.transform(self.raw_features)
        self.features = {name: torch.as_tensor(tensor) for name, tensor in transformed.items()}
        self._prepared_normalizer = self.normalizer

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Any, Optional[torch.Tensor]]:
        features = {k: v[idx] for k, v in self.features.items()}
        label = {head: value[idx] for head, value in self.labels.items()} if isinstance(self.labels, dict) else self.labels[idx]
        weight = self.sample_weights[idx] if self.sample_weights is not None else torch.tensor(1.0, dtype=torch.float32)
        if self.include_indices:
            return features, label, weight, torch.tensor(idx, dtype=torch.long)
        return features, label, weight


class DistributedWeightedSampler(Sampler[int]):
    """Distributed-aware weighted sampler.

    Generates a global weighted sample list and shards it across ranks so that
    each replica processes a distinct subset. Sampling is with replacement to
    mirror ``WeightedRandomSampler`` semantics.
    """

    def __init__(
            self, weights: torch.Tensor,
            num_samples: Optional[int] = None, replacement: bool = True, epoch_size: Optional[int] = None
    ) -> None:
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights should be a 1D tensor")
        self.weights = weights.float()
        self.replacement = replacement
        self.epoch_size = int(epoch_size) if epoch_size is not None else len(weights)
        self.num_samples = num_samples if num_samples is not None else self.epoch_size

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.num_samples_per_replica = (self.num_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples_per_replica * self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=generator).tolist()
        # Subsample for this replica
        offset_indices = indices[self.rank: self.total_size: self.num_replicas]
        return iter(offset_indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica


def build_sampler(
        sampler: Optional[str],
        dataset: EvenetTensorDataset,
        weights: Optional[torch.Tensor],
        epoch_size: Optional[int] = None,
) -> Optional[Sampler[int]]:
    if sampler == "weighted":
        if weights is None:
            if isinstance(dataset.labels, dict):
                raise ValueError("weighted sampler with multi-head labels requires explicit sample weights")
            labels = dataset.labels.long()
            # derive weights from labels
            class_counts = torch.bincount(labels.long())
            class_weights = class_counts.float().reciprocal().clamp_max(class_counts.numel())
            sample_weights = class_weights[labels.long()]
        else:
            sample_weights = torch.as_tensor(weights).float()
        return DistributedWeightedSampler(sample_weights, epoch_size=epoch_size)
    return None
