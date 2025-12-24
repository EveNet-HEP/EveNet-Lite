"""EveNet backbone wrapper used by Evenet-Lite trainers with optional ensembles.

This module keeps the public API intact while providing flexible sharing of
submodules across ensemble members.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import nn

from evenet.control.global_config import DotDict
from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.classification.classification_head import ClassificationHead


def _build_embedding(config: DotDict, global_input_dim: int) -> GlobalVectorEmbedding:
    cfg = config.Body.GlobalEmbedding
    return GlobalVectorEmbedding(
        linear_block_type=cfg.linear_block_type,
        input_dim=global_input_dim,
        hidden_dim_scale=cfg.transformer_dim_scale,
        initial_embedding_dim=cfg.initial_embedding_dim,
        final_embedding_dim=cfg.hidden_dim,
        normalization_type=cfg.normalization,
        activation_type=cfg.linear_activation,
        skip_connection=cfg.skip_connection,
        num_embedding_layers=cfg.num_embedding_layers,
        dropout=cfg.dropout,
    )


def _build_pet(config: DotDict, sequential_input_dim: int) -> PETBody:
    cfg = config.Body.PET
    return PETBody(
        num_feat=sequential_input_dim,
        num_keep=cfg.num_feature_keep,
        feature_drop=cfg.feature_drop,
        projection_dim=cfg.hidden_dim,
        local=cfg.enable_local_embedding,
        K=cfg.local_Krank,
        num_local=cfg.num_local_layer,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        drop_probability=cfg.drop_probability,
        talking_head=cfg.talking_head,
        layer_scale=cfg.layer_scale,
        layer_scale_init=cfg.layer_scale_init,
        dropout=cfg.dropout,
        mode=cfg.mode,
    )


def _build_object_encoder(config: DotDict) -> ObjectEncoder:
    cfg = config.Body.ObjectEncoder
    return ObjectEncoder(
        input_dim=config.Body.PET.hidden_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.hidden_dim,
        position_embedding_dim=cfg.position_embedding_dim,
        num_heads=cfg.num_attention_heads,
        transformer_dim_scale=cfg.transformer_dim_scale,
        num_linear_layers=cfg.num_embedding_layers,
        num_encoder_layers=cfg.num_encoder_layers,
        dropout=cfg.dropout,
        conditioned=False,
        skip_connection=cfg.skip_connection,
        encoder_skip_connection=cfg.encoder_skip_connection,
    )


def _build_classification_head(
    config: DotDict,
    class_label: Dict[str, List[str]],
    num_classes: Dict[str, int],
    input_dim: int,
) -> ClassificationHead:
    cfg = config.Classification
    return ClassificationHead(
        input_dim=input_dim,
        class_label=class_label,
        event_num_classes=num_classes,
        num_layers=cfg.num_classification_layers,
        hidden_dim=cfg.hidden_dim,
        skip_connection=cfg.skip_connection,
        dropout=cfg.dropout,
        num_attention_heads=cfg.num_attention_heads,
    )


class EveNetLite(nn.Module):
    """EveNet-Lite with optional internal ensembles and configurable sharing."""

    def __init__(
        self,
        config: DotDict,
        global_input_dim: int,
        sequential_input_dim: int,
        cls_label: List[str],
        n_ensemble: int = 1,
        ensemble_mode: str = "independent",
        shared_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        if n_ensemble < 1:
            raise ValueError("n_ensemble must be >= 1")
        if ensemble_mode not in {"independent", "shared"}:
            raise ValueError("ensemble_mode must be 'independent' or 'shared'")

        self.n_ensemble = n_ensemble
        self.ensemble_mode = ensemble_mode
        self.network_cfg = config
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim

        self.class_label = {"EVENT": cls_label}
        self.num_classes = {"EVENT": len(cls_label)}

        default_shared = {"GlobalEmbedding", "PET", "ObjectEncoder"} if ensemble_mode == "shared" else set()
        shared_set: Set[str] = set(default_shared if shared_modules is None else shared_modules)
        valid_share = {"GlobalEmbedding", "PET", "ObjectEncoder", "Classification"}
        invalid = shared_set - valid_share
        if invalid:
            raise ValueError(f"Unsupported shared module names: {sorted(invalid)}")
        self.shared_set = shared_set

        # Build shared modules (single instances) when requested.
        self.GlobalEmbedding: nn.Module = (
            _build_embedding(config, global_input_dim) if "GlobalEmbedding" in shared_set else nn.ModuleList()
        )
        self.PET: nn.Module = (
            _build_pet(config, sequential_input_dim) if "PET" in shared_set else nn.ModuleList()
        )
        self.ObjectEncoder: nn.Module = (
            _build_object_encoder(config) if "ObjectEncoder" in shared_set else nn.ModuleList()
        )

        head_dim = config.Body.ObjectEncoder.hidden_dim
        self.Classification: nn.Module = (
            _build_classification_head(config, self.class_label, self.num_classes, head_dim)
            if "Classification" in shared_set
            else nn.ModuleList()
        )

        # Build per-member modules for non-shared components.
        self.GlobalEmbedding_list = (
            self.GlobalEmbedding if isinstance(self.GlobalEmbedding, nn.ModuleList) else nn.ModuleList()
        )
        self.PET_list = self.PET if isinstance(self.PET, nn.ModuleList) else nn.ModuleList()
        self.ObjectEncoder_list = (
            self.ObjectEncoder if isinstance(self.ObjectEncoder, nn.ModuleList) else nn.ModuleList()
        )
        self.Classification_list = (
            self.Classification if isinstance(self.Classification, nn.ModuleList) else nn.ModuleList()
        )

        for _ in range(self.n_ensemble if "GlobalEmbedding" not in shared_set else 0):
            self.GlobalEmbedding_list.append(_build_embedding(config, global_input_dim))
        for _ in range(self.n_ensemble if "PET" not in shared_set else 0):
            self.PET_list.append(_build_pet(config, sequential_input_dim))
        for _ in range(self.n_ensemble if "ObjectEncoder" not in shared_set else 0):
            self.ObjectEncoder_list.append(_build_object_encoder(config))
        for _ in range(self.n_ensemble if "Classification" not in shared_set else 0):
            self.Classification_list.append(
                _build_classification_head(config, self.class_label, self.num_classes, head_dim)
            )

        # Local feature indices are the same for all PET modules.
        pet_ref = self._pet_for_member(0)
        self.local_feature_indices = pet_ref.local_point_index if hasattr(pet_ref, "local_point_index") else []

        self._log_ensemble_structure(shared_set)

    # ------------------------------------------------------------------ helpers
    def _embedding_for_member(self, idx: int) -> GlobalVectorEmbedding:
        if "GlobalEmbedding" in self.shared_set:
            return self.GlobalEmbedding  # type: ignore[return-value]
        return self.GlobalEmbedding_list[idx]

    def _pet_for_member(self, idx: int) -> PETBody:
        if "PET" in self.shared_set:
            return self.PET  # type: ignore[return-value]
        return self.PET_list[idx]

    def _object_encoder_for_member(self, idx: int) -> ObjectEncoder:
        if "ObjectEncoder" in self.shared_set:
            return self.ObjectEncoder  # type: ignore[return-value]
        return self.ObjectEncoder_list[idx]

    def _classification_for_member(self, idx: int) -> ClassificationHead:
        if "Classification" in self.shared_set:
            return self.Classification  # type: ignore[return-value]
        return self.Classification_list[idx]

    # ------------------------------------------------------------------ forward
    def _forward_single(
        self,
        idx: int,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        globals: torch.Tensor,
    ) -> torch.Tensor:
        input_point_cloud = x  # (B, N, D)
        B, _N, _D = input_point_cloud.shape
        input_point_cloud_mask = x_mask.unsqueeze(-1)  # (B, N, 1)
        global_conditions = globals.unsqueeze(1)  # (B, 1, Dg)
        global_conditions_mask = torch.ones((B, 1, 1), device=input_point_cloud.device)  # (B, 1, 1)
        time = torch.zeros((B,), device=input_point_cloud.device)
        full_attn_mask = None
        time_masking = torch.zeros_like(input_point_cloud_mask).float()
        global_feature_mask = torch.ones_like(global_conditions).float()

        global_embedding = self._embedding_for_member(idx)
        pet = self._pet_for_member(idx)
        object_encoder = self._object_encoder_for_member(idx)
        classification_head = self._classification_for_member(idx)

        full_global_conditions = global_embedding(
            x=global_conditions * global_feature_mask,
            mask=global_conditions_mask,
        )

        local_points = input_point_cloud[..., self.local_feature_indices]
        full_input_point_cloud = pet(
            input_features=input_point_cloud,
            input_points=local_points,
            mask=input_point_cloud_mask,
            attn_mask=full_attn_mask,
            time=time,
            time_masking=time_masking,
        )
        embeddings, _embedded_globals, event_token = object_encoder(
            encoded_vectors=full_input_point_cloud,
            mask=input_point_cloud_mask,
            condition_vectors=full_global_conditions,
            condition_mask=global_conditions_mask,
        )
        classifications = classification_head(
            x=embeddings,
            x_mask=input_point_cloud_mask,
            event_token=event_token,
        )
        return classifications["classification/EVENT"]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        outputs = [self._forward_single(idx, x, x_mask, globals) for idx in range(self.n_ensemble)]
        if self.n_ensemble == 1:
            return outputs[0]
        return torch.stack(outputs, dim=0)

    # ------------------------------------------------------------------ logging
    def _log_ensemble_structure(self, shared_set: Set[str]) -> None:
        if self.n_ensemble <= 1:
            return
        logger = logging.getLogger(__name__)
        shared_sorted = sorted(shared_set)
        independent = sorted({"GlobalEmbedding", "PET", "ObjectEncoder", "Classification"} - shared_set)
        lines = [
            f"EveNetLite ensemble initialized:",
            f"  members      : {self.n_ensemble}",
            f"  mode         : {self.ensemble_mode}",
            f"  shared       : {', '.join(shared_sorted) if shared_sorted else 'none'}",
            f"  independent  : {', '.join(independent) if independent else 'none'}",
        ]
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------ ckpt expansion
    def expand_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expand a single-model checkpoint to match ensemble parameter names."""

        if self.n_ensemble == 1:
            return {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}

        normalized: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            stripped = key
            for prefix in ("model.", "module."):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
            normalized[stripped] = value

        expanded: Dict[str, torch.Tensor] = {}
        components = ["GlobalEmbedding", "PET", "ObjectEncoder", "Classification"]

        for key, value in normalized.items():
            target_component: Optional[str] = None
            remainder = ""
            for comp in components:
                if key == comp or key.startswith(f"{comp}."):
                    target_component = comp
                    remainder = key[len(comp) :]
                    break
            if target_component is None:
                continue

            if target_component in self.shared_set:
                expanded[f"{target_component}{remainder}"] = value
            else:
                for idx in range(self.n_ensemble):
                    expanded[f"{target_component}.{idx}{remainder}"] = value

        return expanded


__all__ = ["EveNetLite"]
