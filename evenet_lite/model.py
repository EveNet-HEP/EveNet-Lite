"""EveNet backbone wrapper used by Evenet-Lite trainers.

This module re-exports the reference EveNetLite architecture that stitches
together the upstream `evenet` submodule components (embedding, PET body,
object encoder, and classification head). Users are expected to bring a
populated `evenet` submodule (e.g., via git submodule) so these imports
resolve at runtime.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import torch
from torch import nn

from evenet.control.global_config import DotDict
from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.classification.classification_head import ClassificationHead


def _build_classification_head(
    config: DotDict,
    class_label: Dict[str, List[str]],
    num_classes: Dict[str, int],
    input_dim: int,
) -> ClassificationHead:
    cls_cfg = config.Classification
    return ClassificationHead(
        input_dim=input_dim,
        class_label=class_label,
        event_num_classes=num_classes,
        num_layers=cls_cfg.num_classification_layers,
        hidden_dim=cls_cfg.hidden_dim,
        skip_connection=cls_cfg.skip_connection,
        dropout=cls_cfg.dropout,
        num_attention_heads=cls_cfg.num_attention_heads,
    )


def _apply_classification_head(
    classification_head: ClassificationHead,
    embeddings: torch.Tensor,
    input_point_cloud_mask: torch.Tensor,
    event_token: torch.Tensor,
) -> torch.Tensor:
    classifications = classification_head(
        x=embeddings,
        x_mask=input_point_cloud_mask,
        event_token=event_token,
    )
    return classifications["classification/EVENT"]


class EveNetBackbone(nn.Module):
    """Backbone containing GlobalEmbedding, PET, and ObjectEncoder."""

    def __init__(
        self,
        config: DotDict,
        global_input_dim: int,
        sequential_input_dim: int,
        global_embedding: Optional[GlobalVectorEmbedding] = None,
        pet: Optional[PETBody] = None,
        object_encoder: Optional[ObjectEncoder] = None,
        local_feature_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.network_cfg = config
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim

        # [1] Global Embedding
        global_embedding_cfg = config.Body.GlobalEmbedding
        self.GlobalEmbedding = global_embedding or GlobalVectorEmbedding(
            linear_block_type=global_embedding_cfg.linear_block_type,
            input_dim=global_input_dim,
            hidden_dim_scale=global_embedding_cfg.transformer_dim_scale,
            initial_embedding_dim=global_embedding_cfg.initial_embedding_dim,
            final_embedding_dim=global_embedding_cfg.hidden_dim,
            normalization_type=global_embedding_cfg.normalization,
            activation_type=global_embedding_cfg.linear_activation,
            skip_connection=global_embedding_cfg.skip_connection,
            num_embedding_layers=global_embedding_cfg.num_embedding_layers,
            dropout=global_embedding_cfg.dropout,
        )

        # [2] PET Body
        pet_config = config.Body.PET
        self.PET = pet or PETBody(
            num_feat=sequential_input_dim,
            num_keep=pet_config.num_feature_keep,
            feature_drop=pet_config.feature_drop,
            projection_dim=pet_config.hidden_dim,
            local=pet_config.enable_local_embedding,
            K=pet_config.local_Krank,
            num_local=pet_config.num_local_layer,
            num_layers=pet_config.num_layers,
            num_heads=pet_config.num_heads,
            drop_probability=pet_config.drop_probability,
            talking_head=pet_config.talking_head,
            layer_scale=pet_config.layer_scale,
            layer_scale_init=pet_config.layer_scale_init,
            dropout=pet_config.dropout,
            mode=pet_config.mode,
        )

        # [3] Classification + Regression + Assignment Body
        obj_encoder_cfg = config.Body.ObjectEncoder
        self.ObjectEncoder = object_encoder or ObjectEncoder(
            input_dim=pet_config.hidden_dim,
            hidden_dim=obj_encoder_cfg.hidden_dim,
            output_dim=obj_encoder_cfg.hidden_dim,
            position_embedding_dim=obj_encoder_cfg.position_embedding_dim,
            num_heads=obj_encoder_cfg.num_attention_heads,
            transformer_dim_scale=obj_encoder_cfg.transformer_dim_scale,
            num_linear_layers=obj_encoder_cfg.num_embedding_layers,
            num_encoder_layers=obj_encoder_cfg.num_encoder_layers,
            dropout=obj_encoder_cfg.dropout,
            conditioned=False,
            skip_connection=obj_encoder_cfg.skip_connection,
            encoder_skip_connection=obj_encoder_cfg.encoder_skip_connection,
        )

        self.local_feature_indices = local_feature_indices or pet_config.local_point_index
        self.head_input_dim = obj_encoder_cfg.hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        globals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_point_cloud = x  # (B, N, D)
        B, _N, _D = input_point_cloud.shape
        input_point_cloud_mask = x_mask.unsqueeze(-1)  # (B, N, 1)
        global_conditions = globals.unsqueeze(1)  # (B, 1, Dg)
        global_conditions_mask = torch.ones((B, 1, 1), device=input_point_cloud.device)  # (B, 1, 1)
        time = torch.zeros((B,), device=input_point_cloud.device)
        full_attn_mask = None
        time_masking = torch.zeros_like(input_point_cloud_mask).float()
        global_feature_mask = torch.ones_like(global_conditions).float()

        full_global_conditions = self.GlobalEmbedding(
            x=global_conditions * global_feature_mask,
            mask=global_conditions_mask,
        )

        local_points = input_point_cloud[..., self.local_feature_indices]
        full_input_point_cloud = self.PET(
            input_features=input_point_cloud,
            input_points=local_points,
            mask=input_point_cloud_mask,
            attn_mask=full_attn_mask,
            time=time,
            time_masking=time_masking,
        )
        embeddings, _embedded_globals, event_token = self.ObjectEncoder(
            encoded_vectors=full_input_point_cloud,
            mask=input_point_cloud_mask,
            condition_vectors=full_global_conditions,
            condition_mask=global_conditions_mask,
        )
        return embeddings, input_point_cloud_mask, event_token


class _EveNetLiteSingle(nn.Module):
    """Single EveNet backbone with classification head."""

    def __init__(self, backbone: EveNetBackbone, classification_head: ClassificationHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.Classification = classification_head

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        embeddings, input_point_cloud_mask, event_token = self.backbone(x, x_mask, globals)
        return _apply_classification_head(self.Classification, embeddings, input_point_cloud_mask, event_token)


class EveNetLite(nn.Module):
    """EveNet-Lite with optional internal ensembles."""

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
        shared_set: Set[str] = {name for name in (shared_modules or default_shared)}
        valid_share = {"GlobalEmbedding", "PET", "ObjectEncoder", "Classification"}
        invalid = shared_set - valid_share
        if invalid:
            raise ValueError(f"Unsupported shared module names: {sorted(invalid)}")

        def _build_components() -> Dict[str, object]:
            backbone = EveNetBackbone(config, global_input_dim, sequential_input_dim)
            head_dim = backbone.head_input_dim
            head = _build_classification_head(config, self.class_label, self.num_classes, head_dim)
            return {
                "backbone": backbone,
                "Classification": head,
                "local_feature_indices": backbone.local_feature_indices,
                "head_dim": head_dim,
            }

        template = _build_components()
        self.local_feature_indices = template["local_feature_indices"]

        shared_pool: Dict[str, nn.Module] = {}
        for name in shared_set:
            if name == "Classification":
                shared_pool[name] = template["Classification"]
            else:
                shared_pool[name] = getattr(template["backbone"], name)

        def _member_components() -> Dict[str, object]:
            comps = _build_components()
            if "GlobalEmbedding" in shared_pool:
                comps["backbone"].GlobalEmbedding = shared_pool["GlobalEmbedding"]
            if "PET" in shared_pool:
                comps["backbone"].PET = shared_pool["PET"]
            if "ObjectEncoder" in shared_pool:
                comps["backbone"].ObjectEncoder = shared_pool["ObjectEncoder"]
            if "Classification" in shared_pool:
                comps["Classification"] = shared_pool["Classification"]
            comps["backbone"].local_feature_indices = self.local_feature_indices
            return comps

        if self.ensemble_mode == "independent":
            members: List[_EveNetLiteSingle] = []
            for _ in range(self.n_ensemble):
                comps = _member_components()
                members.append(_EveNetLiteSingle(comps["backbone"], comps["Classification"]))
            self.models = nn.ModuleList(members)
        else:
            self.backbone = template["backbone"]
            head_dim = template["head_dim"]
            self.Classification = nn.ModuleList(
                shared_pool.get("Classification", None) or _build_classification_head(
                    config, self.class_label, self.num_classes, head_dim
                )
                for _ in range(self.n_ensemble)
            )
            if "Classification" in shared_pool:
                for idx in range(1, self.n_ensemble):
                    self.Classification[idx] = shared_pool["Classification"]
            self.local_feature_indices = self.backbone.local_feature_indices
        self._log_ensemble_structure(shared_set)

    @property
    def GlobalEmbedding(self) -> GlobalVectorEmbedding | None:
        if self.ensemble_mode == "shared":
            return self.backbone.GlobalEmbedding
        return None

    @property
    def PET(self) -> PETBody | None:
        if self.ensemble_mode == "shared":
            return self.backbone.PET
        return None

    @property
    def ObjectEncoder(self) -> ObjectEncoder | None:
        if self.ensemble_mode == "shared":
            return self.backbone.ObjectEncoder
        return None

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        if self.ensemble_mode == "independent":
            outputs = [model(x=x, x_mask=x_mask, globals=globals) for model in self.models]
        else:
            embeddings, input_point_cloud_mask, event_token = self.backbone(x, x_mask, globals)
            outputs = [
                _apply_classification_head(head, embeddings, input_point_cloud_mask, event_token)
                for head in self.Classification
            ]

        if self.n_ensemble == 1:
            return outputs[0]
        return torch.stack(outputs, dim=0)

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

    def _expand_independent(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if any(key.startswith("models.") for key in state):
            return state

        expanded: Dict[str, torch.Tensor] = {}
        for idx in range(self.n_ensemble):
            for key, value in state.items():
                base_key = key
                if base_key.startswith("backbone."):
                    base_key = base_key[len("backbone.") :]
                if base_key.startswith("Classification"):
                    expanded[f"models.{idx}.{base_key}"] = value
                else:
                    expanded[f"models.{idx}.backbone.{base_key}"] = value
        return expanded

    def _expand_shared(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        has_backbone_prefix = any(key.startswith("backbone.") for key in state)
        has_indexed_classification = any(
            key.startswith("Classification.")
            and key[len("Classification.") :].split(".", 1)[0].isdigit()
            for key in state
        )
        if has_backbone_prefix and has_indexed_classification:
            return state

        expanded: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            base_key = key
            if base_key.startswith("backbone."):
                base_key = base_key[len("backbone.") :]

            if base_key.startswith("Classification."):
                remainder = base_key[len("Classification.") :]
                first_token = remainder.split(".", 1)[0]
                if first_token.isdigit():
                    expanded[f"Classification.{remainder}"] = value
                else:
                    for idx in range(self.n_ensemble):
                        expanded[f"Classification.{idx}.{remainder}"] = value
            elif base_key.startswith("Classification"):
                suffix = base_key[len("Classification") :]
                for idx in range(self.n_ensemble):
                    expanded[f"Classification.{idx}{suffix}"] = value
            elif base_key:
                expanded[f"backbone.{base_key}"] = value

        return expanded

    def expand_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expand a single-model checkpoint to match ensemble parameter names."""
        normalized_state: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            stripped = key
            for prefix in ("model.", "module."):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
            normalized_state[stripped] = value

        if self.ensemble_mode == "independent":
            return self._expand_independent(normalized_state)
        return self._expand_shared(normalized_state)


__all__ = ["EveNetLite"]
