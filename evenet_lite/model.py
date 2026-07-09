"""EveNet backbone wrapper used by Evenet-Lite trainers.

This module re-exports the reference EveNetLite architecture that stitches
together the upstream `evenet` submodule components (embedding, PET body,
object encoder, and classification head). Users are expected to bring a
populated `evenet` submodule (e.g., via git submodule) so these imports
resolve at runtime.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict, List

import torch
from torch import nn

from evenet.control.global_config import DotDict
from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.classification.classification_head import ClassificationHead
from .heads import DEFAULT_HEAD, normalize_class_heads


def _build_classification_head(
    config: DotDict,
    class_labels: List[str],
    input_dim: int,
) -> ClassificationHead:
    cls_cfg = config.Classification
    return ClassificationHead(
        input_dim=input_dim,
        class_label={DEFAULT_HEAD: class_labels},
        event_num_classes={DEFAULT_HEAD: len(class_labels)},
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
    return classifications[f"classification/{DEFAULT_HEAD}"]


def _build_classification_heads(
    config: DotDict,
    head_class_labels: Dict[str, List[str]],
    input_dim: int,
) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            head: _build_classification_head(config, labels, input_dim)
            for head, labels in head_class_labels.items()
        }
    )


def _apply_classification_heads(
    classification_heads: nn.ModuleDict,
    embeddings: torch.Tensor,
    input_point_cloud_mask: torch.Tensor,
    event_token: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        head: _apply_classification_head(head_module, embeddings, input_point_cloud_mask, event_token)
        for head, head_module in classification_heads.items()
    }


class EveNetBackbone(nn.Module):
    """Backbone containing GlobalEmbedding, PET, and ObjectEncoder."""

    def __init__(
        self,
        config: DotDict,
        global_input_dim: int,
        sequential_input_dim: int,
        use_adapter: bool = False,
    ) -> None:
        super().__init__()

        self.network_cfg = config
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim

        # [1] Global Embedding
        global_embedding_cfg = config.Body.GlobalEmbedding
        self.GlobalEmbedding = GlobalVectorEmbedding(
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
        self.PET = PETBody(
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
            use_adapter=use_adapter
        )

        # [3] Classification + Regression + Assignment Body
        obj_encoder_cfg = config.Body.ObjectEncoder
        self.ObjectEncoder = ObjectEncoder(
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

        self.local_feature_indices = pet_config.local_point_index
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

    def __init__(self, backbone: EveNetBackbone, classification_heads: nn.ModuleDict, return_dict: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.Classification = classification_heads
        self.return_dict = return_dict

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, globals: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        embeddings, input_point_cloud_mask, event_token = self.backbone(x, x_mask, globals)
        outputs = _apply_classification_heads(self.Classification, embeddings, input_point_cloud_mask, event_token)
        return outputs if self.return_dict else outputs[DEFAULT_HEAD]


class EveNetLite(nn.Module):
    """EveNet-Lite with optional internal ensembles."""

    def __init__(
        self,
        config: DotDict,
        global_input_dim: int,
        sequential_input_dim: int,
        cls_label: List[str] | Mapping[str, Mapping[str, Any]],
        n_ensemble: int = 1,
        ensemble_mode: str = "independent",
        use_adapter: bool = False,
    ) -> None:
        super().__init__()

        if n_ensemble < 1:
            raise ValueError("n_ensemble must be >= 1")
        if ensemble_mode not in {"independent", "shared_backbone"}:
            raise ValueError("ensemble_mode must be 'independent' or 'shared_backbone'")

        self.n_ensemble = n_ensemble
        self.ensemble_mode = ensemble_mode

        self.network_cfg = config
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim

        self.head_class_labels, self.head_loss_weights, self.return_dict = normalize_class_heads(cls_label)
        self.head_names = list(self.head_class_labels)
        self.num_classes = {head: len(labels) for head, labels in self.head_class_labels.items()}
        self.class_label = {DEFAULT_HEAD: self.head_class_labels[DEFAULT_HEAD]} if not self.return_dict else self.head_class_labels

        backbone_builder = lambda: EveNetBackbone(config, global_input_dim, sequential_input_dim, use_adapter=use_adapter)
        head_dim = config.Body.ObjectEncoder.hidden_dim
        head_builder = lambda: _build_classification_heads(config, self.head_class_labels, head_dim)

        if self.ensemble_mode == "independent":
            self.models = nn.ModuleList(
                _EveNetLiteSingle(backbone_builder(), head_builder(), self.return_dict) for _ in range(self.n_ensemble)
            )
            self.local_feature_indices = self.models[0].backbone.local_feature_indices
        else:
            self.backbone = backbone_builder()
            self.Classification = nn.ModuleList(head_builder() for _ in range(self.n_ensemble))
            self.local_feature_indices = self.backbone.local_feature_indices

        self._log_ensemble_structure()

    @property
    def GlobalEmbedding(self) -> GlobalVectorEmbedding | None:
        if self.ensemble_mode == "shared_backbone":
            return self.backbone.GlobalEmbedding
        return None

    @property
    def PET(self) -> PETBody | None:
        if self.ensemble_mode == "shared_backbone":
            return self.backbone.PET
        return None

    @property
    def ObjectEncoder(self) -> ObjectEncoder | None:
        if self.ensemble_mode == "shared_backbone":
            return self.backbone.ObjectEncoder
        return None

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        globals: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if x_mask is None:
            if mask is None:
                raise ValueError("EveNetLite.forward requires x_mask or mask")
            x_mask = mask
        if globals is None:
            raise ValueError("EveNetLite.forward requires globals")
        if self.ensemble_mode == "independent":
            outputs = [model(x=x, x_mask=x_mask, globals=globals) for model in self.models]
        else:
            embeddings, input_point_cloud_mask, event_token = self.backbone(x, x_mask, globals)
            outputs = [
                _apply_classification_heads(head, embeddings, input_point_cloud_mask, event_token)
                for head in self.Classification
            ]

        if self.return_dict:
            if self.n_ensemble == 1:
                return outputs[0]
            return {
                head: torch.stack([output[head] for output in outputs], dim=0)
                for head in self.head_names
            }

        if self.n_ensemble == 1:
            return outputs[0]
        return torch.stack(outputs, dim=0)

    def component_copies(self) -> Dict[str, int]:
        """Return the number of copies for each major component."""

        base = {
            "GlobalEmbedding": 1,
            "PET": 1,
            "ObjectEncoder": 1,
            "Classification": len(self.head_names),
        }
        if self.ensemble_mode == "independent":
            return {
                name: count * self.n_ensemble
                for name, count in base.items()
            }

        base["Classification"] = self.n_ensemble * len(self.head_names)
        return base

    def _expand_classification_remainders(self, remainder: str) -> List[str]:
        if not self.return_dict:
            first = remainder.split(".", 1)[0]
            return [remainder] if first == DEFAULT_HEAD else [f"{DEFAULT_HEAD}.{remainder}"]
        first = remainder.split(".", 1)[0]
        if first in self.head_names:
            return [remainder]
        return [f"{head}.{remainder}" for head in self.head_names]

    def _log_ensemble_structure(self) -> None:
        if self.n_ensemble <= 1:
            return

        logger = logging.getLogger(__name__)
        copies = self.component_copies()
        replicated = {name: count for name, count in copies.items() if count > 1}
        shared = [name for name, count in copies.items() if count == 1]

        logger.info(
            "Configured EveNet-Lite ensemble: n_ensemble=%d, mode=%s",
            self.n_ensemble,
            self.ensemble_mode,
        )
        if replicated:
            logger.info(
                "Replicated components: %s",
                ", ".join(f"{name} x{count}" for name, count in replicated.items()),
            )
        if shared:
            logger.info("Shared components: %s", ", ".join(shared))

    def _expand_independent(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        expanded: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            tokens = key.split(".", 2)
            if len(tokens) >= 3 and tokens[0] == "models" and tokens[1].isdigit():
                idx, rest = tokens[1], tokens[2]
                if rest.startswith("Classification."):
                    remainder = rest[len("Classification.") :]
                    for expanded_remainder in self._expand_classification_remainders(remainder):
                        expanded[f"models.{idx}.Classification.{expanded_remainder}"] = value
                else:
                    expanded[key] = value
                continue

            base_key = key[len("backbone.") :] if key.startswith("backbone.") else key
            for idx in range(self.n_ensemble):
                if base_key.startswith("Classification."):
                    remainder = base_key[len("Classification.") :]
                    for expanded_remainder in self._expand_classification_remainders(remainder):
                        expanded[f"models.{idx}.Classification.{expanded_remainder}"] = value
                elif base_key.startswith("Classification"):
                    suffix = base_key[len("Classification") :].lstrip(".")
                    for expanded_remainder in self._expand_classification_remainders(suffix):
                        expanded[f"models.{idx}.Classification.{expanded_remainder}"] = value
                else:
                    expanded[f"models.{idx}.backbone.{base_key}"] = value
        return expanded

    def _expand_shared(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        expanded: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            base_key = key
            if base_key.startswith("backbone."):
                base_key = base_key[len("backbone.") :]

            if base_key.startswith("Classification."):
                remainder = base_key[len("Classification.") :]
                first_token = remainder.split(".", 1)[0]
                if first_token.isdigit():
                    idx, sub_remainder = remainder.split(".", 1)
                    for expanded_remainder in self._expand_classification_remainders(sub_remainder):
                        expanded[f"Classification.{idx}.{expanded_remainder}"] = value
                else:
                    for idx in range(self.n_ensemble):
                        for expanded_remainder in self._expand_classification_remainders(remainder):
                            expanded[f"Classification.{idx}.{expanded_remainder}"] = value
            elif base_key.startswith("Classification"):
                suffix = base_key[len("Classification") :].lstrip(".")
                for idx in range(self.n_ensemble):
                    for expanded_remainder in self._expand_classification_remainders(suffix):
                        expanded[f"Classification.{idx}.{expanded_remainder}"] = value
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
