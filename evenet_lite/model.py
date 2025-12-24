"""EveNet backbone wrapper used by Evenet-Lite trainers.

This module re-exports the reference EveNetLite architecture that stitches
together the upstream `evenet` submodule components (embedding, PET body,
object encoder, and classification head). Users are expected to bring a
populated `evenet` submodule (e.g., via git submodule) so these imports
resolve at runtime.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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


def _build_evenet_components(
    config: DotDict,
    global_input_dim: int,
    sequential_input_dim: int,
    cls_label: List[str],
) -> Tuple[
    List[int],
    GlobalVectorEmbedding,
    PETBody,
    ObjectEncoder,
    ClassificationHead,
    Dict[str, List[str]],
    Dict[str, int],
    int,
]:
    class_label = {"EVENT": cls_label}
    num_classes = {"EVENT": len(cls_label)}

    # [1] Global Embedding
    global_embedding_cfg = config.Body.GlobalEmbedding
    global_embedding = GlobalVectorEmbedding(
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
    pet = PETBody(
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
    obj_encoder = ObjectEncoder(
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

    # [4] Classification Head
    classification = _build_classification_head(
        config=config,
        class_label=class_label,
        num_classes=num_classes,
        input_dim=obj_encoder_cfg.hidden_dim,
    )

    return (
        pet_config.local_point_index,
        global_embedding,
        pet,
        obj_encoder,
        classification,
        class_label,
        num_classes,
        obj_encoder_cfg.hidden_dim,
    )


def _encode_backbone(
    local_feature_indices: List[int],
    global_embedding: GlobalVectorEmbedding,
    pet: PETBody,
    object_encoder: ObjectEncoder,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    globals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_point_cloud = x  # (B, N, D)
    B, _N, _D = input_point_cloud.shape
    input_point_cloud_mask = x_mask.unsqueeze(-1)  # (B, N, 1)
    global_conditions = globals.unsqueeze(1)  # (B, 1, Dg)
    global_conditions_mask = torch.ones((B, 1, 1), device=input_point_cloud.device)  # (B, 1, 1)
    time = torch.zeros((B,), device=input_point_cloud.device)
    full_attn_mask = None
    time_masking = torch.zeros_like(input_point_cloud_mask).float()
    global_feature_mask = torch.ones_like(global_conditions).float()

    full_global_conditions = global_embedding(
        x=global_conditions * global_feature_mask,
        mask=global_conditions_mask,
    )

    local_points = input_point_cloud[..., local_feature_indices]
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
    return embeddings, input_point_cloud_mask, event_token


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


class _EveNetLiteSingle(nn.Module):
    """Single EveNet backbone with classification head."""

    def __init__(
        self,
        config: DotDict,
        global_input_dim: int,
        sequential_input_dim: int,
        cls_label: List[str],
    ) -> None:
        super().__init__()

        (
            self.local_feature_indices,
            self.GlobalEmbedding,
            self.PET,
            self.ObjectEncoder,
            self.Classification,
            self.class_label,
            self.num_classes,
            _head_dim,
        ) = _build_evenet_components(config, global_input_dim, sequential_input_dim, cls_label)

        self.network_cfg = config
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        embeddings, input_point_cloud_mask, event_token = _encode_backbone(
            self.local_feature_indices,
            self.GlobalEmbedding,
            self.PET,
            self.ObjectEncoder,
            x,
            x_mask,
            globals,
        )
        return _apply_classification_head(self.Classification, embeddings, input_point_cloud_mask, event_token)


class EveNetLite(nn.Module):
    """EveNet-Lite with optional internal ensembles.

    Parameters
    ----------
    config:
        Parsed EveNet configuration (DotDict) providing Body and Classification
        sections.
    global_input_dim:
        Dimensionality of global (per-event) features.
    sequential_input_dim:
        Dimensionality of sequential/object-level features.
    cls_label:
        List of class labels for the classification head.
    n_ensemble:
        Number of ensemble members to instantiate. When set to ``1`` the
        behavior matches the original single-model implementation.
    ensemble_mode:
        ``"independent"`` creates fully isolated backbones/heads per ensemble
        member, while ``"shared_backbone"`` reuses a single backbone with
        independent heads.
    """

    def __init__(
        self,
        config: DotDict,
        global_input_dim: int,
        sequential_input_dim: int,
        cls_label: List[str],
        n_ensemble: int = 1,
        ensemble_mode: str = "independent",
    ) -> None:
        super().__init__()

        if n_ensemble < 1:
            raise ValueError("n_ensemble must be >= 1")
        if ensemble_mode not in {"independent", "shared_backbone"}:
            raise ValueError("ensemble_mode must be 'independent' or 'shared_backbone'")

        self.n_ensemble = n_ensemble
        self.ensemble_mode = ensemble_mode

        (
            self.local_feature_indices,
            global_embedding,
            pet,
            obj_encoder,
            classification,
            self.class_label,
            self.num_classes,
            head_dim,
        ) = _build_evenet_components(config, global_input_dim, sequential_input_dim, cls_label)

        self.network_cfg = config
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim

        if self.n_ensemble == 1 and self.ensemble_mode == "independent":
            self.model = _EveNetLiteSingle(config, global_input_dim, sequential_input_dim, cls_label)
            self.GlobalEmbedding = self.model.GlobalEmbedding
            self.PET = self.model.PET
            self.ObjectEncoder = self.model.ObjectEncoder
            self.Classification = self.model.Classification
        elif self.ensemble_mode == "independent":
            self.models = nn.ModuleList(
                _EveNetLiteSingle(config, global_input_dim, sequential_input_dim, cls_label)
                for _ in range(self.n_ensemble)
            )
        else:
            self.GlobalEmbedding = global_embedding
            self.PET = pet
            self.ObjectEncoder = obj_encoder
            self.Classification = nn.ModuleList(
                [_build_classification_head(config, self.class_label, self.num_classes, head_dim)
                 for _ in range(self.n_ensemble)]
            )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        if self.n_ensemble == 1 and self.ensemble_mode == "independent":
            return self.model(x=x, x_mask=x_mask, globals=globals)

        if self.ensemble_mode == "independent":
            outputs = [model(x=x, x_mask=x_mask, globals=globals) for model in self.models]
            return torch.stack(outputs, dim=0)

        embeddings, input_point_cloud_mask, event_token = _encode_backbone(
            self.local_feature_indices,
            self.GlobalEmbedding,
            self.PET,
            self.ObjectEncoder,
            x,
            x_mask,
            globals,
        )

        outputs = [
            _apply_classification_head(head, embeddings, input_point_cloud_mask, event_token)
            for head in self.Classification
        ]
        return torch.stack(outputs, dim=0)

    def expand_state_dict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expand a single-model checkpoint to match ensemble parameter names."""

        # Already in ensemble form
        if self.n_ensemble == 1:
            return state

        has_ensemble_prefix = any(key.startswith("models.") or key.startswith("Classification.") for key in state)
        if has_ensemble_prefix:
            return state

        if self.ensemble_mode == "independent":
            expanded: Dict[str, torch.Tensor] = {}
            for idx in range(self.n_ensemble):
                for key, value in state.items():
                    expanded[f"models.{idx}.{key}"] = value
            return expanded

        expanded = {k: v for k, v in state.items() if not k.startswith("Classification")}
        for idx in range(self.n_ensemble):
            for key, value in state.items():
                if key.startswith("Classification"):
                    suffix = key[len("Classification") :]
                    expanded[f"Classification.{idx}{suffix}"] = value
        return expanded


__all__ = ["EveNetLite"]
