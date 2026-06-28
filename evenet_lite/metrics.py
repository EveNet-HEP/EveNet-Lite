import logging
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scipy.special import expit, softmax

from .transform_binning import binned_sig


def _flatten_ensemble(
        logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Flatten an ensemble logits tensor to align with repeated targets/weights.

    Shapes
    ------
    logits: [E, B, C] -> [E * B, C]
    targets: [B] -> [E * B]
    weights: [B] -> [E * B]
    """

    if logits.dim() == 3:
        ensemble, batch, channels = logits.shape
        logits = logits.view(ensemble * batch, channels)
        targets = targets.repeat(ensemble)
        if weights is not None:
            weights = weights.repeat(ensemble)
    return logits, targets, weights


def _mean_ensemble_logits(logits: torch.Tensor) -> torch.Tensor:
    """Average ensemble logits along the ensemble dimension."""
    return logits.mean(dim=0) if logits.dim() == 3 else logits


def compute_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor],
        gamma: float = 1.0,
        eps: float = 1e-8,
) -> torch.Tensor:
    logits, targets, weights = _flatten_ensemble(logits, targets, weights)

    # Standard CE per sample
    ce = F.cross_entropy(logits, targets, reduction="none")

    # p_t = exp(-CE)
    pt = torch.exp(-ce)

    # Focal modulation
    focal = (1.0 - pt).clamp(min=eps) ** gamma

    per_sample = focal * ce

    if weights is not None:
        weights = weights.to(per_sample.device)
        return torch.sum(per_sample * weights) / torch.sum(weights)

    return per_sample.mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    logits = _mean_ensemble_logits(logits)
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def classification_probabilities(logits: np.ndarray) -> np.ndarray:
    """Return class probabilities from logits, averaging ensembles first."""
    logits = np.asarray(logits)
    if logits.ndim == 3:
        logits = logits.mean(axis=0)
    if logits.ndim == 1:
        sig = expit(logits)
        return np.stack([1.0 - sig, sig], axis=1)
    if logits.shape[1] == 1:
        sig = expit(logits[:, 0])
        return np.stack([1.0 - sig, sig], axis=1)
    return softmax(logits, axis=1)


def _weights_or_ones(targets: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        return np.ones(targets.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.where(np.isfinite(weights), weights, 0.0)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def compute_classification_metrics(
        logits: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
        class_labels: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray | float]:
    """Compute weighted multiclass classification metrics."""
    probs = classification_probabilities(logits)
    targets = np.asarray(targets, dtype=int)
    num_classes = len(class_labels) if class_labels is not None else probs.shape[1]
    num_classes = max(num_classes, probs.shape[1], int(targets.max() + 1) if targets.size else 0)
    weights = _weights_or_ones(targets, weights)

    valid = (targets >= 0) & (targets < num_classes) & (weights >= 0)
    targets = targets[valid]
    probs = probs[valid]
    weights = weights[valid]

    matrix = np.zeros((num_classes, num_classes), dtype=float)
    if targets.size:
        preds = np.argmax(probs, axis=1)
        np.add.at(matrix, (targets, preds), weights)

    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    total = float(support.sum())
    present = support > 0

    class_auc = np.full(num_classes, np.nan, dtype=float)
    if targets.size:
        for cls in range(num_classes):
            if cls >= probs.shape[1]:
                continue
            y_true = (targets == cls).astype(int)
            if weights[y_true == 1].sum() <= 0 or weights[y_true == 0].sum() <= 0:
                continue
            try:
                class_auc[cls], _, _, _ = weighted_roc_curve(y_true, probs[:, cls], weights)
            except ValueError:
                continue

    finite_auc = np.isfinite(class_auc)
    row_sums = support[:, None]
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums > 0,
    )

    return {
        "accuracy": float(true_positive.sum() / total) if total > 0 else 0.0,
        "balanced_accuracy": float(np.mean(recall[present])) if np.any(present) else 0.0,
        "macro_precision": float(np.mean(precision[present])) if np.any(present) else 0.0,
        "macro_recall": float(np.mean(recall[present])) if np.any(present) else 0.0,
        "macro_f1": float(np.mean(f1[present])) if np.any(present) else 0.0,
        "weighted_precision": float(np.sum(precision * support) / total) if total > 0 else 0.0,
        "weighted_recall": float(np.sum(recall * support) / total) if total > 0 else 0.0,
        "weighted_f1": float(np.sum(f1 * support) / total) if total > 0 else 0.0,
        "macro_auc": float(np.mean(class_auc[finite_auc])) if np.any(finite_auc) else 0.5,
        "weighted_auc": float(np.sum(class_auc[finite_auc] * support[finite_auc]) / support[finite_auc].sum())
        if np.any(finite_auc) and support[finite_auc].sum() > 0 else 0.5,
        "class_precision": precision,
        "class_recall": recall,
        "class_f1": f1,
        "class_support": support,
        "class_auc": class_auc,
        "confusion_matrix": matrix,
        "confusion_matrix_normalized": normalized_matrix,
        "probabilities": probs,
    }


def weighted_roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: np.ndarray,
        bin_edges: Optional[np.ndarray] = None,
        n_points: int = 1000,
        safe_eps: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # Sort by score descending
    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    sample_weight = sample_weight[sorted_idx]
    is_sig = y_true == 1
    is_bkg = y_true == 0

    total_sig = np.sum(sample_weight[is_sig])
    total_bkg = np.sum(sample_weight[is_bkg])

    # Cumulative signal/background sums
    cum_sig = np.cumsum(sample_weight * is_sig)
    cum_bkg = np.cumsum(sample_weight * is_bkg)

    tpr_raw = np.concatenate(([0.0], cum_sig / (total_sig + safe_eps)))
    fpr_raw = np.concatenate(([0.0], cum_bkg / (total_bkg + safe_eps)))

    # Prepare cumulative background weight and weight^2
    w_bkg = sample_weight * is_bkg
    w_bkg2 = (sample_weight ** 2) * is_bkg
    cum_w_bkg = np.cumsum(w_bkg)
    cum_w2_bkg = np.cumsum(w_bkg2)

    # Insert zero at start to align with tpr/fpr
    cum_w_bkg = np.concatenate(([0.0], cum_w_bkg))
    cum_w2_bkg = np.concatenate(([0.0], cum_w2_bkg))

    fpr_clipped = np.clip(fpr_raw, 0.0, 1.0)

    # Poisson-style uncertainty: σ = sqrt(sum w^2) / total_bkg
    sigma_fpr_raw = np.sqrt(cum_w2_bkg) / (total_bkg + safe_eps)

    # Interpolate everything to fixed TPR grid
    tpr_uniform = np.linspace(0, 1, n_points)
    fpr_interp = np.interp(tpr_uniform, fpr_raw, fpr_clipped)
    sigma_fpr_interp = np.interp(tpr_uniform, tpr_raw, sigma_fpr_raw)

    auc = np.trapz(tpr_raw, fpr_raw)
    return auc, fpr_interp, tpr_uniform, sigma_fpr_interp


def convert_to_SIC(sig_eff: float, bkg_rej: float, bkg_rej_unc: Optional[float] = None) -> Tuple[
    Optional[float], Optional[float]]:
    """Convert background rejection to SIC and propagate uncertainty."""

    if bkg_rej <= 0:
        return None, None

    sic = sig_eff * math.sqrt(bkg_rej)

    if bkg_rej_unc is None:
        return sic, None

    sic_unc = sig_eff * (0.5 / math.sqrt(bkg_rej)) * bkg_rej_unc
    return sic, sic_unc


def compute_sic_from_scores(
        y_true: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        edges: np.ndarray,
        min_bkg_events: int = 10,
        min_bkg_ratio: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Compute SIC curve and related quantities on weighted scores.

    The calculation follows::

        SIC = ε_s / sqrt(ε_b) = ε_s * sqrt(1 / ε_b)

    where ``ε_s`` and ``ε_b`` are the weighted signal and background efficiencies.
    Background rejection is defined as ``1 / ε_b`` and its uncertainty is derived
    from the cumulative sum of squared background weights.
    """

    if scores.size == 0:
        empty = np.array([], dtype=float)
        return {
            "sig_eff": empty,
            "bkg_eff": empty,
            "bkg_eff_unc": empty,
            "bkg_rej": empty,
            "bkg_rej_unc": empty,
            "sic": empty,
            "sic_unc": empty,
            "max_sic": 0.0,
            "max_sic_unc": 0.0,
            "best_idx": 0,
        }

    eps = 1e-12

    # Sort by score descending
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    y_sorted = y_true[order]
    w_sorted = weights[order]

    sig_mask = y_sorted == 1
    bkg_mask = y_sorted == 0

    # Cumulative weighted sums
    w_sig = w_sorted * sig_mask
    w_bkg = w_sorted * bkg_mask
    w_bkg2 = (w_sorted ** 2) * bkg_mask

    cum_sig = np.cumsum(w_sig)
    cum_bkg = np.cumsum(w_bkg)
    cum_bkg2 = np.cumsum(w_bkg2)

    total_sig = cum_sig[-1]
    total_bkg = cum_bkg[-1]

    if total_sig <= 0 or total_bkg <= 0:
        empty = np.zeros_like(edges, dtype=float)
        return {
            "sig_eff": empty,
            "bkg_eff": empty,
            "bkg_eff_unc": empty,
            "bkg_rej": empty,
            "bkg_rej_unc": empty,
            "sic": empty,
            "sic_unc": empty,
            "max_sic": 0.0,
            "max_sic_unc": 0.0,
            "best_idx": 0,
        }

    # Indices corresponding to score cuts
    idxs = np.searchsorted(-scores_sorted, -edges, side="left")
    idxs = np.clip(idxs, 0, len(cum_sig) - 1)

    # Efficiencies at cuts
    sig_eff = cum_sig[idxs] / (total_sig + eps)
    bkg_eff = cum_bkg[idxs] / (total_bkg + eps)
    bkg_eff_unc = np.sqrt(cum_bkg2[idxs]) / (total_bkg + eps)

    bkg_yield = cum_bkg[idxs]

    # Valid region
    min_bkg_eff = 0.0 if min_bkg_ratio is None else min_bkg_ratio
    valid = (bkg_eff > min_bkg_eff) & (bkg_yield >= min_bkg_events)

    # Full curves (without the minimum-background cut) for plotting
    bkg_rej_full = np.full_like(sig_eff, np.nan, dtype=float)
    sic_full = np.full_like(sig_eff, np.nan, dtype=float)
    positive_bkg = bkg_eff > 0
    bkg_rej_full[positive_bkg] = 1.0 / bkg_eff[positive_bkg]
    sic_full[positive_bkg] = sig_eff[positive_bkg] * np.sqrt(bkg_rej_full[positive_bkg])

    sic = np.full_like(sig_eff, np.nan, dtype=float)
    sic_unc = np.full_like(sig_eff, np.nan, dtype=float)
    bkg_rej = np.full_like(sig_eff, np.nan, dtype=float)
    bkg_rej_unc = np.full_like(sig_eff, np.nan, dtype=float)

    bkg_rej[valid] = 1.0 / bkg_eff[valid]
    bkg_rej_unc[valid] = bkg_eff_unc[valid] * (bkg_rej[valid] ** 2)
    sic[valid] = sig_eff[valid] * np.sqrt(bkg_rej[valid])
    sic_unc[valid] = sig_eff[valid] * 0.5 / np.sqrt(bkg_rej[valid]) * bkg_rej_unc[valid]

    if np.any(valid):
        best_idx = int(np.nanargmax(sic))
        max_sic = float(sic[best_idx])
        max_sic_unc = float(sic_unc[best_idx])
    else:
        best_idx = 0
        max_sic = 0.0
        max_sic_unc = 0.0

    valid = bkg_yield >= min_bkg_events

    if not np.any(valid):
        min_bkg_idx = None  # or raise / handle gracefully
    else:
        min_bkg_idx = np.where(valid)[0][-1]

    return {
        "sig_eff": sig_eff,
        "bkg_eff": bkg_eff,
        "bkg_eff_unc": bkg_eff_unc,
        "bkg_rej": bkg_rej,
        "bkg_rej_unc": bkg_rej_unc,
        "sic": sic,
        "sic_unc": sic_unc,
        "sic_full": sic_full,
        "bkg_rej_full": bkg_rej_full,
        "valid_mask": valid,
        "min_bkg_idx": min_bkg_idx,
        "max_sic": max_sic,
        "max_sic_unc": max_sic_unc,
        "best_idx": best_idx,
    }


def find_score_at_min_bkg(
        scores: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray],
        min_bkg_events: float,
) -> Optional[float]:
    """Return score threshold where remaining bkg yield == min_bkg_events."""
    bkg_mask = targets == 0

    bkg_scores = scores[bkg_mask]
    bkg_weights = weights[bkg_mask] if weights is not None else np.ones_like(bkg_scores)

    if bkg_scores.size == 0:
        return None

    # Sort by score descending (tightest cut first)
    order = np.argsort(-bkg_scores)
    bkg_scores = bkg_scores[order]
    bkg_weights = bkg_weights[order]

    # Cumulative remaining background
    cum_bkg = np.cumsum(bkg_weights)

    # Find first point where we exceed min_bkg_events
    idx = np.searchsorted(cum_bkg, min_bkg_events)

    if idx >= len(bkg_scores):
        return None

    return bkg_scores[idx]


def calculate_physics_metrics(
        logits: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        training: bool,
        bins: int = 1000,
        min_bkg_events: int = 100,
        log_plots: bool = False,
        wandb_run: Optional[object] = None,
        log_step: Optional[int] = None,
        f_name: Optional[str] = None,
        min_bkg_ratio: Optional[float] = None,
        Zs: int = 10,
        Zb: int = 5,
        min_bkg_per_bin: int = 3,
        min_mc_stats: float = 1.0,
        include_signal_in_stat: bool = False,
        edges_low=None,
        edges_high=None,
        logger: logging.Logger | None = None,
) -> Dict[str, np.ndarray]:
    """Calculates AUC and Max SIC with statistical uncertainty."""

    if logits.ndim == 3:
        logits = logits.mean(axis=0)

    # Convert logits → scores
    if logits.ndim == 1 or logits.shape[1] == 1:
        scores = logits
    else:
        scores = softmax(logits, axis=1)[:, 1]

    edges = np.linspace(0, 1, bins + 1)

    sic_result = compute_sic_from_scores(
        targets, scores, weights, edges, min_bkg_events=min_bkg_events, min_bkg_ratio=min_bkg_ratio
    )

    try:
        # auc_val = roc_auc_score(targets, scores, sample_weight=weights)
        auc_val, _, _, _ = weighted_roc_curve(targets, scores, sample_weight=weights)
    except ValueError:
        auc_val = 0.5

    trafo_edge, bin_sig = binned_sig(
        test_data=scores,
        test_label=targets,
        test_weights=weights,
        Zb=Zb,
        Zs=Zs,
        min_bkg_per_bin=min_bkg_per_bin,
        min_mc_stats=min_mc_stats,
        include_signal=include_signal_in_stat,
        edges_low=edges_low,
        edges_high=edges_high,
        logger=logger,
    )

    metrics = {
        "auc": float(auc_val),
        "max_sic": float(sic_result["max_sic"]),
        "max_sic_unc": float(sic_result["max_sic_unc"]),
        "trafo_bin_sig": float(bin_sig),
        "sic": sic_result["sic"],
        "sic_unc": sic_result["sic_unc"],
        "edges": edges,
        "trafo_edge": trafo_edge,
    }

    if log_plots:
        from .plots import close_figure, plot_sic_diagnostics

        fig = plot_sic_diagnostics(
            targets=targets,
            scores=scores,
            weights=weights,
            sic_result=sic_result,
            min_bkg_events=min_bkg_events,
        )

        if f_name is not None:
            fig.savefig(f_name, dpi=300, bbox_inches="tight")
        if wandb_run is not None:
            try:
                import wandb

                if training:
                    log_name = "Physics/train-SIC"
                else:
                    log_name = "Physics/valid-SIC"

                log_kwargs = {"step": log_step} if log_step is not None else {}
                wandb_run.log({log_name: wandb.Image(fig)}, **log_kwargs)
            finally:
                pass
        close_figure(fig)

    return metrics


def summarize_metrics(accumulator: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {name: accumulator[name] / max(1, counts[name]) for name in accumulator}
