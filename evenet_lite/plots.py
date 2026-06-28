from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


PALETTE = ["#4C78A8", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#7A7A7A", "#6D8EF7", "#A38E89"]
_STYLE_CONFIGURED = False


def configure_style() -> None:
    global _STYLE_CONFIGURED
    if _STYLE_CONFIGURED:
        return
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    _STYLE_CONFIGURED = True


def close_figure(fig) -> None:
    plt.close(fig)


def _clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)


def _class_names(class_labels: Optional[Sequence[str]], num_classes: int) -> list[str]:
    if class_labels is None:
        return [str(i) for i in range(num_classes)]
    return [class_labels[i] if i < len(class_labels) else str(i) for i in range(num_classes)]


def _weighted_efficiency_curve(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray):
    order = np.argsort(-scores)
    y_true = y_true[order]
    weights = weights[order]
    sig = y_true == 1
    bkg = y_true == 0
    total_sig = weights[sig].sum()
    total_bkg = weights[bkg].sum()
    if total_sig <= 0 or total_bkg <= 0:
        return None
    sig_eff = np.concatenate(([0.0], np.cumsum(weights * sig) / total_sig))
    bkg_eff = np.concatenate(([0.0], np.cumsum(weights * bkg) / total_bkg))
    return sig_eff, bkg_eff


def plot_confusion_matrix(
        matrix: np.ndarray,
        class_labels: Optional[Sequence[str]] = None,
        *,
        normalize: bool = True,
        title: str = "Confusion matrix",
):
    configure_style()
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    shown = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0) if normalize else matrix
    names = _class_names(class_labels, matrix.shape[0])
    fig, ax = plt.subplots(figsize=(max(4.8, 0.65 * len(names) + 2.0), max(4.0, 0.58 * len(names) + 1.8)), dpi=300)
    im = ax.imshow(shown, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=34, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fmt = ".2f" if normalize else ".0f"
    threshold = np.nanmax(shown) * 0.55 if shown.size else 0.0
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            ax.text(
                j, i, format(shown[i, j], fmt),
                ha="center", va="center",
                color="white" if shown[i, j] > threshold else "black",
                fontsize=8.5,
            )
    fig.tight_layout()
    return fig


def plot_rejection_curves(
        probabilities: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        class_labels: Optional[Sequence[str]] = None,
        *,
        title: str = "One-vs-rest background rejection",
):
    configure_style()
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets, dtype=int)
    weights = np.asarray(weights, dtype=float)
    num_classes = probabilities.shape[1]
    names = _class_names(class_labels, num_classes)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=300)
    for cls in range(num_classes):
        curve = _weighted_efficiency_curve((targets == cls).astype(int), probabilities[:, cls], weights)
        if curve is None:
            continue
        sig_eff, bkg_eff = curve
        rejection = np.full_like(bkg_eff, np.nan, dtype=float)
        np.divide(1.0, bkg_eff, out=rejection, where=bkg_eff > 0)
        ax.plot(
            sig_eff,
            rejection,
            color=PALETTE[cls % len(PALETTE)],
            linewidth=1.6,
            label=names[cls],
        )
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection (1 / background efficiency)")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    _clean_spines(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_score_distributions(
        probabilities: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        class_labels: Optional[Sequence[str]] = None,
        *,
        bins: int = 50,
        title: str = "Score distributions by true class",
):
    configure_style()
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets, dtype=int)
    weights = np.asarray(weights, dtype=float)
    num_classes = probabilities.shape[1]
    names = _class_names(class_labels, num_classes)
    cols = min(3, num_classes)
    rows = int(np.ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.8 * rows), dpi=300, squeeze=False)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for true_cls, ax in enumerate(axes.flat):
        if true_cls >= num_classes:
            ax.axis("off")
            continue
        mask = targets == true_cls
        for score_cls in range(num_classes):
            if not np.any(mask):
                continue
            ax.hist(
                probabilities[mask, score_cls],
                bins=edges,
                weights=weights[mask],
                histtype="step",
                density=True,
                linewidth=1.3,
                color=PALETTE[score_cls % len(PALETTE)],
                label=names[score_cls] if true_cls == 0 else None,
            )
        ax.set_title(f"True {names[true_cls]}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        _clean_spines(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(num_classes, 4), frameon=False)
        fig.subplots_adjust(top=0.86)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return fig


def _score_at_min_bkg(scores: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray], min_bkg_events: float):
    bkg_mask = targets == 0
    bkg_scores = scores[bkg_mask]
    bkg_weights = weights[bkg_mask] if weights is not None else np.ones_like(bkg_scores)
    if bkg_scores.size == 0:
        return None
    order = np.argsort(-bkg_scores)
    cum_bkg = np.cumsum(bkg_weights[order])
    idx = np.searchsorted(cum_bkg, min_bkg_events)
    return None if idx >= bkg_scores.size else bkg_scores[order][idx]


def _plain_number(x, _pos):
    if x >= 1:
        return f"{x:g}"
    return f"{x:.2g}"


def plot_sic_diagnostics(
        targets: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        sic_result: Dict[str, np.ndarray],
        min_bkg_events: int = 0,
        *,
        figsize: Tuple[float, float] = (7.2, 6.2),
        dpi: int = 300,
):
    configure_style()
    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    sig_eff = sic_result["sig_eff"]
    bkg_eff = sic_result["bkg_eff"]
    sic = sic_result["sic"]
    sic_unc = sic_result["sic_unc"]
    bkg_rej = sic_result["bkg_rej"]
    bkg_rej_unc = sic_result["bkg_rej_unc"]
    sic_full = sic_result.get("sic_full", sic)
    bkg_rej_full = sic_result.get("bkg_rej_full", bkg_rej)
    valid_mask = sic_result.get("valid_mask", np.isfinite(sic))
    min_bkg_idx = sic_result.get("min_bkg_idx")

    axs[0, 0].plot(sig_eff, bkg_eff, lw=1.6, color=PALETTE[0])
    axs[0, 0].set_ylabel("Background efficiency")
    axs[0, 0].set_xlabel("Signal efficiency")
    axs[0, 0].set_title("Efficiency curve")
    axs[0, 0].set_xlim(0.0, 1.0)

    axs[0, 1].plot(sig_eff, sic_full, lw=1.3, color=PALETTE[1], alpha=0.45)
    axs[0, 1].plot(sig_eff, sic, lw=1.6, color=PALETTE[1])
    axs[0, 1].fill_between(sig_eff, sic - sic_unc, sic + sic_unc, where=valid_mask, color=PALETTE[1], alpha=0.18)
    axs[0, 1].set_xlabel("Signal efficiency")
    axs[0, 1].set_ylabel("SIC")
    axs[0, 1].set_title("SIC")
    axs[0, 1].set_xlim(0.0, 1.0)

    axs[1, 0].plot(sig_eff, bkg_rej_full, lw=1.3, color=PALETTE[2], alpha=0.45)
    axs[1, 0].plot(sig_eff, bkg_rej, lw=1.6, color=PALETTE[2])
    axs[1, 0].fill_between(
        sig_eff, bkg_rej - bkg_rej_unc, bkg_rej + bkg_rej_unc, where=valid_mask, color=PALETTE[2], alpha=0.18
    )
    axs[1, 0].set_xlabel("Signal efficiency")
    axs[1, 0].set_ylabel("Background rejection")
    axs[1, 0].set_yscale("log")
    axs[1, 0].yaxis.set_major_formatter(FuncFormatter(_plain_number))
    axs[1, 0].set_title("Background rejection")
    axs[1, 0].set_xlim(0.0, 1.0)

    sig_mask = targets == 1
    bkg_mask = targets == 0
    if np.any(bkg_mask):
        axs[1, 1].hist(scores[bkg_mask], bins=50, weights=weights[bkg_mask], histtype="step", density=True,
                       linewidth=1.6, color=PALETTE[0], label="Background")
    if np.any(sig_mask):
        axs[1, 1].hist(scores[sig_mask], bins=50, weights=weights[sig_mask], histtype="step", density=True,
                       linewidth=1.6, color=PALETTE[1], label="Signal")
    axs[1, 1].set_xlabel("Classifier score")
    axs[1, 1].set_ylabel("Density")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title("Score distribution")
    axs[1, 1].set_xlim(0.0, 1.0)
    axs[1, 1].legend(frameon=False)

    if min_bkg_idx is not None:
        min_bkg_x = float(sig_eff[min_bkg_idx])
        score_cut = _score_at_min_bkg(scores=scores, targets=targets, weights=weights, min_bkg_events=min_bkg_events)
        if score_cut is not None:
            axs[1, 1].axvline(score_cut, color="0.35", linestyle="--", linewidth=0.9)
        for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
            ax.axvline(min_bkg_x, color="0.35", linestyle="--", linewidth=0.9)

    for ax in axs.flat:
        _clean_spines(ax)
    fig.suptitle("SIC diagnostics", fontsize=13)
    return fig
