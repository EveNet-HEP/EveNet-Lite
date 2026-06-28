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
            "axes.unicode_minus": False,
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


def _panel_grid(num_panels: int) -> tuple[int, int]:
    if num_panels <= 3:
        return 1, max(1, num_panels)
    cols = int(np.ceil(np.sqrt(num_panels)))
    rows = int(np.ceil(num_panels / cols))
    return rows, cols


def _compact_number(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-2):
        return f"{value:.2e}"
    return f"{value:.3g}"


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
        entries_matrix: Optional[np.ndarray] = None,
        normalize: bool = True,
        title: str = "Confusion matrix",
):
    configure_style()
    matrix = np.asarray(matrix, dtype=float)
    entries = np.asarray(entries_matrix, dtype=float) if entries_matrix is not None else None
    row_sums = matrix.sum(axis=1, keepdims=True)
    shown = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0) if normalize else matrix
    names = _class_names(class_labels, matrix.shape[0])
    fig, ax = plt.subplots(figsize=(max(5.6, 0.85 * len(names) + 2.4), max(4.8, 0.75 * len(names) + 2.2)), dpi=300)
    im = ax.imshow(shown, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=34, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = np.nanmax(shown) * 0.55 if shown.size else 0.0
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            lines = [f"{100.0 * shown[i, j]:.1f}%" if normalize else _compact_number(shown[i, j])]
            lines.append(f"w={_compact_number(matrix[i, j])}")
            if entries is not None:
                lines.append(f"n={int(round(entries[i, j]))}")
            ax.text(
                j, i, "\n".join(lines),
                ha="center", va="center",
                color="white" if shown[i, j] > threshold else "black",
                fontsize=7.2 if len(names) > 4 else 8.0,
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


def _histogram_efficiency_curve(sig_hist: np.ndarray, bkg_hist: np.ndarray):
    total_sig = sig_hist.sum()
    total_bkg = bkg_hist.sum()
    if total_sig <= 0 or total_bkg <= 0:
        return None
    sig_eff = np.concatenate(([0.0], np.cumsum(sig_hist[::-1]) / total_sig))
    bkg_eff = np.concatenate(([0.0], np.cumsum(bkg_hist[::-1]) / total_bkg))
    return sig_eff, bkg_eff


def plot_rejection_curves_from_histograms(
        score_histograms: np.ndarray,
        class_labels: Optional[Sequence[str]] = None,
        *,
        title: str = "One-vs-rest background rejection",
):
    configure_style()
    score_histograms = np.asarray(score_histograms, dtype=float)
    num_classes = score_histograms.shape[0]
    names = _class_names(class_labels, num_classes)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=300)
    for cls in range(num_classes):
        sig_hist = score_histograms[cls, cls]
        bkg_hist = score_histograms[:, cls].sum(axis=0) - sig_hist
        curve = _histogram_efficiency_curve(sig_hist, bkg_hist)
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
        bins: int = 100,
        title: str = "Score distributions by true class",
):
    configure_style()
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets, dtype=int)
    weights = np.asarray(weights, dtype=float)
    num_classes = probabilities.shape[1]
    names = _class_names(class_labels, num_classes)
    rows, cols = _panel_grid(num_classes)
    fig, axes = plt.subplots(rows, cols, figsize=(4.7 * cols, 3.6 * rows), dpi=300, squeeze=False)
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
        ax.set_yscale("log")
        _clean_spines(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(num_classes, 4),
            frameon=False,
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if handles else 0.94))
    return fig


def plot_score_distributions_from_histograms(
        score_histograms: np.ndarray,
        class_labels: Optional[Sequence[str]] = None,
        *,
        bin_edges: Optional[np.ndarray] = None,
        title: str = "Score distributions by true class",
):
    configure_style()
    score_histograms = np.asarray(score_histograms, dtype=float)
    num_classes = score_histograms.shape[0]
    names = _class_names(class_labels, num_classes)
    edges = np.asarray(bin_edges, dtype=float) if bin_edges is not None else np.linspace(0.0, 1.0, score_histograms.shape[2] + 1)
    width = np.diff(edges)
    rows, cols = _panel_grid(num_classes)
    fig, axes = plt.subplots(rows, cols, figsize=(4.7 * cols, 3.6 * rows), dpi=300, squeeze=False)
    for true_cls, ax in enumerate(axes.flat):
        if true_cls >= num_classes:
            ax.axis("off")
            continue
        for score_cls in range(num_classes):
            counts = score_histograms[true_cls, score_cls]
            norm = np.sum(counts * width)
            if norm <= 0:
                continue
            ax.stairs(
                counts / norm,
                edges,
                linewidth=1.3,
                color=PALETTE[score_cls % len(PALETTE)],
                label=names[score_cls] if true_cls == 0 else None,
            )
        ax.set_title(f"True {names[true_cls]}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        _clean_spines(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(num_classes, 4),
            frameon=False,
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if handles else 0.94))
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


def plot_sic_summary(
        sic_result: Dict[str, np.ndarray],
        *,
        title: str = "SIC diagnostics",
        figsize: Tuple[float, float] = (8.4, 3.0),
        dpi: int = 300,
):
    configure_style()
    sig_eff = sic_result["sig_eff"]
    bkg_eff = sic_result["bkg_eff"]
    sic = sic_result["sic"]
    sic_full = sic_result.get("sic_full", sic)
    bkg_rej = sic_result["bkg_rej"]
    bkg_rej_full = sic_result.get("bkg_rej_full", bkg_rej)
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)
    axs[0].plot(sig_eff, bkg_eff, lw=1.6, color=PALETTE[0])
    axs[0].set_xlabel("Signal efficiency")
    axs[0].set_ylabel("Background efficiency")
    axs[0].set_xlim(0.0, 1.0)
    axs[0].set_yscale("log")

    axs[1].plot(sig_eff, sic_full, lw=1.2, color=PALETTE[1], alpha=0.45)
    axs[1].plot(sig_eff, sic, lw=1.6, color=PALETTE[1])
    axs[1].set_xlabel("Signal efficiency")
    axs[1].set_ylabel("SIC")
    axs[1].set_xlim(0.0, 1.0)

    axs[2].plot(sig_eff, bkg_rej_full, lw=1.2, color=PALETTE[2], alpha=0.45)
    axs[2].plot(sig_eff, bkg_rej, lw=1.6, color=PALETTE[2])
    axs[2].set_xlabel("Signal efficiency")
    axs[2].set_ylabel("Background rejection")
    axs[2].set_yscale("log")
    axs[2].set_xlim(0.0, 1.0)

    for ax in axs:
        _clean_spines(ax)
    fig.suptitle(title, fontsize=13)
    return fig
