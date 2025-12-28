import numpy as np
import torch
import matplotlib.pyplot as plt

# ==========================================
# 3. Plotting Helpers (accept torch or numpy; convert internally)
# ==========================================

def plot_score_overlay(y_eval, y_pred, w_eval, p_eval, bins=None, fname=None):
    # Convert tensors to numpy for matplotlib
    if isinstance(y_eval, torch.Tensor): y_eval = y_eval.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    if isinstance(w_eval, torch.Tensor): w_eval = w_eval.detach().cpu().numpy()

    mask_signal = (y_eval == 1)
    mask_bkg = (y_eval == 0)

    bkg_processes = np.unique(p_eval[mask_bkg])
    bkg_data, bkg_weights, bkg_labels = [], [], []

    for proc in bkg_processes:
        mask_proc = (p_eval == proc) & mask_bkg
        if np.sum(mask_proc) > 0:
            bkg_data.append(y_pred[mask_proc])
            bkg_weights.append(w_eval[mask_proc])
            bkg_labels.append(proc)

    plt.figure(figsize=(10, 7))

    if bins is None:
        bins = np.linspace(0, 1, 40)
    else:
        if bins[-1] < 1.0:
            bins[-1] = 1.0

    if bkg_data:
        plt.hist(
            bkg_data, bins=bins, weights=bkg_weights, stacked=True,
            label=bkg_labels, alpha=0.7, edgecolor="white", linewidth=0.3,
            density=True, log=True
        )

    if np.sum(mask_signal) > 0:
        plt.hist(
            y_pred[mask_signal], bins=bins, weights=w_eval[mask_signal],
            histtype="step", linewidth=2.5, color="red", label="Signal",
            density=True, log=True
        )

    plt.xlabel("Score ($y_{pred}$)")
    plt.ylabel("Weighted Events")
    plt.title("Score Distribution (EveNet)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    if fname:
        plt.savefig(fname)
        plt.close()
