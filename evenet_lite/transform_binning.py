import numpy as np


def trafo60_binning(
        scores,
        labels,
        weights,
        Zb,
        Zs,
        edges_low=None,
        edges_high=None,
        mc_stat_bound=0.3,
        min_mc_yield=3.0,
        include_signal=True,
):
    """
    Trafo-60 binning with built-in histogramming.

    Parameters
    ----------
    scores : np.ndarray
        Classifier scores in [0, 1].
    labels : np.ndarray
        0 = background, 1 = signal.
    weights : np.ndarray
        Event weights.
    Zb, Zs : float
        trafoSixZ (background) and trafoSixY (signal).
    edges_low, edges_high : np.ndarray or None
        Fine binning for [0, x) and [x, 1]. Defaults to
        1000 bins in [0,0.99) and 1000 bins in [0.99,1].
    mc_stat_bound : float
        trafoSixMCstatUpBound.
    min_mc_yield : float
        trafoSixtyMCLowBound.
    include_signal : bool
        trafoSixtyIncludeS.

    Returns
    -------
    np.ndarray
        Final bin edges (ascending).
    """

    # --- default fine binning ---
    if edges_low is None:
        edges_low = np.linspace(0.0, 0.999, 10001)
    if edges_high is None:
        edges_high = np.linspace(0.999, 1.0, 10001)

    bin_edges = np.concatenate([edges_low[:-1], edges_high])
    nbins = len(bin_edges) - 1

    # --- build histograms ---
    is_bkg = labels == 0
    is_sig = labels == 1

    bkg_hist, _ = np.histogram(scores[is_bkg], bins=bin_edges, weights=weights[is_bkg])
    sig_hist, _ = np.histogram(scores[is_sig], bins=bin_edges, weights=weights[is_sig])
    bkg_w2_hist, _ = np.histogram(
        scores[is_bkg], bins=bin_edges, weights=weights[is_bkg] ** 2
    )

    # Totals
    N_b = bkg_hist.sum()
    N_s = sig_hist.sum()

    # --- Trafo-60 (right → left) ---
    rebin_edges = [nbins]
    i = nbins - 1

    while i >= 0:
        sum_b = sum_s = err2_b = 0.0
        best_dist = np.inf
        best_j = None
        passed = False

        j = i
        while j >= 0:
            sum_b += bkg_hist[j]
            sum_s += sig_hist[j]
            err2_b += bkg_w2_hist[j]

            # MC stat
            rel_mc_stat = np.sqrt(err2_b) / sum_b if sum_b > 0 else np.inf

            # Trafo-D core
            denom = 0.0
            if sum_b > 0:
                denom += sum_b / (N_b / Zb)
            if sum_s > 0:
                denom += sum_s / (N_s / Zs)
            if denom <= 0:
                break

            err2Rel = 1.0 / denom
            dist = abs(err2Rel - 1.0)

            pass_core = np.sqrt(err2Rel) < 1.0
            pass_mc = rel_mc_stat < mc_stat_bound
            pass_yield = (
                (sum_b + sum_s) >= min_mc_yield
                if include_signal
                else sum_b >= min_mc_yield
            )

            if pass_yield:
                print(f"{j}): sum_b: {sum_b}, passed: {passed}, yield: {pass_yield}, pass_mc: {pass_mc}, pass_core: {pass_core}")

            if pass_core and pass_mc and pass_yield:
                passed = True
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                else:
                    break  # distance worsened
            j -= 1

        if not passed:
            break

        rebin_edges.append(best_j)
        i = best_j - 1

    # --- convert bin indices → score edges ---
    rebin_edges = sorted(set(rebin_edges))
    final_edges = [bin_edges[idx] for idx in rebin_edges if idx < len(bin_edges)]
    if final_edges[-1] < bin_edges[-1]:
        final_edges.append(bin_edges[-1])

    print(rebin_edges)
    print(final_edges)

    return np.array(final_edges)


def calculate_binned_significance(N_sig, N_bkg, method="asimov"):
    """
    Calculate the significance for each bin.

    Parameters:
    - signal_counts: np.array, array of signal event counts per bin.
    - background_counts: np.array, array of background event counts per bin.
    - method: str, method to calculate significance ('simple' or 'asimov').

    Returns:
    - np.array of significances for each bin.
    """
    significances = np.zeros_like(N_sig)

    if method == "simple":
        with np.errstate(divide='ignore', invalid='ignore'):
            significances = np.where(
                N_bkg > 0,
                N_sig / np.sqrt(N_bkg),
                0
            )

    elif method == "asimov":
        with np.errstate(divide='ignore', invalid='ignore'):
            significances = np.where(
                N_bkg > 0,
                np.sqrt(2 * ((N_sig + N_bkg) * np.log(1 + N_sig / N_bkg) - N_sig)),
                0
            )

    else:
        raise ValueError("Invalid method specified. Choose 'simple' or 'asimov'.")

    return significances


def binned_sig(
        test_data, test_label, test_weights,
        Zb=5, Zs=10, min_bkg_per_bin=3, min_mc_stats=1.0, method="asimov",
        reweight_factor=1, include_signal=True, edges_low=None, edges_high=None,
):
    """
    Calculate the binned significance based on Transformation D binning.

    Parameters:
    - test_data: np.array, data values to bin (e.g., classifier scores).
    - test_label: np.array, binary labels (0 for background, 1 for signal).
    - test_weights: np.array, weights for each event.
    - Zb, Zs: float, weight factors for background and signal in TrafoD binning.
    - min_bkg_per_bin: int, minimum number of background events per bin.
    - min_mc_stats: float, minimum statistical uncertainty for MC stats.
    - method: str, significance calculation method ('simple' or 'asimov').

    Returns:
    - bin_edges: list of bin edges used.
    - significances: np.array of significances for each bin.
    """
    # Step 1: Perform TrafoD binning to get bin edges
    bin_edges = trafo60_binning(
        test_data, test_label, test_weights, Zb, Zs,
        edges_low=edges_low,
        edges_high=edges_high,
        min_mc_yield=min_bkg_per_bin,
        mc_stat_bound=min_mc_stats,
        include_signal=include_signal
    )

    # Step 2: Initialize lists to store signal and background counts per bin
    signal_counts = []
    background_counts = []

    # Step 3: Calculate counts in each bin
    for i in range(len(bin_edges) - 1):
        # Select events within the bin range
        if i == len(bin_edges) - 1:
            bin_mask = (test_data >= bin_edges[i]) & (test_data <= bin_edges[i + 1])
        else:
            bin_mask = (test_data >= bin_edges[i]) & (test_data < bin_edges[i + 1])
        bin_labels = test_label[bin_mask]
        bin_weights = test_weights[bin_mask]

        # Sum weighted counts for signal and background
        signal_counts.append(np.sum(bin_weights[bin_labels == 1]))
        background_counts.append(np.sum(bin_weights[bin_labels == 0]))

    # Convert lists to arrays
    signal_counts = np.array(signal_counts) / reweight_factor
    background_counts = np.array(background_counts)

    # Step 4: Calculate significance for each bin
    significances = calculate_binned_significance(signal_counts, background_counts, method=method)

    print(f"Significance: {significances}")
    print(f"bkg: {background_counts}")
    print(f"signal: {signal_counts}")
    print(f"edge: {bin_edges}")

    return bin_edges, sum(significances)
