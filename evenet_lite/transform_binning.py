import numpy as np


def trafoD_binning(test_data, test_label, test_weights, Zb, Zs, min_bkg_per_bin=3, min_mc_stats=0.2, reweight_factor=1):
    """
    Optimized TrafoD binning function.

    Parameters:
    - test_data: np.array, array of data values (e.g., classifier scores).
    - test_label: np.array, binary labels (0 for background, 1 for signal).
    - test_weights: np.array, weights for each event.
    - Zb: float, weight factor for background events in Z calculation.
    - Zs: float, weight factor for signal events in Z calculation.
    - min_bkg_per_bin: int, minimum number of background events per bin.
    - min_mc_stats: float, required statistical uncertainty.

    Returns:
    - bin_edges: list of bin edges satisfying the TrafoD criteria.
    """
    # Initialize lists to store bin edges and statistics
    bin_edges = []
    total_events = len(test_data)
    mc_threshold = 1 / (min_mc_stats ** 2)  # Minimum required events based on uncertainty

    # Sort data by the test data values for proper binning
    sorted_indices = np.argsort(test_data)
    data_sorted = test_data[sorted_indices]
    label_sorted = test_label[sorted_indices]
    weights_sorted = test_weights[sorted_indices]

    weights_sorted[label_sorted == 1] /= reweight_factor

    # Precompute total signal and background weights
    N_b = np.sum(weights_sorted[label_sorted == 0])
    N_s = np.sum(weights_sorted[label_sorted == 1])

    # Compute cumulative sums for signal and background counts
    cumulative_weights = np.cumsum(weights_sorted)
    cumulative_signal = np.cumsum(weights_sorted * (label_sorted == 1))
    cumulative_background = np.cumsum(weights_sorted * (label_sorted == 0))

    # Initialize bin starting point
    bin_start = 0

    while bin_start < total_events:
        # Try to find a suitable bin edge by iterating from the starting point
        for bin_end in range(bin_start + 1, total_events + 1):
            # Calculate weighted counts for signal and background in the current bin using cumulative sums
            n_s = cumulative_signal[bin_end - 1] - (cumulative_signal[bin_start - 1] if bin_start > 0 else 0)
            n_b = cumulative_background[bin_end - 1] - (cumulative_background[bin_start - 1] if bin_start > 0 else 0)

            # Apply TrafoD formula for significance
            Z = Zb * (n_b / N_b) + Zs * (n_s / N_s)

            # Check TrafoD criteria for minimum background and MC stats
            if Z > 1 and n_b >= min_bkg_per_bin and (n_b + n_s) >= mc_threshold:
                bin_edges.append(data_sorted[bin_end - 1])  # Use the last value as the bin edge
                bin_start = bin_end  # Move to the next starting point
                break
        else:
            # If no bin end satisfies the criteria, stop the binning process
            break

    # Add the last bin edge to cover all data
    bin_edges.append(data_sorted[-1])

    return bin_edges


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
        Zb=5, Zs=5, min_bkg_per_bin=3, min_mc_stats=1.0, method="asimov",
        reweight_factor=1,
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
    bin_edges = trafoD_binning(
        test_data, test_label, test_weights, Zb, Zs, min_bkg_per_bin, min_mc_stats, reweight_factor
    )

    # Step 2: Initialize lists to store signal and background counts per bin
    signal_counts = []
    background_counts = []

    # Step 3: Calculate counts in each bin
    for i in range(len(bin_edges) - 1):
        # Select events within the bin range
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

    return bin_edges, sum(significances)
