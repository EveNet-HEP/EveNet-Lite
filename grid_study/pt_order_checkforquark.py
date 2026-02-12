import argparse
import json
import numpy as np
import awkward as ak
import vector
from pathlib import Path
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import warnings

# --- Setup ---
warnings.filterwarnings("ignore", module="coffea")
warnings.filterwarnings("ignore", category=RuntimeWarning)
vector.register_awkward()
plt.style.use(hep.style.CMS)


class JetRankCheckProcessor(processor.ProcessorABC):
    def __init__(self):
        # Axes
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="Dataset", growth=True)
        rank_axis = hist.axis.IntCategory(range(8), name="rank", label="Jet Rank (pT ordered)")

        self._hists = {}

        # Histograms
        self._hists["q1_rank"] = hist.Hist(dataset_axis, rank_axis, storage=hist.storage.Weight())
        self._hists["q2_rank"] = hist.Hist(dataset_axis, rank_axis, storage=hist.storage.Weight())

        # Correlation plot
        self._hists["rank_correlation"] = hist.Hist(
            dataset_axis,
            hist.axis.Integer(0, 6, name="q1_rank", label="q1 (Leading Gen) Jet Rank"),
            hist.axis.Integer(0, 6, name="q2_rank", label="q2 (Subleading Gen) Jet Rank"),
            storage=hist.storage.Weight()
        )

    @property
    def accumulator(self):
        return self._hists

    def process(self, events):
        dataset = events.metadata["dataset"]

        # =================================================================
        # 1. Object Selection
        # =================================================================
        ele = events.Electron
        good_ele = ele[(ele.pt > 35) & (abs(ele.eta) < 2.5) & (ele.mvaFall17V2Iso_WP90)]

        mu = events.Muon
        good_mu = mu[(mu.pt > 30) & (abs(mu.eta) < 2.4) & (mu.mediumId)]

        jets = events.Jet
        good_jet = jets[
            (jets.pt > 20) &
            (abs(jets.eta) < 2.4) &
            (jets.jetId >= 4) &
            ~((jets.pt < 50) & ~(jets.puId == 7))
            ]

        # Cleaning
        def dr_clean(obj, ref, dr=0.4):
            nearest = obj.nearest(ref)
            dR = obj.delta_r(nearest)
            return obj[ak.fill_none(dR, 999) > dr]

        good_jet = dr_clean(good_jet, good_ele)
        good_jet = dr_clean(good_jet, good_mu)

        # Sort Reco Jets by pT (High to Low)
        good_jet = good_jet[ak.argsort(good_jet.pt, axis=1, ascending=False)]
        good_jet["rank_idx"] = ak.local_index(good_jet, axis=1)

        # =================================================================
        # 2. Gen Matching (W -> q q')
        # =================================================================
        partons = events.GenPart
        q_from_W = partons[
            (abs(partons.pdgId) <= 5) &
            (abs(partons.distinctParent.pdgId) == 24) &
            (partons.hasFlags(['isLastCopy']))
            ]
        q_from_W = ak.drop_none(q_from_W)

        # Require at least 2 quarks from W
        has_W_decay = ak.num(q_from_W) >= 2
        events = events[has_W_decay]
        good_jet = good_jet[has_W_decay]
        q_from_W = q_from_W[has_W_decay]

        # Sort Gen Quarks by pT to define q1 (Lead) and q2 (Sublead)
        q_from_W = q_from_W[ak.argsort(q_from_W.pt, axis=1, ascending=False)]
        q1 = q_from_W[:, 0]
        q2 = q_from_W[:, 1]

        # =================================================================
        # 3. Match Gen -> Reco
        # =================================================================

        # --- Helper for Event-Level Matching ---
        def get_rank_per_event(gen_obj, jet_collection, dr_cut=0.4):
            nearest_jet = gen_obj.nearest(jet_collection)
            dr = gen_obj.delta_r(nearest_jet)
            rank = ak.fill_none(nearest_jet.rank_idx, -1)
            pass_dr = ak.fill_none(dr < dr_cut, False)
            return ak.where(pass_dr, rank, -1)

        r1_ev = get_rank_per_event(q1, good_jet)
        r2_ev = get_rank_per_event(q2, good_jet)

        # --- Fill 1D Histograms (Flattening) ---
        r1_flat = ak.flatten(r1_ev, axis=None)
        r2_flat = ak.flatten(r2_ev, axis=None)

        # Remove -1s (failed matches)
        r1_flat = r1_flat[r1_flat >= 0]
        r2_flat = r2_flat[r2_flat >= 0]

        if len(r1_flat) > 0:
            self._hists["q1_rank"].fill(dataset=dataset, rank=r1_flat)
        if len(r2_flat) > 0:
            self._hists["q2_rank"].fill(dataset=dataset, rank=r2_flat)

        # --- Fill Correlation Histogram (Event Aligned) ---
        # We need r1 and r2 from the SAME event.
        # Ensure we are looking at single values per event (pad/fill ensures length match)
        r1_val = ak.flatten(ak.fill_none(ak.pad_none(r1_ev, 1), -1))
        r2_val = ak.flatten(ak.fill_none(ak.pad_none(r2_ev, 1), -1))

        # Mask where both matches were successful
        mask_both = (r1_val >= 0) & (r2_val >= 0)

        r1_corr = r1_val[mask_both]
        r2_corr = r2_val[mask_both]

        if len(r1_corr) > 0:
            self._hists["rank_correlation"].fill(dataset=dataset, q1_rank=r1_corr, q2_rank=r2_corr)

        return self._hists

    def postprocess(self, accumulator):
        return accumulator


# ==========================================
# Plotting
# ==========================================
def plot_results(hists, outdir):
    # Plot 1: 1D Distribution of Ranks
    for name, title in [("q1_rank", "Leading Gen Quark (from W)"), ("q2_rank", "Subleading Gen Quark (from W)")]:
        h = hists[name]
        fig, ax = plt.subplots(figsize=(10, 7))

        # --- FIXED: Slice by dataset manually ---
        datasets = list(h.axes[0])  # Get list of dataset names
        h_list = []
        labels = []

        for ds in datasets:
            # Select the 1D histogram for this dataset
            h_slice = h[{"dataset": ds}]

            # Skip if empty to avoid plotting errors
            if h_slice.values().sum() > 0:
                h_list.append(h_slice)
                labels.append(ds)

        if len(h_list) > 0:
            # Pass list of 1D histograms to mplhep
            hep.histplot(h_list, label=labels, ax=ax, stack=False, density=True, yerr=False)
            ax.legend(title="Dataset")
        else:
            ax.text(0.5, 0.5, "No Events Matched", ha='center', transform=ax.transAxes)

        ax.set_xlabel("Matched Jet Rank (0=Leading, 1=Subleading...)")
        ax.set_ylabel("Density (Normalized)")
        ax.set_title(f"{title}")
        ax.set_xticks(range(8))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"rank_check_{name}.png")
        plt.close()

    # Plot 2: Correlation Matrix
    # Sum over datasets to get a global picture
    h_corr = hists["rank_correlation"][{"dataset": sum}]

    if h_corr.values().sum() > 0:
        fig, ax = plt.subplots(figsize=(9, 8))

        # Get counts and edges
        w, x_edges, y_edges = h_corr.to_numpy()

        # Normalize to percentages
        w_norm = np.divide(w, np.sum(w), out=np.zeros_like(w), where=np.sum(w) != 0) * 100

        # Plot heatmap
        im = ax.imshow(w_norm.T, origin='lower', cmap='viridis', aspect='auto')

        # Annotate
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                val = w_norm[i, j]
                text = ax.text(i, j, f"{val:.1f}%",
                               ha="center", va="center",
                               color="w" if val < 50 else "k",
                               fontsize=10)

        ax.set_xlabel("q1 (Leading Gen) Rank")
        ax.set_ylabel("q2 (Subleading Gen) Rank")
        ax.set_title("Correlation of Jet Ranks (q1 vs q2)")
        ax.set_xticks(np.arange(len(x_edges) - 1))
        ax.set_yticks(np.arange(len(y_edges) - 1))
        plt.colorbar(im, label="Percentage of Events [%]")
        plt.tight_layout()
        plt.savefig(outdir / "rank_correlation.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Input JSON")
    parser.add_argument("--outdir", default="rank_check_output")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files per dataset for testing")
    args = parser.parse_args()

    with open(args.json) as f:
        fileset_raw = json.load(f)

    # Flatten Fileset
    fileset = {}
    for top_key, value in fileset_raw.items():
        if "sig" not in top_key:
            continue
        if isinstance(value, list):
            fileset[top_key] = value
        elif isinstance(value, dict) and "files" in value:
            fileset[top_key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_key = f"{top_key}_{sub_key}"
                if isinstance(sub_value, list):
                    fileset[new_key] = sub_value
                elif isinstance(sub_value, dict) and "files" in sub_value:
                    fileset[new_key] = sub_value

    runner = processor.Runner(
        executor=processor.FuturesExecutor(workers=args.workers),
        schema=NanoAODSchema,
        chunksize=50_000,
        maxchunks=args.limit,
    )

    results = runner(
        fileset,
        treename="Events",
        processor_instance=JetRankCheckProcessor()
    )

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating Plots...")
    plot_results(results, out_path)
    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()