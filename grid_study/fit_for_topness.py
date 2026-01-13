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
from scipy.optimize import curve_fit
import warnings

# --- Setup ---
warnings.filterwarnings("ignore", module="coffea")
warnings.filterwarnings("ignore", category=RuntimeWarning)
vector.register_awkward()
plt.style.use(hep.style.CMS)


# ==========================================
# 1. Calibration Processor
# ==========================================
class SigmaCalibrationProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="Dataset", growth=True)

        self._hists = {}

        # -- Physics Plots --
        self._hists["Whad_res"] = hist.Hist(dataset_axis,
                                            hist.axis.Regular(60, 40, 120, name="mass", label="W_{had} Mass [GeV]"),
                                            storage=hist.storage.Weight())
        self._hists["Thad_res"] = hist.Hist(dataset_axis,
                                            hist.axis.Regular(60, 100, 250, name="mass", label="t_{had} Mass [GeV]"),
                                            storage=hist.storage.Weight())
        self._hists["Tlep_res"] = hist.Hist(dataset_axis,
                                            hist.axis.Regular(60, 100, 250, name="mass", label="t_{lep} Mass [GeV]"),
                                            storage=hist.storage.Weight())

        # -- Debug: Cutflow (Raw Counts) --
        self._hists["cutflow"] = hist.Hist(
            dataset_axis,
            hist.axis.StrCategory([], name="cut", label="Selection Step", growth=True),
            storage=hist.storage.Weight()
        )

        # -- Debug: Aux Plots --
        self._hists["dR_match"] = hist.Hist(
            dataset_axis,
            hist.axis.Regular(50, 0, 1.0, name="dr", label="Min dR(Gen, Reco)"),
            storage=hist.storage.Weight()
        )
        self._hists["n_matched"] = hist.Hist(
            dataset_axis,
            hist.axis.Regular(6, 0, 6, name="n", label="Number of Matched Objects (out of 4)"),
            storage=hist.storage.Weight()
        )

    @property
    def accumulator(self):
        return self._hists

    def process(self, events):
        dataset = events.metadata["dataset"]

        # --- Helper: Fill Cutflow with Raw Counts ---
        def fill_cutflow(step_name, ev_subset):
            N = len(ev_subset)
            if N > 0:
                self._hists["cutflow"].fill(
                    dataset=np.full(N, dataset),
                    cut=np.full(N, step_name),
                    weight=np.ones(N)
                )

        fill_cutflow("0_Initial", events)

        # =================================================================
        # 1. Reco Selection
        # =================================================================
        ele = events.Electron
        ele["iso"] = ele.pfRelIso03_all
        mu = events.Muon
        mu["iso"] = mu.pfRelIso04_all

        good_ele = ele[(ele.pt > 35) & (abs(ele.eta) < 2.5) & (ele.mvaFall17V2Iso_WP90) & (abs(ele.dxy) < 0.045) & (
                    abs(ele.dz) < 0.2)]
        good_mu = mu[(mu.pt > 30) & (abs(mu.eta) < 2.4) & (mu.iso < 0.15) & (mu.mediumId) & (abs(mu.dxy) < 0.045) & (
                    abs(mu.dz) < 0.2)]
        good_jet = events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
                            & (events.Jet.jetId >= 4) & ~((events.Jet.pt < 50) & ~(events.Jet.puId == 7))]

        # Cleaning
        def dr_clean(obj, ref, dr=0.4):
            nearest = obj.nearest(ref)
            dR = obj.delta_r(nearest)
            return obj[ak.fill_none(dR, 999) > dr]

        good_jet = dr_clean(good_jet, good_ele)
        good_jet = dr_clean(good_jet, good_mu)


        leptons = ak.concatenate([good_ele, good_mu], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]

        event_mask = (ak.num(leptons) == 1) & (ak.num(good_jet) >= 4)

        events = events[event_mask]
        leptons = leptons[event_mask]
        good_jet = good_jet[event_mask]
        met = events.MET  # auto-sliced

        fill_cutflow("1_RecoSelection", events)
        if len(events) == 0: return self._hists

        # =================================================================
        # 2. Gen Matching (Using distinctParent)
        # =================================================================
        partons = events.GenPart

        # A. Find Partons based on lineage
        q_from_W = partons[
            (abs(partons.pdgId) <= 5) &
            (abs(partons.distinctParent.pdgId) == 24) &
            (partons.hasFlags(['isLastCopy']))
        ]
        q_from_W = ak.drop_none(q_from_W)

        lep_from_W = partons[
            (abs(partons.pdgId) >= 11) & (abs(partons.pdgId) <= 16) &
            (abs(partons.distinctParent.pdgId) == 24) &
            (partons.hasFlags(['isLastCopy']))
        ]
        lep_from_W = ak.drop_none(lep_from_W)

        b_from_Top = partons[
            (abs(partons.pdgId) == 5) &
            (abs(partons.distinctParent.pdgId) == 6) &
            (partons.hasFlags(['isLastCopy']))
        ]
        b_from_Top = ak.drop_none(b_from_Top)

        # Filter: Semileptonic Topology
        is_semilep = (
                (ak.num(q_from_W) >= 2) &
                (ak.num(lep_from_W) >= 2) &
                (ak.num(b_from_Top) >= 2)
        )
        events = events[is_semilep]
        fill_cutflow("2_GenSemilep", events)
        if len(events) == 0: return self._hists

        # Filter Arrays
        q_from_W = q_from_W[is_semilep]
        b_from_Top = b_from_Top[is_semilep]
        good_jet = good_jet[is_semilep]
        leptons = leptons[is_semilep]
        met = met[is_semilep]

        # --- Resolve Ambiguities ---
        q1 = q_from_W[:, 0]
        q2 = q_from_W[:, 1]

        w_had_gen = q1.distinctParent
        t_had_gen = w_had_gen.distinctParent

        # Identify b_had
        b_parents = b_from_Top.distinctParent
        is_b_had = b_parents.delta_r(t_had_gen) < 0.01

        b_had_cands = b_from_Top[is_b_had]
        b_lep_cands = b_from_Top[~is_b_had]

        clean_split = (ak.num(b_had_cands) == 1) & (ak.num(b_lep_cands) == 1)
        events = events[clean_split]
        fill_cutflow("3_GenTopologyClean", events)

        if len(events) == 0: return self._hists

        # Final Objects
        good_jet = good_jet[clean_split]
        leptons = leptons[clean_split]
        met = met[clean_split]

        q1 = q1[clean_split]
        q2 = q2[clean_split]
        b_had = ak.firsts(b_had_cands[clean_split])
        b_lep = ak.firsts(b_lep_cands[clean_split])


        def get_p4(obj):
            return ak.zip(
                {"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass, "idx": ak.local_index(obj, axis=1)},
                with_name="PtEtaPhiMLorentzVector"
            )
        # =================================================================
        # 3. Reco Matching (Gen -> Reco)
        # =================================================================
        def get_match_info(reco, gen):
            # We want the RECO jet nearest to the GEN particle
            near = gen.nearest(reco)
            dr = near.delta_r(gen)
            return ak.firsts(near), ak.firsts(dr)

        good_jet = get_p4(good_jet)
        good_jet["index"] = ak.local_index(good_jet, axis=1)
        j_q1, dr_q1 = get_match_info(good_jet, q1)
        j_q2, dr_q2 = get_match_info(good_jet, q2)
        j_bhad, dr_bhad = get_match_info(good_jet, b_had)
        j_blep, dr_blep = get_match_info(good_jet, b_lep)

        idxs = ak.concatenate(
            [
                j_q1["index"][:, None],
                j_q2["index"][:, None],
                j_bhad["index"][:, None],
                j_blep["index"][:, None]
            ],
            axis=1,
        )  # shape: (events, 4)
        print("idxs:", idxs)

        # Uniqueness check: sort then look for adjacent duplicates
        idxs_filled = ak.fill_none(idxs, -1)  # sentinel
        sorted_idxs = ak.sort(idxs_filled, axis=1)
        has_dup = ak.any(sorted_idxs[:, 1:] == sorted_idxs[:, :-1], axis=1)

        unique_veto_mask = (~has_dup)

        # Debug Plots
        N_debug = len(events)
        if N_debug > 0:
            def safe_fill(hname, arr):
                # Flatten and remove Nones for plotting dR
                flat = ak.flatten(ak.fill_none(arr, -99), axis=None)
                flat = flat[flat > -1]
                if len(flat) > 0:
                    self._hists[hname].fill(dataset=dataset, **{hname.split("_")[0]: flat})

            # safe_fill("dR_match", dr_q1)


        pass_q1 = ak.fill_none(dr_q1, 999.0) < 0.4
        pass_q2 = ak.fill_none(dr_q2, 999.0) < 0.4
        pass_bhad = ak.fill_none(dr_bhad, 999.0) < 0.4
        pass_blep = ak.fill_none(dr_blep, 999.0) < 0.4

        found_all = pass_q1 & pass_q2 & pass_bhad & pass_blep & unique_veto_mask

        # Count matched objects
        n_matched_ak = (
                ak.values_astype(pass_q1, "int32") +
                ak.values_astype(pass_q2, "int32") +
                ak.values_astype(pass_bhad, "int32") +
                ak.values_astype(pass_blep, "int32")
        )

        # FIX: Flatten the numpy array to ensure it is 1D
        n_matched_np = ak.to_numpy(ak.fill_none(n_matched_ak, 0)).flatten()

        self._hists["n_matched"].fill(dataset=np.full(len(n_matched_np), dataset), n=n_matched_np)

        events = events[found_all]
        fill_cutflow("4_FullRecoMatch", events)

        if len(events) == 0: return self._hists

        j_q1 = j_q1[found_all]
        j_q2 = j_q2[found_all]
        j_bhad = j_bhad[found_all]
        j_blep = j_blep[found_all]
        l1 = leptons[found_all][:, 0]
        met_sel = met[found_all]

        # =================================================================
        # 4. Reconstruction
        # =================================================================
        w_had_reco = j_q1 + j_q2
        t_had_reco = w_had_reco + j_bhad

        MW_PDG = 80.4
        l1_p4 = ak.zip({"pt": l1.pt, "eta": l1.eta, "phi": l1.phi, "mass": l1.mass}, with_name="PtEtaPhiMLorentzVector")
        metpt_safe = ak.where(met_sel.pt > 0, met_sel.pt, 1e-6)
        met = met_sel

        mu = (MW_PDG ** 2) / 2.0 + l1_p4.pt * met.pt * np.cos(l1_p4.phi - met.phi)
        A = mu * l1_p4.pz / (l1_p4.pt ** 2)
        B = (mu ** 2 * l1_p4.pz ** 2) / (l1_p4.pt ** 4) - (
                (l1_p4.energy ** 2 * met.pt ** 2 - mu ** 2) / (l1_p4.pt ** 2)
        )
        cand_ans = ak.where(abs(A - np.sqrt(np.maximum(B, 0))) < abs(A + np.sqrt(np.maximum(B, 0))),
                            A - np.sqrt(np.maximum(B, 0)), A + np.sqrt(np.maximum(B, 0)))
        pz_nu = ak.where(B < 0, A, cand_ans)
        nu_p4 = ak.zip(
            {
                "pt": met.pt,
                "eta": np.arcsinh(pz_nu / metpt_safe),
                "phi": met.phi,
                "mass": 0.0,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        w_lep_reco = l1_p4 + nu_p4
        t_lep_reco = w_lep_reco + j_blep

        N_fin = len(events)
        self._hists["Whad_res"].fill(dataset=np.full(N_fin, dataset), mass=w_had_reco.mass)
        self._hists["Thad_res"].fill(dataset=np.full(N_fin, dataset), mass=t_had_reco.mass)
        self._hists["Tlep_res"].fill(dataset=np.full(N_fin, dataset), mass=t_lep_reco.mass)

        return self._hists

    def postprocess(self, accumulator):
        return accumulator


# ==========================================
# 2. Main & Plotting
# ==========================================
def plot_debug(hists, outdir):
    h_cf = hists["cutflow"][{"dataset": sum}]
    fig, ax = plt.subplots(figsize=(10, 6))
    cuts = [x for x in h_cf.axes[0]]
    counts = h_cf.values()

    ax.plot(cuts, counts, marker='o', linestyle='-', linewidth=2)
    for i, txt in enumerate(counts):
        ax.annotate(f"{int(txt)}", (i, counts[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    ax.set_ylabel("Raw Events")
    ax.set_title("Selection Efficiency Cutflow")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / "debug_cutflow.png")
    plt.close()

    h_n = hists["n_matched"][{"dataset": sum}]
    fig, ax = plt.subplots()
    hep.histplot(h_n, ax=ax)
    ax.set_xlabel("Number of Gen Objects Matched (Max 4)")
    ax.set_title("Matching Efficiency")
    plt.savefig(outdir / "debug_n_matched.png")
    plt.close()

    h_dr = hists["dR_match"][{"dataset": sum}]
    fig, ax = plt.subplots()
    hep.histplot(h_dr, ax=ax)
    ax.set_xlabel("Min dR(Gen, Reco)")
    ax.set_title("Matching Quality")
    plt.savefig(outdir / "debug_dr.png")
    plt.close()


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def perform_fit(hist_obj, ax, title):
    h_vals = hist_obj.values()
    h_edges = hist_obj.axes[0].edges
    h_centers = (h_edges[:-1] + h_edges[1:]) / 2

    mask = h_vals > 0
    x = h_centers[mask]
    y = h_vals[mask]

    if len(x) < 5 or np.sum(y) < 5:
        ax.text(0.5, 0.5, f"Not enough data\nN={np.sum(y)}", transform=ax.transAxes, ha='center')
        return None

    mean_guess = x[np.argmax(y)]
    sigma_guess = 15.0
    amp_guess = np.max(y)

    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=[amp_guess, mean_guess, sigma_guess], maxfev=5000)
        fit_amp, fit_mean, fit_sigma = popt
        fit_sigma = abs(fit_sigma)

        hep.histplot(hist_obj, ax=ax, histtype="errorbar", color="black", label="Data")
        x_fine = np.linspace(h_edges[0], h_edges[-1], 200)
        ax.plot(x_fine, gaussian(x_fine, *popt), color="red", linewidth=2,
                label=fr"Fit:\n$\mu={fit_mean:.1f}$\n$\sigma={fit_sigma:.2f}$")

        ax.set_title(title)
        ax.legend()
        return fit_sigma
    except Exception as e:
        print(f"Fit failed for {title}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Input JSON")
    parser.add_argument("--outdir", default="sigma_debug")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.json) as f:
        fileset_raw = json.load(f)

    fileset = {}
    if "background" in fileset_raw:
        for k, v in fileset_raw["background"].items():
            if "tt1l" in k.lower(): fileset[k] = v
    else:
        fileset = fileset_raw

    print(f"Running on: {list(fileset.keys())}")

    runner = processor.Runner(
        executor=processor.FuturesExecutor(workers=args.workers),
        schema=NanoAODSchema,
        chunksize=50_000,
        maxchunks=args.limit
    )

    results = runner(
        fileset,
        treename="Events",
        processor_instance=SigmaCalibrationProcessor()
    )

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating Debug Plots...")
    plot_debug(results, out_path)

    combined_hists = {
        "Whad_res": results["Whad_res"][{"dataset": sum}],
        "Thad_res": results["Thad_res"][{"dataset": sum}],
        "Tlep_res": results["Tlep_res"][{"dataset": sum}]
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sigma_mw = perform_fit(combined_hists["Whad_res"], axes[0], "W_{had} Resolution")
    sigma_mt_had = perform_fit(combined_hists["Thad_res"], axes[1], "t_{had} Resolution")
    sigma_mt_lep = perform_fit(combined_hists["Tlep_res"], axes[2], "t_{lep} Resolution")

    plt.tight_layout()
    plt.savefig(out_path / "calibration_result.png")

    print("\n" + "=" * 40)
    print(f"Sigma Results: MW={sigma_mw}, MT_had={sigma_mt_had}, MT_lep={sigma_mt_lep}")
    print("=" * 40)


if __name__ == "__main__":
    main()