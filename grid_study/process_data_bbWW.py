import argparse
import json
import numpy as np
import awkward as ak
import torch
import vector
from pathlib import Path
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.processor import Runner
from coffea.processor import FuturesExecutor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import warnings
import uuid
import collections
from accumulators import DQMAccumulator, _edges

# Suppress performance warnings
warnings.filterwarnings("ignore", module="coffea")
# Enable vector behavior
vector.register_awkward()
import matplotlib.pyplot as plt
import mplhep as hep  # Recommended for HEP style plots: pip install mplhep

try:
    plt.style.use(hep.style.CMS)
except Exception:
    pass

MW_PDG = 80.4 # https://pdg.lbl.gov/2025/tables/contents_tables.html
MT_PDG = 172.5 # https://pdg.lbl.gov/2025/tables/contents_tables.html
MH_PDG = 125.2 # https://pdg.lbl.gov/2025/tables/contents_tables.html
PI = np.pi

HIST_DEFS = {
    # event weight
    "event_weight": _edges(60, -1, 9),
    # EveNet Point Cloud
    "x_E": _edges(60, 0, 1000), "x_pt": _edges(60, 0, 600),
    "x_eta": _edges(50, -3, 3), "x_phi": _edges(50, -3.14, 3.14),
    "x_isbtag": np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float64),
    "x_isLepton": np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64),
    "x_charge": np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64),

    # EveNet Global (Full 10 vars)
    "g_met": _edges(60, 0, 600), "g_met_phi": _edges(50, -3.14, 3.14),
    "g_nLepton": np.arange(-0.5, 6.5, 1.0), "g_nbJet": np.arange(-0.5, 10.5, 1.0), "g_nJet": np.arange(-0.5, 16.5, 1.0),
    "g_HT": _edges(60, 0, 2500), "g_HT_lep": _edges(60, 0, 1000),
    "g_M_all": _edges(60, 0, 4000), "g_M_leps": _edges(60, 0, 1500), "g_M_bjets": _edges(60, 0, 2000),

    # --- Raw low-level ---
    "lep_pt":  _edges(60, 0, 600),
    "lep_eta": _edges(50, -2.5, 2.5),
    "lep_phi": _edges(50, -PI, PI),
    "lep_m":   _edges(50, 0, 5),

    "met_pt":  _edges(60, 0, 800),
    "met_eta": _edges(20, -0.5, 0.5),   # always ~0 by construction
    "met_phi": _edges(50, -PI, PI),
    "met_m":   _edges(20, -0.5, 0.5),   # always 0 by construction

    "b1_pt":   _edges(60, 0, 800),
    "b1_eta":  _edges(50, -2.5, 2.5),
    "b1_phi":  _edges(50, -PI, PI),
    "b1_m":    _edges(60, 0, 300),

    "b2_pt":   _edges(60, 0, 800),
    "b2_eta":  _edges(50, -2.5, 2.5),
    "b2_phi":  _edges(50, -PI, PI),
    "b2_m":    _edges(60, 0, 300),

    "n_b_jets": _edges(7, -0.5, 6.5),

    # --- bb system ---
    "bb_inv": _edges(60, 0, 1500),
    "bb_dR":  _edges(50, 0, 6),
    "bb_mT":  _edges(60, 0, 3000),

    "W_mT":  _edges(60, 0, 3000),

    "topness": _edges(100, 0, 100),
    "top_dr_lb": _edges(50, 0, 6),
    "top_m_lb": _edges(60, 0, 300),
    "top_tlep_reco_mass": _edges(60, 0, 300),
    "top_whad_reco_mass": _edges(60, 0, 120),
    "top_whad_dr_qq": _edges(50, 0, 6),
    "top_thad_reco_mass": _edges(60, 0, 400),
    "top_thad_dr_bqq": _edges(50, 0, 6),

}

# --- Per-hypothesis features ---
for h in ["A", "B"]:
    HIST_DEFS.update({
        # W_had (qq)
        f"{h}_Whad_m":  _edges(60, 0, 500),
        f"{h}_Whad_dR": _edges(50, 0, 6),
        f"{h}_Whad_pt": _edges(60, 0, 800),

        f"{h}_q1_pt":_edges(60, 0, 800),
        f"{h}_q2_pt": _edges(60, 0, 800),
        f"{h}_q1_eta":  _edges(50, -2.5, 2.5),
        f"{h}_q2_eta":  _edges(50, -2.5, 2.5),

        # WW visible (l + qq)
        f"{h}_WW_vis_m":  _edges(60, 0, 2000),
        f"{h}_WW_vis_dR": _edges(50, 0, 6),
        f"{h}_WW_mT":     _edges(60, 0, 2000),

        # bbWW visible / cluster mT (X proxy)
        f"{h}_bbWW_vis_m":  _edges(60, 0, 4000),
        f"{h}_bbWW_vis_dR": _edges(50, 0, 6),
        f"{h}_bbWW_mT":     _edges(60, 0, 4000),

        # Cross-terms
        f"{h}_dPhi_bb_Whad": _edges(50, 0, PI),  # abs(dphi)
        f"{h}_min_dR_l_q":   _edges(50, 0, 6),
    })


# --- Add these OUTSIDE your class, near imports ---
def cutflow_factory():
    return processor.defaultdict_accumulator(int)

def nested_dict_int():
    """Helper to replace lambda: collections.defaultdict(int)"""
    return collections.defaultdict(int)

def dqm_factory():
    """Helper to replace lambda: DQMAccumulator(HIST_DEFS)"""
    # Assuming HIST_DEFS is a global variable defined earlier in the file
    return DQMAccumulator(HIST_DEFS)

def plot_dqm(dqm_data, dataset_name, out_dir):

    plot_dir = Path(out_dir) / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    hists = dqm_data["hists"]
    hist_defs = dqm_data["hist_defs"]

    for name, counts_dict in hists.items():
        edges = hist_defs[name]
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1:] - edges[:-1]

        counts_tr = counts_dict["train"].astype(np.float64)
        counts_va = counts_dict["valid"].astype(np.float64)

        # --- Density ---
        sum_tr = np.sum(counts_tr)
        sum_va = np.sum(counts_va)

        if sum_tr == 0 or sum_va == 0:
            continue

        dens_tr = counts_tr / (sum_tr * width)
        dens_va = counts_va / (sum_va * width)

        # --- (Main Pad + Ratio Pad) ---
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [3, 1]},
            figsize=(10, 8),
            sharex=True
        )

        # 1. Main Plot (Density)
        ax_main.step(centers, dens_tr, where='mid', label=f'Train ({int(sum_tr)})', color='black', linewidth=1.5)
        ax_main.step(centers, dens_va, where='mid', label=f'Valid ({int(sum_va)})', color='red', linestyle='--',
                     linewidth=1.5)

        # Error bars (Poisson approximation)
        err_tr = np.sqrt(counts_tr) / (sum_tr * width)
        err_va = np.sqrt(counts_va) / (sum_va * width)

        ax_main.errorbar(centers, dens_tr, yerr=err_tr, fmt='none', color='black')
        ax_main.errorbar(centers, dens_va, yerr=err_va, fmt='none', color='red')

        ax_main.set_ylabel("Density")
        ax_main.set_title(f"[{dataset_name}] {name}", loc='left', fontsize=16, fontweight='bold')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # 2. Ratio Plot (Valid / Train)
        # Avoid divide by zero
        ratio = np.divide(dens_va, dens_tr, out=np.ones_like(dens_va), where=dens_tr != 0)

        ax_ratio.step(centers, ratio, where='mid', color='blue', linewidth=1)
        ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel("Valid / Train")
        ax_ratio.set_xlabel(name)
        ax_ratio.grid(True, alpha=0.3)

        # Save
        plt.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=100)
        plt.close(fig)

    print(f"  Plots saved to {plot_dir}")

# ==========================================
# 2. Processor
# ==========================================
class FullLogicProcessor(processor.ProcessorABC):
    def __init__(self, config):
        self.cfg = config
        self._accumulator = processor.dict_accumulator({
            "cutflow": processor.defaultdict_accumulator(
                lambda: processor.defaultdict_accumulator(int)
            ),
            "dqm": processor.defaultdict_accumulator(lambda: DQMAccumulator(HIST_DEFS))
        })

    @property
    def accumulator(self):
        return self._accumulator

    def solve_neutrino(self, l1_p4, met):
        metpt_safe = ak.where(met.pt > 0, met.pt, 1e-6)

        mu = (MW_PDG ** 2) / 2.0 + l1_p4.pt * met.pt * np.cos(l1_p4.phi - met.phi)
        A = mu * l1_p4.pz / (l1_p4.pt ** 2)
        B = (mu ** 2 * l1_p4.pz ** 2) / (l1_p4.pt ** 4) - (
                (l1_p4.energy ** 2 * met.pt ** 2 - mu ** 2) / (l1_p4.pt ** 2)
        )
        cand_ans = ak.where(abs(A- np.sqrt(np.maximum(B, 0))) < abs(A + np.sqrt(np.maximum(B, 0))), A- np.sqrt(np.maximum(B, 0)), A + np.sqrt(np.maximum(B, 0)))
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
        return nu_p4


    def topness(self, leptons, jets, met):
        MW = MW_PDG
        MT = MT_PDG

        sigma_MW = 11.83
        sigma_MT_had = 20.87
        sigma_MT_lep = 28.72

        def p4_from_components(pt, eta, phi, mass):
            return ak.zip(
                {"pt": pt, "eta": eta, "phi": phi, "mass": mass},
                with_name="PtEtaPhiMLorentzVector",
            )

        def p4(obj):
            return p4_from_components(obj.pt, obj.eta, obj.phi, obj.mass)

        # ----------------------------
        # Basic masks and sorting
        # ----------------------------
        has_lep = ak.num(leptons, axis=1) > 0

        jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        jets_sorted["index"] = ak.local_index(jets_sorted, axis=1)
        is_btag = jets_sorted.btagDeepFlavB > self.cfg["btag_wp"]

        bjets = jets_sorted[is_btag]
        ljets = jets_sorted[~is_btag]
        # We need: >=1 lepton, >=2 bjets (one leptonic b + one hadronic b), >=2 light jets
        valid = has_lep & (ak.num(bjets, axis=1) >= 2) & (ak.num(ljets, axis=1) >= 2)
        # ----------------------------
        # Leading lepton (filled for safe math)
        # ----------------------------
        l1 = ak.firsts(leptons)
        l1_pt = ak.fill_none(l1.pt, 0.0)
        l1_eta = ak.fill_none(l1.eta, 0.0)
        l1_phi = ak.fill_none(l1.phi, 0.0)
        l1_mass = ak.fill_none(l1.mass, 0.0)
        l1_p4 = p4_from_components(l1_pt, l1_eta, l1_phi, l1_mass)

        # ----------------------------
        # Neutrino pz from W-mass constraint (safe for met.pt=0 and missing lepton)
        # ----------------------------

        nu_p4 = self.solve_neutrino(l1_p4, met)
        w_lep = l1_p4 + nu_p4

        # ----------------------------
        # Pick leptonic b: minimize |m(w_lep + b) - MT|
        # (index-free selection to avoid crashes on empty lists)
        # ----------------------------
        b_p4 = p4(bjets)
        m_wb = (w_lep[:, None] + b_p4).mass
        delta = np.abs(m_wb - MT)

        min_delta = ak.fill_none(ak.min(delta, axis=1), np.inf)
        best_mask = delta == min_delta[:, None]
        b_lep = ak.firsts(bjets[best_mask])

        # Build leptonic-top mass term (filled if b_lep missing)
        b_lep_pt = ak.fill_none(b_lep.pt, 0.0)
        b_lep_eta = ak.fill_none(b_lep.eta, 0.0)
        b_lep_phi = ak.fill_none(b_lep.phi, 0.0)
        b_lep_mass = ak.fill_none(b_lep.mass, 0.0)
        b_lep_p4 = p4_from_components(b_lep_pt, b_lep_eta, b_lep_phi, b_lep_mass)

        m_tlep = (w_lep + b_lep_p4).mass

        # Remaining b-jets (remove chosen leptonic b by index)
        best_idx = b_lep["index"]
        best_idx_filled = ak.fill_none(best_idx, -1)
        b_other = bjets[bjets["index"] != best_idx_filled[:, None]]
        b_other_p4 = p4(b_other)

        # ----------------------------
        # Hadronic W candidates from light-jet pairs
        # ----------------------------
        ljets_p4 = p4(ljets)
        wjj = ak.combinations(ljets_p4, 2, axis=1, fields=["j1", "j2"])
        w_had = wjj.j1 + wjj.j2

        # Pair each hadronic W with each remaining b (nested lists)
        pairs = ak.cartesian({"w": w_had, "b": b_other_p4}, axis=1, nested=True)

        m_whad = pairs.w.mass
        m_thad = (pairs.w + pairs.b).mass

        chi2 = (
                ((m_whad - MW) / sigma_MW) ** 2
                + ((m_thad - MT) / sigma_MT_had) ** 2
                + ((m_tlep[:, None, None] - MT) / sigma_MT_lep) ** 2
        )

        chi2_min_b = ak.min(chi2, axis=2)  # (events, nW)
        ib_best_for_w = ak.argmin(chi2, axis=2)  # (events, nW)

        iw_best = ak.argmin(chi2_min_b, axis=1)  # (events,)
        iw_best1 = ak.singletons(iw_best)  # (events, 1)

        # Best W (LorentzVector)
        whad_best = ak.firsts(w_had[iw_best1])  # (events,)
        # Best b-index for that chosen W
        ib_best = ak.firsts(ib_best_for_w[iw_best1])  # (events,)
        ib_best1 = ak.singletons(ib_best)  # (events, 1)

        # Best hadronic b (LorentzVector)
        bhad_best = ak.firsts(b_other_p4[ib_best1])  # (events,)

        # Now you can get qq dr from the record wjj
        wjj_best = ak.firsts(wjj[iw_best1])  # (events,)
        whad_dr_qq = wjj_best.j1.delta_r(wjj_best.j2)  # (events,)

        thad_m = (whad_best + bhad_best).mass
        thad_dr_bqq = whad_best.delta_r(bhad_best)

        # Global min over all hypotheses; if no hypotheses -> None
        chi2_best = ak.min(ak.min(chi2, axis=2), axis=1)
        # Get best hadronic W and hadronic top masses

        # Only keep "physically valid" events, otherwise return None
        return {
            "topness": ak.where(valid, chi2_best, 99999),
            "top_dr_lb": ak.where(valid, l1_p4.delta_r(b_lep_p4), 0),
            "top_m_lb": ak.where(valid, (l1_p4 + b_lep_p4).mass, 0),
            "top_tlep_reco_mass": ak.where(valid, m_tlep, 0),
            "top_whad_reco_mass": ak.where(valid, whad_best.mass, 0),
            "top_whad_dr_qq": ak.where(valid, whad_dr_qq, 0),
            "top_thad_reco_mass": ak.where(valid, thad_m, 0),
            "top_thad_dr_bqq": ak.where(valid, thad_dr_bqq, 0),
        }



    def process(self, events):
        # 1. You can use lambda here freely for convenience while processing
        #    (It's fine as long as we don't try to return it directly)
        temp_cutflow = collections.defaultdict(lambda: collections.defaultdict(int))
        temp_dqm = collections.defaultdict(lambda: DQMAccumulator(HIST_DEFS))

        dataset = events.metadata["dataset"]
        filename = events.metadata["filename"]
        temp_cutflow[dataset]["total"] += len(events)

        # --- 1. Object Prep (Keep needed branches) ---
        # Attach 'iso' explicitly for XGB sorting later
        ele = events.Electron
        ele["iso"] = ele.pfRelIso03_all
        mu = events.Muon
        mu["iso"] = mu.pfRelIso04_all

        # ID Selection
        good_ele = ele[(ele.pt > 35) & (abs(ele.eta) < 2.5) & (ele.mvaFall17V2Iso_WP90) & (abs(ele.dxy) < 0.045) & (abs(ele.dz)<0.2)]
        good_mu = mu[(mu.pt > 30) & (abs(mu.eta) < 2.4) & (mu.iso < 0.15) & (mu.mediumId) & (abs(mu.dxy) < 0.045) & (abs(mu.dz)<0.2)]
        good_tau = events.Tau[
            (events.Tau.pt > 30) & (abs(events.Tau.eta) < 2.3) & (abs(events.Tau.dz) < 0.2) & ((events.Tau.decayMode < 5) | (events.Tau.decayMode >= 10))
            & (events.Tau.idDeepTau2017v2p1VSjet >= 16) & (events.Tau.idDeepTau2017v2p1VSe >= 32) & (events.Tau.idDeepTau2017v2p1VSmu >= 1)]
        good_jet = events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
                            & (events.Jet.jetId >= 4) & ~((events.Jet.pt < 50) & ~(events.Jet.puId == 7))]

        def dr_clean(obj, ref, dr=0.4):
            nearest = obj.nearest(ref)
            dR = obj.delta_r(nearest)
            return obj[ak.fill_none(dR, 999) > dr]  # ref 空 -> dR=None -> 當作 999 -> 保留 obj

        # Cross Cleaning
        good_tau = dr_clean(good_tau, good_ele, 0.4)
        good_tau = dr_clean(good_tau, good_mu, 0.4)
        good_jet = dr_clean(good_jet, good_ele, 0.4)
        good_jet = dr_clean(good_jet, good_mu, 0.4)
        good_jet = dr_clean(good_jet, good_tau, 0.4)

        good_tau = ak.drop_none(good_tau, axis=1)
        good_jet = ak.drop_none(good_jet, axis=1)
        good_light_jet = good_jet[good_jet.btagDeepFlavB < self.cfg['btag_wp']]
        good_bjet = good_jet[good_jet.btagDeepFlavB > self.cfg['btag_wp']]

        def make_lep_p4(coll):
            return ak.zip({
                "pt": coll.pt,
                "eta": coll.eta,
                "phi": coll.phi,
                "mass": coll.mass,
                "charge": coll.charge,
                "iso": coll.iso
            }, with_name="PtEtaPhiMLorentzVector")
        def make_jet_p4(coll):
            return ak.zip({
                "pt": coll.pt,
                "eta": coll.eta,
                "phi": coll.phi,
                "mass": coll.mass
            }, with_name="PtEtaPhiMLorentzVector")

        good_ele = make_lep_p4(good_ele)
        good_mu = make_lep_p4(good_mu)
        # Lepton Merging (Keep 'iso' field)
        leptons = ak.concatenate([good_ele, good_mu], axis=1)
        # Sort by pT for standard selection logic
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]

        leading_lepton = ak.firsts(leptons)
        light_jets_p4 = make_jet_p4(good_light_jet)
        dr = light_jets_p4.delta_r(leading_lepton)
        sorted_jets = light_jets_p4[ak.argsort(dr, axis=1)]
        nearest_jet = ak.firsts(sorted_jets)
        dr_lj = ak.fill_none(leading_lepton.delta_r(nearest_jet), 999)
        topness = self.topness(leptons, good_jet, events.MET)["topness"]

        # --- 2. Event Selection ---
        selection = PackedSelection()
        selection.add("one_lep", ak.num(leptons) == 1)
        selection.add("had_tau_veto", ak.num(good_tau) == 0)
        selection.add("two_ljets", ak.num(good_light_jet) >= 2)
        selection.add("two_bjets", ak.num(good_bjet) >= 2)
        selection.add("dr_lj", dr_lj < 1.6)
        selection.add("topness", topness > 3)
        cut = selection.all("one_lep", "had_tau_veto", "two_ljets", "two_bjets", "dr_lj", "topness")

        sel_ev = events[cut]
        if len(sel_ev) == 0: return {
            "cutflow": {k: dict(v) for k, v in temp_cutflow.items()},
            "dqm": dict(temp_dqm)
        }
        temp_cutflow[dataset]["passed"] += len(sel_ev)

        # Slice collections
        sel_leps = leptons[cut]
        sel_taus = good_tau[cut]
        sel_jets = good_jet[cut]
        sel_met = events.MET[cut]

        # --- 3. Train/Valid Split ---
        fhash = abs(hash(filename)) % (2 ** 32 - 1)
        rng = np.random.default_rng(self.cfg['seed'] + fhash)
        is_train = rng.random(len(sel_ev)) < self.cfg['train_frac']

        temp_dqm[dataset].meta["n_train"] += int(np.sum(is_train))
        temp_dqm[dataset].meta["n_valid"] += int(np.sum(~is_train))

        # --- 4. Logic Execution ---

        entry_start = events.metadata['entrystart']
        entry_stop = events.metadata['entrystop']
        output_filename = f"{filename.replace('.root','')}_{entry_start}_{entry_stop}.root"


        # A. EveNet (Full 10 Globals, Max 18 Objs)
        x_ev, mask_ev, glob_ev, dqm_ev = self.get_evenet_features(sel_leps, sel_taus, sel_jets, sel_met)

        dqm_ev["event_weight"] = ak.to_numpy(sel_ev.genWeight / abs(sel_ev.genWeight))

        # B. XGB (Complex Sorting & Logic)
        x_xgb, name_xgb, dqm_xgb = self.get_xgb_features(sel_leps, sel_taus, sel_jets, sel_met)

        # --- 5. Saving ---
        self.save_file(dataset, output_filename, "evenet", ".pt", is_train,
                       {"x": x_ev, "x_mask": mask_ev, "global": glob_ev, "weights": ak.to_numpy(sel_ev.genWeight/abs(sel_ev.genWeight))})

        self.save_file(dataset, output_filename, "xgb", ".npz", is_train,
                       {"X": x_xgb, "features": name_xgb, "weights": ak.to_numpy(sel_ev.genWeight/abs(sel_ev.genWeight))})

        # --- 6. Filling DQM ---
        def fill_dqm(source, mask, split):
            for k, v in source.items():
                if k in HIST_DEFS:
                    temp_dqm[dataset].fill(k, split, v[mask])

        fill_dqm(dqm_ev, is_train, "train")
        fill_dqm(dqm_ev, ~is_train, "valid")
        fill_dqm(dqm_xgb, is_train, "train")
        fill_dqm(dqm_xgb, ~is_train, "valid")

        final_cutflow = {k: dict(v) for k, v in temp_cutflow.items()}
        final_dqm = dict(temp_dqm)

        return {
            "cutflow": final_cutflow,
            "dqm": final_dqm
        }

    def postprocess(self, accumulator):
        return accumulator
    # ==========================================
    # LOGIC A: EveNet (Exact Physics)
    # ==========================================
    def get_evenet_features(self, leptons, taus, jets, met):
        MAX_JETS = self.cfg['max_jets']  # 16
        MAX_OBJS = self.cfg['max_objs']  # 18

        # 1. Jet sorting & Cut (Max 16)
        jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        jet_btag = ak.values_astype((jets_sorted.btagDeepFlavB > self.cfg['btag_wp']), np.float32)

        # 2. Record Building
        def make_rec(coll, is_lep, is_btag_val, charge=None):
            coll_p4 = ak.zip({
                "pt": coll.pt,
                "eta": coll.eta,
                "phi": coll.phi,
                "mass": coll.mass
            }, with_name="PtEtaPhiMLorentzVector")
            return ak.zip({
                "E": coll_p4.E, "pt": coll.pt, "eta": coll.eta, "phi": coll.phi,
                "btag": is_btag_val,
                "isLepton": ak.full_like(coll.pt, is_lep, dtype=np.float32),
                "charge": ak.values_astype(charge, np.float32) if charge is not None else ak.full_like(coll.pt, 0.0, dtype=np.float32)
            })

        jet_rec = make_rec(jets_sorted, 0.0, jet_btag, ak.full_like(jets_sorted.pt, 0.0, dtype=np.float32))  # Jet: btag=0/1, isLep=0
        tau_rec = make_rec(taus, 0.0, ak.full_like(taus.pt, 2, dtype=np.float32), taus.charge)  # Tau: btag=2,   isLep=0
        lep_rec = make_rec(leptons, 1.0, ak.full_like(leptons.pt, 0, dtype=np.float32), leptons.charge)  # Lep: btag=0,   isLep=1

        # 3. Merge -> Sort -> Pad 18
        all_objs = ak.concatenate([lep_rec, tau_rec, jet_rec], axis=1)
        all_objs = all_objs[ak.argsort(all_objs.pt, axis=1, ascending=False)]

        padded = ak.pad_none(all_objs, MAX_OBJS, axis=1, clip=True)
        mask = (ak.fill_none(padded.pt, -1.0)) > -0.05
        # 4. Safe Numpy Conversion

        def safe(arr): return ak.to_numpy(ak.fill_none(arr, 0.0)).astype(np.float32)

        x_np = np.stack([
            safe(padded.E), safe(padded.pt), safe(padded.eta), safe(padded.phi),
            safe(padded.btag), safe(padded.isLepton), safe(padded.charge)
        ], axis=2)

        # 5. Global Features (Calculated from PADDED array to match tensor)
        # Re-construct 4-vectors from padded arrays
        pad_p4 = ak.zip({
            "pt": ak.fill_none(padded.pt, 0.0),
            "eta": ak.fill_none(padded.eta, 0.0),
            "phi": ak.fill_none(padded.phi, 0.0),
            "E": ak.fill_none(padded.E, 0.0)
        }, with_name="PtEtaPhiELorentzVector")

        is_lepton_padded = ak.fill_none(padded.isLepton, 0.0)
        is_btag_padded = ak.fill_none(padded.btag, 0.0)

        # Masks on padded array
        is_lep_mask = is_lepton_padded > 0.5
        # btag logic: > 0.5 and < 1.5 (Strictly jets, excludes Taus=2.0)
        is_bjet_mask = (is_btag_padded > 0.5) & (is_btag_padded < 1.5)
        # Valid objects (exists and not None)
        valid_mask = ak.fill_none(mask, False)

        # Summation
        def calc_mass(vec_slice):
            return ak.to_numpy(vec_slice.sum(axis=1).mass).astype(np.float32)

        g_met = safe(met.pt)
        g_met_phi = safe(met.phi)
        g_nLep = safe(ak.sum(is_lep_mask, axis=1))
        g_nbJet = safe(ak.sum(is_bjet_mask, axis=1))
        g_nJet = safe(ak.num(jets_sorted) + ak.num(taus))  # Original logic: Jets + Taus count
        g_HT = safe(ak.sum(pad_p4.pt * (~is_lep_mask), axis=1))
        g_HT_lep = safe(ak.sum(pad_p4.pt * is_lep_mask, axis=1))
        g_Mall = calc_mass(ak.mask(pad_p4, valid_mask))  # Sum all valid
        g_Mlep = calc_mass(ak.mask(pad_p4, is_lep_mask))
        g_Mbjet = calc_mass(ak.mask(pad_p4, is_bjet_mask))

        g_Mall = np.where(g_Mall > 0.0, g_Mall, 0.0)
        g_Mlep = np.where(g_Mlep > 0.0, g_Mlep, 0.0)
        g_Mbjet = np.where(g_Mbjet > 0.0, g_Mbjet, 0.0)

        g_np = np.stack([
            g_met, g_met_phi, g_nLep, g_nbJet, g_nJet,
            g_HT, g_HT_lep, g_Mall, g_Mlep, g_Mbjet
        ], axis=1)

        dqm_dict = {
            "x_E": padded.E, "x_pt": padded.pt, "x_eta": padded.eta, "x_phi": padded.phi,
            "x_isLepton": padded.isLepton, "x_isbtag": padded.btag, "x_charge": padded.charge,
            "g_met": g_met, "g_met_phi": g_met_phi, "g_nLepton": g_nLep,
            "g_nbJet": g_nbJet, "g_nJet": g_nJet,
            "g_HT": g_HT, "g_HT_lep": g_HT_lep,
            "g_M_all": g_Mall,
            "g_M_leps": g_Mlep,
            "g_M_bjets": g_Mbjet
        }

        return torch.from_numpy(x_np), torch.from_numpy(safe(mask).astype(bool)), torch.from_numpy(g_np), dqm_dict

    # ==========================================
    # LOGIC B: XGBoost (Exact Complex Logic)
    # ==========================================

    def get_xgb_features(self, leptons, taus, jets, met):
        # ==============================================================================
        # 0. Setup & Constants
        # ==============================================================================

        # --- Helpers ---
        def get_p4(obj):
            # NOTE: awkward records don't reliably work with hasattr(); use fields instead.
            fields = set(ak.fields(obj)) if obj is not None else set()

            return ak.zip(
                {"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass},
                with_name="PtEtaPhiMLorentzVector",
            )

        def safe(arr):
            return ak.to_numpy(ak.fill_none(arr, -999.0)).astype(np.float32)

        def save_p4(prefix, p4):
            return {
                f"{prefix}_pt": safe(p4.pt),
                f"{prefix}_eta": safe(p4.eta),
                f"{prefix}_phi": safe(p4.phi),
                f"{prefix}_m": safe(p4.mass),
            }

        # Cluster transverse mass:
        # mT^2 = (ET_sys + MET)^2 - (pT_sys_vec + MET_vec)^2
        def calc_mt(sys_p4, met_p4):
            et_sys = np.sqrt(sys_p4.pt ** 2 + sys_p4.mass ** 2)
            pt_sum_x = sys_p4.pt * np.cos(sys_p4.phi) + met_p4.pt * np.cos(met_p4.phi)
            pt_sum_y = sys_p4.pt * np.sin(sys_p4.phi) + met_p4.pt * np.sin(met_p4.phi)
            pt_sum_sq = pt_sum_x ** 2 + pt_sum_y ** 2
            return np.sqrt(np.maximum(0, (et_sys + met_p4.pt) ** 2 - pt_sum_sq))

        # ==============================================================================
        # 1. Object Preparation
        # ==============================================================================
        l1 = ak.firsts(leptons)
        l1_p4 = get_p4(l1)

        met_p4 = ak.zip(
            {"pt": met.pt, "eta": ak.zeros_like(met.pt), "phi": met.phi, "mass": 0.0},
            with_name="PtEtaPhiMLorentzVector",
        )
        nu_p4 = self.solve_neutrino(l1_p4, met)
        # Full leptonic W (for B2 selection logic)
        w_lep_full_p4 = l1_p4 + nu_p4

        # --- Jets / b-jets ---
        jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        is_btag = jets_sorted.btagDeepFlavB > self.cfg["btag_wp"]

        # IMPORTANT BUGFIX: count b-jets BEFORE padding (padding makes length look like 2 always)
        data = {}
        topness_arrays = self.topness(leptons, jets, met)
        for k, v in topness_arrays.items():
            data[k] = safe(v)
        data["n_b_jets"] = safe(ak.sum(is_btag, axis=1))

        # IMPORTANT CHANGE: choose bb pair as top-2 by btag score (not by pT)
        bjets = jets_sorted[is_btag]
        bjets_by_btag = bjets[ak.argsort(bjets.btagDeepFlavB, axis=1, ascending=False)]
        b_cands = ak.pad_none(bjets_by_btag, 2)
        b1_p4, b2_p4 = get_p4(b_cands[:, 0]), get_p4(b_cands[:, 1])
        bb_sys_p4 = b1_p4 + b2_p4

        # --- Light Jet Pairs (Candidate Pool) ---
        l_cands = jets_sorted[~is_btag]
        # take up to top-8 light jets per event (no padding => no Nones => no unions)
        l_cands_top8 = l_cands[:, :8]

        qq_pairs = ak.combinations(l_cands_top8, 2, axis=1)
        q1, q2 = ak.unzip(qq_pairs)
        q1_p4, q2_p4 = get_p4(q1), get_p4(q2)
        whad_cands_p4 = q1_p4 + q2_p4

        # ==============================================================================
        # 2. Selection Logics (The 3 Scenarios)
        # ==============================================================================
        # (A) W_had On-shell (Closest to MW)
        idx_A = ak.argmin(abs(whad_cands_p4.mass - MW_PDG), axis=1, keepdims=True)

        # (B) Higgs Constraint (Min |M(lnu+qq) - MH|)
        w_lep_broad = ak.broadcast_arrays(w_lep_full_p4, whad_cands_p4)[0]
        h_cands = w_lep_broad + whad_cands_p4
        idx_B = ak.argmin(abs(h_cands.mass - MH_PDG), axis=1, keepdims=True)

        # ==============================================================================
        # 3. Feature Assembly
        # ==============================================================================
        # Save raw low-level
        data.update(save_p4("lep", l1_p4))
        data["met_pt"] = safe(met.pt)
        data["met_phi"] = safe(met.phi)
        data["W_mT"] = safe(calc_mt(l1_p4, met_p4))
        data.update(save_p4("b1", b1_p4))
        data.update(save_p4("b2", b2_p4))

        # Shared bb variables
        data["bb_inv"] = safe(bb_sys_p4.mass)
        data["bb_dR"] = safe(b1_p4.delta_r(b2_p4))


        hypos = {"A": idx_A, "B": idx_B}

        for name, idx in hypos.items():
            # Retrieve the specific W_had candidate
            w_qq_curr = ak.firsts(whad_cands_p4[idx])
            q1_curr = ak.firsts(q1_p4[idx])
            q2_curr = ak.firsts(q2_p4[idx])

            # Systems
            ww_vis_p4 = l1_p4 + w_qq_curr  # visible WW (no nu)
            bbww_vis_p4 = bb_sys_p4 + ww_vis_p4  # visible X proxy

            # [W_had system]
            data[f"{name}_Whad_m"] = safe(w_qq_curr.mass)
            data[f"{name}_Whad_dR"] = safe(q1_curr.delta_r(q2_curr))
            data[f"{name}_Whad_pt"] = safe(w_qq_curr.pt)
            data.update(save_p4(f"{name}_q1", q1_curr))
            data.update(save_p4(f"{name}_q2", q2_curr))

            # [WW system]
            data[f"{name}_WW_vis_m"] = safe(ww_vis_p4.mass)
            data[f"{name}_WW_vis_dR"] = safe(l1_p4.delta_r(w_qq_curr))
            data[f"{name}_WW_mT"] = safe(calc_mt(ww_vis_p4, met_p4))

            # [bbWW system]
            data[f"{name}_bbWW_vis_m"] = safe(bbww_vis_p4.mass)
            data[f"{name}_bbWW_vis_dR"] = safe(bb_sys_p4.delta_r(ww_vis_p4))
            data[f"{name}_bbWW_mT"] = safe(calc_mt(bbww_vis_p4, met_p4))

            # [Cross terms]
            data[f"{name}_dPhi_bb_Whad"] = safe(abs(bb_sys_p4.delta_phi(w_qq_curr)))
            data[f"{name}_min_dR_l_q"] = safe(np.minimum(l1_p4.delta_r(q1_curr), l1_p4.delta_r(q2_curr)))

        keys = sorted(list(data.keys()))
        matrix = np.stack([data[k] for k in keys], axis=1)
        return matrix, keys, data

    # ==========================================
    # Helper: Save
    # ==========================================
    def save_file(self, dataset, filename, subdir, ext, train_mask, data_dict):
        base = Path(self.cfg['outdir']) / dataset / subdir
        base.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem
        # uid = str(uuid.uuid4())[:6]

        for split, mask in [("train", train_mask), ("valid", ~train_mask)]:
            out_dir = base / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_{ext}"

            sliced = {}
            for k, v in data_dict.items():
                if isinstance(v, list):
                    sliced[k] = v
                elif isinstance(v, torch.Tensor):
                    sliced[k] = v[torch.from_numpy(mask)]
                else:
                    sliced[k] = v[mask]

            if ext == ".pt":
                torch.save(sliced, out_path)
            else:
                np.savez_compressed(out_path, **sliced)


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Input JSON {dataset: [files]}")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.json) as f: fileset = json.load(f)

    fileset_for_coffea = {}
    for category, sample_dict in fileset.items():
        for name, sample_list in sample_dict.items():
            fileset_for_coffea[name] = {
                "files": sample_list,
                "metadata": {"sample_name": name}
            }

    config = {
        "outdir": args.outdir,
        "btag_wp": 0.2489,  # 2016postapv WPs, medium
        "max_objs": 18,
        "max_jets": 16,
        "seed": 42,
        "train_frac": 0.5
    }

    print(f"Running Full Logic Processor with {args.workers} workers...")
    runner = Runner(
        executor=FuturesExecutor(workers=args.workers),
        schema=NanoAODSchema,
        chunksize=100_000
    )

    # Run by calling the runner instance
    output = runner(
        fileset_for_coffea,
        treename="Events",
        processor_instance=FullLogicProcessor(config)
    )


    print("\nProcessing Complete.")
    print("\n=== Cutflow Summary ===")
    print(f"{'Dataset':<20} | {'Total':<10} | {'Passed':<10} | {'Eff (%)':<10}")
    print("-" * 60)

    for dataset, counts in output["cutflow"].items():
        total = counts["total"]
        passed = counts["passed"]
        eff = 100 * passed / total if total > 0 else 0
        print(f"{dataset:<20} | {total:<10} | {passed:<10} | {eff:<10.2f}")

    # --- Save Cutflow JSON ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cutflow_out = {
        ds: {k: int(v) for k, v in counts.items()}
        for ds, counts in output["cutflow"].items()
    }

    cutflow_path = outdir / "cutflow.json"
    with open(cutflow_path, "w") as f:
        json.dump(cutflow_out, f, indent=2, sort_keys=True)

    print(f"\nCutflow saved to: {cutflow_path}")

    # --- Save DQM (NPZ + Plots) ---
    dqm_dir = Path(args.outdir) / "dqm_summary"
    dqm_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Saving DQM and Generating Plots ===")

    for name, obj in output["dqm"].items():
        print(f"Processing {name}...")

        # save npz
        final_dqm_data = {
            "hists": obj.hists,
            "meta": obj.meta,
            "hist_defs": obj.hist_defs
        }
        np.savez(dqm_dir / f"dqm_{name}.npz", **final_dqm_data)

        # generate plots
        plot_dqm(final_dqm_data, name, dqm_dir)

    print(f"\nAll done! Check plots in: {dqm_dir}")
if __name__ == "__main__":
    main()