import argparse
import json
import numpy as np
import awkward as ak
import vector
from pathlib import Path
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import warnings

# --- Setup ---
warnings.filterwarnings("ignore", module="coffea")
vector.register_awkward()
plt.style.use(hep.style.CMS)

# ==========================================
# 1. Histogram Definitions
# ==========================================
axis_defs = {
    # -- Global Event Variables --
    "g_met": hist.axis.Regular(40, 0, 600, name="g_met", label="MET [GeV]"),
    "g_nJet": hist.axis.Regular(12, 0, 12, name="g_nJet", label="N Jets"),
    "g_nbJet": hist.axis.Regular(6, 0, 6, name="g_nbJet", label="N b-Jets"),
    "g_HT": hist.axis.Regular(40, 0, 2000, name="g_HT", label="HT [GeV]"),

    # -- Specific Kinematics --
    "lep_pt": hist.axis.Regular(40, 0, 500, name="lep_pt", label="Lepton pT [GeV]"),
    "bb_inv": hist.axis.Regular(40, 0, 1000, name="bb_inv", label="Mass(bb) [GeV]"),
    "bb_dR": hist.axis.Regular(30, 0, 5, name="bb_dR", label="dR(b,b)"),

    "lb_dR": hist.axis.Regular(30, 0, 5, name="lb_dR", label="dR(lep,b)"),
    "lb_m": hist.axis.Regular(40, 0, 500, name="lb_m", label="Mass(lep,b) [GeV]"),

    # -- Hypothesis Variables (XGB Inputs) --
    "A_Whad_m": hist.axis.Regular(40, 0, 300, name="A_Whad_m", label="Hypo A: W_had Mass [GeV]"),
    "B1_Whad_m": hist.axis.Regular(40, 0, 300, name="B1_Whad_m", label="Hypo B1: W_had Mass [GeV]"),
    "B2_Whad_m": hist.axis.Regular(40, 0, 300, name="B2_Whad_m", label="Hypo B2: W_had Mass [GeV]"),

    "A_bbWW_vis_m": hist.axis.Regular(40, 0, 2000, name="A_bbWW_vis_m", label="Hypo A: Visible Mass bbWW [GeV]"),
    "min_dR_l_q": hist.axis.Regular(30, 0, 5, name="min_dR_l_q", label="Min dR(lep, q)"),
    "top_lep_m": hist.axis.Regular(40, 0, 500, name="top_lep_m", label="Mass(lep, nu, b) [GeV]"),
    "top_nearest_m": hist.axis.Regular(40, 0, 500, name="top_nearest_m", label="Mass(lep, nu, nearest b) [GeV]"),
    "H_vis_m": hist.axis.Regular(40, 0, 1000, name="H_vis_m", label="Visible Mass (H) [GeV]"),
    "Whad_m": hist.axis.Regular(40, 0, 300, name="Whad_m", label="Mass(W_had) [GeV]"),
    "Whad_dR": hist.axis.Regular(30, 0, 5, name="Whad_dR", label="dR(q1, q2)"),
    "lbb_dR": hist.axis.Regular(30, 0, 5, name="lbb_dR", label="dR(lep, bb)"),
    "topness": hist.axis.Regular(100, 0, 100, name="topness", label="Topness (chi2)"),
    "l1_pt": hist.axis.Regular(40, 0, 500, name="l1_pt", label="Leading Lepton pT [GeV]"),
    "l1_eta": hist.axis.Regular(30, -3, 3, name="l1_eta", label="Leading Lepton eta"),
    "l1_iso": hist.axis.Regular(30, 0, 0.5, name="l1_iso", label="Leading Lepton Isolation"),
}



# ==========================================
# 2. Processor
# ==========================================
class QuickValidationProcessor(processor.ProcessorABC):
    def __init__(self, config):
        self.cfg = config

        # Initialize histograms
        self._hists = {}
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="Dataset", growth=True)

        for name, axis in axis_defs.items():
            self._hists[name] = hist.Hist(dataset_axis, axis, storage=hist.storage.Weight())

        # Cutflow histogram
        self._hists["cutflow"] = hist.Hist(
            dataset_axis,
            hist.axis.StrCategory([], name="cut", label="Selection Step", growth=True),
            storage=hist.storage.Weight()
        )

    @property
    def accumulator(self):
        return self._hists

    import awkward as ak
    import numpy as np

    def topness(self, leptons, jets, met):
        MW = 80.379
        MT = 172.5

        sigma_MW = 12.03
        sigma_MT_had = 21.42
        sigma_MT_lep = 29.49

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
        metpt_safe = ak.where(met.pt > 0, met.pt, 1e-6)

        mu = (MW ** 2) / 2.0 + l1_p4.pt * met.pt * np.cos(l1_p4.phi - met.phi)

        pt2 = l1_p4.pt ** 2 + 1e-12
        A = mu * l1_p4.pz / pt2
        B = (mu ** 2 * l1_p4.pz ** 2) / (pt2 ** 2) - ((l1_p4.energy ** 2 * met.pt ** 2 - mu ** 2) / pt2)

        sqrtB = np.sqrt(np.maximum(B, 0.0))
        cand1 = A - sqrtB
        cand2 = A + sqrtB
        pz_nu = ak.where(np.abs(cand1) < np.abs(cand2), cand1, cand2)
        pz_nu = ak.where(B < 0, A, pz_nu)

        nu_p4 = ak.zip(
            {
                "pt": met.pt,
                "eta": np.arcsinh(pz_nu / metpt_safe),
                "phi": met.phi,
                "mass": 0.0,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

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

        # Global min over all hypotheses; if no hypotheses -> None
        chi2_best = ak.min(ak.min(chi2, axis=2), axis=1)

        # Only keep "physically valid" events, otherwise return None
        return ak.where(valid, chi2_best, 99999)


    def process(self, events):
        dataset = events.metadata["dataset"]

        # --- 1. Object Prep ---
        ele = events.Electron
        ele["iso"] = ele.pfRelIso03_all
        mu = events.Muon
        mu["iso"] = mu.pfRelIso04_all

        good_ele = ele[(ele.pt > 35) & (abs(ele.eta) < 2.5) & (ele.mvaFall17V2Iso_WP90) &
                       (abs(ele.dxy) < 0.045) & (abs(ele.dz) < 0.2) & (ele.iso < 0.15)]
        good_mu = mu[(mu.pt > 25) & (abs(mu.eta) < 2.4) & (mu.iso < 0.15) & (mu.mediumId) &
                     (abs(mu.dxy) < 0.045) & (abs(mu.dz) < 0.2)]

        good_tau = events.Tau[
            (events.Tau.pt > 30) & (abs(events.Tau.eta) < 2.3) & (abs(events.Tau.dz) < 0.2) &
            ((events.Tau.decayMode < 5) | (events.Tau.decayMode >= 10)) &
            (events.Tau.idDeepTau2017v2p1VSjet >= 16) &
            (events.Tau.idDeepTau2017v2p1VSe >= 32) &
            (events.Tau.idDeepTau2017v2p1VSmu >= 1)
            ]

        good_jet = events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4) &
                              (events.Jet.jetId >= 4) & ~((events.Jet.pt < 50) & ~(events.Jet.puId == 7))]

        def dr_clean(obj, ref, dr=0.4):
            nearest = obj.nearest(ref)
            dR = obj.delta_r(nearest)
            return obj[ak.fill_none(dR, 999) > dr]

        good_ele = dr_clean(good_ele, good_mu)
        good_tau = dr_clean(good_tau, good_ele)
        good_tau = dr_clean(good_tau, good_mu)
        good_jet = dr_clean(good_jet, good_ele)
        good_jet = dr_clean(good_jet, good_mu)
        good_jet = dr_clean(good_jet, good_tau)

        good_jet = ak.drop_none(good_jet, axis=1)
        good_bjet = good_jet[good_jet.btagDeepFlavB > self.cfg['btag_wp']]
        good_light_jet = good_jet[good_jet.btagDeepFlavB < self.cfg['btag_wp']]



        def make_lep_p4(coll):
            return ak.zip({
                "pt": coll.pt, "eta": coll.eta, "phi": coll.phi, "mass": coll.mass, "charge": coll.charge, "iso": coll.iso,
            }, with_name="PtEtaPhiMLorentzVector")
        def make_jet_p4(coll):
            return ak.zip({
                "pt": coll.pt, "eta": coll.eta, "phi": coll.phi, "mass": coll.mass
            }, with_name="PtEtaPhiMLorentzVector")

        leptons = ak.concatenate([make_lep_p4(good_ele), make_lep_p4(good_mu)], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]


        leading_lepton = ak.firsts(leptons)
        # 1. Ensure jets have 4-vector properties
        jets_p4 = make_jet_p4(good_light_jet)

        # 2. Calculate Delta R between the leading lepton and ALL jets
        # This creates a jagged array of dR values matching the jet structure
        dr = jets_p4.delta_r(leading_lepton)

        # 3. Sort the jets by distance (closest first)
        sorted_jets = jets_p4[ak.argsort(dr, axis=1)]

        # 4. Select the second nearest (index 1)
        # We use pad_none to safely handle events that might have fewer than 2 jets
        # (If an event has 0 or 1 jet, this returns None instead of crashing)
        second_nearest_jet = ak.pad_none(sorted_jets, 2, axis=1)[:, 1]
        nearest_jet = ak.firsts(sorted_jets)
        dr_lj = ak.fill_none(leading_lepton.delta_r(nearest_jet), 999)

        H_vis_m = (leading_lepton + nearest_jet + second_nearest_jet).mass


        # --- 2. Event Selection & Cutflow ---
        selection = PackedSelection()

        leading_two_bjets = good_jet[ak.argsort(good_jet.btagDeepFlavB, axis=1, ascending=False)][:, :2]
        leading_two_bjets = ak.pad_none(leading_two_bjets, 2)
        leading_two_bjets = ak.fill_none(leading_two_bjets, 0)
        m_bb = ak.fill_none((leading_two_bjets[:, 0] + leading_two_bjets[:, 1]).mass, 0)
        dr_bb = ak.fill_none(leading_two_bjets[:, 0].delta_r(leading_two_bjets[:, 1]), 0)
        topness = self.topness(leptons, good_jet, events.MET)



        # FIX 1: Use ak.ones_like instead of ak.num for the Initial step
        cuts_to_apply = [
            ("Initial", ak.ones_like(events.run, dtype=bool)),
            ("One Lepton", ak.num(leptons) == 1),
            ("Tau Veto", ak.num(good_tau) == 0),
            (">=2 Light Jets", ak.num(good_light_jet) >= 2),
            (">=2 b-Jets", ak.num(good_bjet) >= 2),
            ("dr_lj > 0.4", dr_lj < 1.6),
            ("topness > 3", topness > 3),
            # ("m_bb < 150 GeV", m_bb < 150),
            # ("dR_bb < 2.6", dr_bb < 2.6),
        ]

        current_mask = ak.ones_like(events.run, dtype=bool)
        weights = ak.to_numpy(events.genWeight / abs(events.genWeight))

        for name, cut_mask in cuts_to_apply:
            current_mask = current_mask & cut_mask
            selection.add(name, cut_mask)

            # FIX 2: Explicitly broadcast string scalars to match array length
            w_subset = weights[current_mask]
            N = len(w_subset)
            if N > 0:
                self._hists["cutflow"].fill(
                    dataset=np.full(N, dataset),  # Broadcast dataset name
                    cut=np.full(N, name),  # Broadcast cut name
                    weight=w_subset
                )

        final_cut = selection.all(*[c[0] for c in cuts_to_apply[1:]])
        sel_ev = events[final_cut]
        sel_w = weights[final_cut]

        if len(sel_ev) == 0:
            return self._hists

        # Slice objects
        sel_leps = leptons[final_cut]
        sel_jets = good_jet[final_cut]
        sel_met = events.MET[final_cut]

        # --- 3. Feature Calculation ---
        data = self.get_physics_variables(sel_leps, sel_jets, sel_met)

        data["g_met"] = sel_met.pt
        data["g_nJet"] = ak.num(sel_jets)
        data["g_nbJet"] = ak.num(sel_jets[sel_jets.btagDeepFlavB > self.cfg['btag_wp']])
        data["g_HT"] = ak.sum(sel_jets.pt, axis=1)
        data["lep_pt"] = sel_leps[:, 0].pt

        # --- 4. Fill Histograms ---
        N_final = len(sel_w)
        for name, arr in data.items():
            if name in self._hists:
                val = ak.to_numpy(ak.fill_none(arr, -999)).flatten()

                # Double check length alignment (paranoid check)
                if len(val) == N_final:
                    self._hists[name].fill(
                        dataset=np.full(N_final, dataset),
                        **{name: val, "weight": sel_w}
                    )

        return self._hists

    def postprocess(self, accumulator):
        return accumulator

    # ==========================================
    # Logic: Variables
    # ==========================================
    def get_physics_variables(self, leptons, jets, met):
        def get_p4(obj):
            return ak.zip(
                {"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass},
                with_name="PtEtaPhiMLorentzVector"
            )

        l1 = leptons[:, 0]
        l1_p4 = get_p4(l1)

        jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        is_btag = jets_sorted.btagDeepFlavB > self.cfg["btag_wp"]

        bjets = jets_sorted[is_btag]

        nearest_bjet = l1_p4.nearest(bjets)
        dr_lb = l1_p4.delta_r(nearest_bjet)

        MW_PDG = 80.4
        MH_PDG = 125.0


        # ==============================================================================
        # 1. Object Preparation
        # ==============================================================================
        l1 = ak.firsts(leptons)
        l1_p4 = get_p4(l1)
        met_p4 = ak.zip(
            {"pt": met.pt, "eta": ak.zeros_like(met.pt), "phi": met.phi, "mass": 0.0},
            with_name="PtEtaPhiMLorentzVector",
        )

        leading_lepton = ak.firsts(leptons)
        # 1. Ensure jets have 4-vector properties
        light_jets_p4 = get_p4(jets[~is_btag])

        # 2. Calculate Delta R between the leading lepton and ALL jets
        # This creates a jagged array of dR values matching the jet structure
        dr = light_jets_p4.delta_r(leading_lepton)

        # 3. Sort the jets by distance (closest first)
        sorted_light_jets = light_jets_p4[ak.argsort(dr, axis=1)]

        # 4. Select the second nearest (index 1)
        # We use pad_none to safely handle events that might have fewer than 2 jets
        # (If an event has 0 or 1 jet, this returns None instead of crashing)
        second_nearest_jet = ak.pad_none(sorted_light_jets, 2, axis=1)[:, 1]
        nearest_jet = ak.firsts(sorted_light_jets)
        H_vis_m = (leading_lepton + nearest_jet + second_nearest_jet).mass
        Hhad_m = (nearest_jet + second_nearest_jet).mass





        # --- Neutrino Reconstruction (Needed for Hypo B2 selection) ---
        # IMPORTANT BUGFIX: use l1_p4 (not l1.pz / l1.energy) for schema robustness
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

        w_lep = l1_p4 + nu_p4
        # find b that forms with w_lep mass closest to Mtop_PDG
        Mtop_PDG = 172.5
        b_candidates = bjets

        # 1. Calculate Mass
        # Note: Ensure w_lep matches the jagged structure of b_candidates if not already broadcasted
        wlep_b_masses = (w_lep[:, None] + get_p4(b_candidates)).mass

        # 2. Get Index (Jagged)
        # keepdims=True ensures the output is a jagged array like [[0], [1], [], [0]]
        # instead of a flat array with Nones [0, 1, None, 0].
        # This preserves the structure needed to slice 'b_candidates'.
        idx_closest = ak.argmin(abs(wlep_b_masses - Mtop_PDG), axis=1, keepdims=True)

        # 3. Select the Candidate
        # b_candidates[idx_closest] keeps the jagged structure (1 jet per event, or 0 if empty)
        # ak.firsts(...) converts the jagged array "[[Jet], [Jet], []]" into an Option Array "[Jet, Jet, None]"
        wlep_b = ak.firsts(b_candidates[idx_closest])


        # ==============================================================================
        chi2_best = self.topness(leptons, jets, met)


        bjets_by_score = bjets[ak.argsort(bjets.btagDeepFlavB, axis=1, ascending=False)]
        b_cands = ak.pad_none(bjets_by_score, 2)
        b1, b2 = b_cands[:, 0], b_cands[:, 1]
        b1_p4, b2_p4 = get_p4(b1), get_p4(b2)
        bb_sys = b1_p4 + b2_p4



        l_cands = jets_sorted[~is_btag]
        l_cands = l_cands[:, :5]
        qq_pairs = ak.combinations(l_cands, 2, axis=1)
        q1, q2 = ak.unzip(qq_pairs)
        q1_p4, q2_p4 = get_p4(q1), get_p4(q2)
        whad_cands = q1_p4 + q2_p4

        MW = 80.4
        idx_A = ak.argmin(abs(whad_cands.mass - MW), axis=1, keepdims=True)
        idx_B1 = ak.argmin(q1_p4.delta_r(q2_p4), axis=1, keepdims=True)

        def get_hypo(idx):
            w = ak.firsts(whad_cands[idx])
            q1_sel = ak.firsts(q1_p4[idx])
            q2_sel = ak.firsts(q2_p4[idx])
            return w, q1_sel, q2_sel

        w_A, q1_A, q2_A = get_hypo(idx_A)
        w_B1, _, _ = get_hypo(idx_B1)
        w_B2, _, _ = get_hypo(idx_B1)

        res = {}
        res["l1_iso"] = ak.firsts(leptons.iso)
        res["l1_eta"] = l1.eta
        res["l1_pt"] = l1.pt
        res["bb_inv"] = bb_sys.mass
        res["bb_dR"] = b1_p4.delta_r(b2_p4)
        res["lb_dR"] = dr_lb
        res["lbb_dR"] = (l1_p4.delta_r(bb_sys))
        res["lb_m"] = (l1_p4 + get_p4(nearest_bjet)).mass
        res["top_lep_m"] = (w_lep + get_p4(wlep_b)).mass
        res["top_nearest_m"] = (w_lep + get_p4(nearest_bjet)).mass

        res["A_Whad_m"] = w_A.mass
        res["B1_Whad_m"] = w_B1.mass
        res["B2_Whad_m"] = w_B2.mass
        res["H_vis_m"] = H_vis_m
        res["Whad_m"] = Hhad_m
        res["Whad_dR"] = nearest_jet.delta_r(second_nearest_jet)
        ww_vis_A = l1_p4 + w_A
        bbww_vis_A = bb_sys + ww_vis_A
        res["A_bbWW_vis_m"] = bbww_vis_A.mass
        res["min_dR_l_q"] = np.minimum(l1_p4.delta_r(q1_A), l1_p4.delta_r(q2_A))
        res["topness"] = chi2_best

        return res


# ==========================================
# 3. Plotting Functions
# ==========================================
def plot_cutflow(h_cutflow, outdir):
    fig, ax = plt.subplots(figsize=(10, 8))

    datasets = [x for x in h_cutflow.axes["dataset"]]
    cuts = [x for x in h_cutflow.axes["cut"]]

    for ds in datasets:
        vals = h_cutflow[{"dataset": ds}].values()
        if vals[0] > 0:
            vals_norm = vals / vals[0]
            ax.plot(cuts, vals_norm, marker='o', label=ds, linewidth=2)
            print(f"[{ds}] Final Eff: {vals_norm[-1] * 100:.2f}% ({int(vals[-1])} events)")

    ax.set_ylabel("Efficiency (normalized to Initial)")
    ax.set_title("Cutflow Selection Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    hep.cms.label(ax=ax, label="Internal", rlabel="Run2")

    fig.savefig(outdir / "cutflow_comparison.png")
    plt.close(fig)


def plot_variables(hists, outdir, signal_names):
    for var_name, h in hists.items():
        if var_name == "cutflow": continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine which datasets are in the histogram
        present_datasets = [d for d in h.axes["dataset"]]

        # Separate into Signal vs Background based on JSON list
        bkg_list = [d for d in present_datasets if d not in signal_names]
        sig_list = [d for d in present_datasets if d in signal_names]

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Plot Backgrounds (Filled)
        for i, bkg in enumerate(bkg_list):
            h_bkg = h[{"dataset": bkg}]
            if h_bkg.sum() == 0: continue

            hep.histplot(
                h_bkg,
                ax=ax,
                density=True,
                label=f"{bkg} (Bkg)",
                histtype="fill",
                alpha=0.3,
                edgecolor=color_cycle[i % len(color_cycle)],
                linewidth=1
            )

        # Plot Signals (Lines)
        for i, sig in enumerate(sig_list):
            h_sig = h[{"dataset": sig}]
            if h_sig.sum() == 0: continue

            hep.histplot(
                h_sig,
                ax=ax,
                density=True,
                label=f"{sig} (Sig)",
                histtype="step",
                linewidth=2.5,
                linestyle="--"
            )

        ax.set_ylabel("Density (a.u.)")
        ax.set_title(f"Shape Comparison: {var_name}", fontsize=16)
        ax.legend(title="Normalized to 1")
        hep.cms.label(ax=ax, label="Internal", rlabel="Run2")

        fig.savefig(outdir / f"dist_{var_name}.png")
        print(f"Saved plot: dist_{var_name}.png")
        plt.close(fig)


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Input JSON {signal: {...}, background: {...}}")
    parser.add_argument("--outdir", default="quick_plots")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # 1. Parse JSON strictly
    with open(args.json) as f:
        fileset_raw = json.load(f)

    fileset = {}
    signal_names = []

    # Process Signals
    if "signal" in fileset_raw:
        for name, files in fileset_raw["signal"].items():
            fileset[name] = {"files": files, "metadata": {}}
            signal_names.append(name)

    # Process Backgrounds
    if "background" in fileset_raw:
        for name, files in fileset_raw["background"].items():
            fileset[name] = {"files": files, "metadata": {}}

    print(f"Identified Signals: {signal_names}")
    print(f"Total Datasets: {len(fileset)}")

    config = {
        "btag_wp": 0.2489,
    }

    print(f"Running quick validation on {args.limit} chunks per dataset...")

    runner = processor.Runner(
        executor=processor.FuturesExecutor(workers=args.workers),
        schema=NanoAODSchema,
        chunksize=50_000,
        maxchunks=args.limit
    )

    results = runner(
        fileset,
        treename="Events",
        processor_instance=QuickValidationProcessor(config)
    )

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Plots...")
    plot_cutflow(results["cutflow"], out_path)
    plot_variables(results, out_path, signal_names)

    print(f"\nDone! Check folder: {args.outdir}")


if __name__ == "__main__":
    main()