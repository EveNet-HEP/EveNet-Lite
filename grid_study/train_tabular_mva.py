import sys
import os
import argparse
import re
import json
import yaml
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
import pickle
from tabpfn.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from shared_metrics import plot_score_overlay

from config_loader import ConfigLoader, DatasetInfo
# --- Optional Imports ---
try:
    from tabpfn import TabPFNClassifier

    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

import matplotlib.pyplot as plt
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except ImportError:
    print("mplhep not found, using default style")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Physics Metric Fallback ---
try:
    from evenet_lite.metrics import calculate_physics_metrics
except ImportError as e:
    logger.warning("evenet_lite not found, using simplified physics metrics.")
    print("error", e)
    def calculate_physics_metrics(probs, targets, weights):
        return {
            'max_sic_unc': 0.0, 'max_sic': 0.0,
            'auc': roc_auc_score(targets, probs, sample_weight=weights)
        }



# ==========================================
# 2. Data Management
# ==========================================

class DatasetManager:
    def __init__(self, config_loader: ConfigLoader, parameterize: bool = False, features: List[str] = None):
        self.cfg = config_loader
        self.parameterize = parameterize
        self.features = features
        self.feature_indices = None
        self.feature_names_loaded = None

    def load_data(self, datasets: List[DatasetInfo], split: str = "train",
                  target_masses: Optional[np.ndarray] = None, lumi:float = 1.0, max_entries=None) -> Dict[str, np.ndarray]:
        """
        Loads .npz files for the given list of datasets and split (train/valid).
        Handles:
          - Weights Calculation (xsec/nEvent)
          - Mass Parameterization (Random injection for Bkg)
          - Feature Selection
        """
        X_list, y_list, w_list, m_list, p_list = [], [], [], [], []

        for ds in datasets:
            search_path = ds.path / "xgb" / split
            files = list(search_path.glob("*.npz"))

            if not files:
                continue

            max_events = getattr(ds, "max_events", None)
            seen = 0  # events kept so far for this dataset
            total_number = 0
            norm_factor = 1.0
            if max_events is not None:
                for fp in files:
                    with np.load(fp, allow_pickle=True) as data:
                        if 'X' not in data: continue
                        arr = data['X']
                        N = len(arr)
                        total_number += N
                if max_events < total_number:
                    norm_factor = total_number / max_events
                print(f"only use {max_events} out of {total_number} events from {ds.name}")

            for fp in files:
                if (max_events is not None and seen >= max_events):
                    break
                try:
                    with np.load(fp, allow_pickle=True) as data:
                        if 'X' not in data: continue
                        arr = data['X']
                        if len(arr) == 0: continue

                        # --- Feature Management ---
                        # Initialize feature mapping on first successful load
                        if self.feature_names_loaded is None and 'features' in data:
                            self.feature_names_loaded = list(data['features'])
                            if self.features:
                                self.feature_indices = [self.feature_names_loaded.index(f) for f in self.features if
                                                        f in self.feature_names_loaded]
                            else:
                                self.feature_indices = list(range(len(arr[0])))

                        # Select Columns
                        if self.feature_indices:
                            arr = arr[:, self.feature_indices]

                        # --- Weights ---
                        # Weight = (Sign of genWeight) * (xsec / total_nevents)
                        raw_w = data['weights'] if 'weights' in data else np.ones(len(arr))
                        phys_w = raw_w * (ds.xsec * lumi / ds.nevents) * 2 * norm_factor # Factor 2 for train/valid split
                        # if split == "train":
                        #     phys_w = abs(phys_w)  # Use absolute weights for training
                        #
                        # --- Mass Injection ---
                        N = len(arr)
                        if ds.is_signal:
                            mass_arr = np.column_stack([np.full(N, ds.mx), np.full(N, ds.my)])
                        else:
                            # For Background Training: Inject random mass hypotheses
                            if self.parameterize and split == "train":
                                if target_masses is None:
                                    raise ValueError("Target masses required for Background parameterization")
                                rand_idx = np.random.randint(0, len(target_masses), size=N)
                                mass_arr = target_masses[rand_idx]
                            else:
                                mass_arr = np.zeros((N, 2))

                        X_list.append(arr)
                        y_list.append(np.ones(N) if ds.is_signal else np.zeros(N))
                        w_list.append(phys_w)
                        m_list.append(mass_arr)
                        p_list.append([ds.category] * N)

                        seen += N

                except Exception as e:
                    logger.warning(f"Corrupt file {fp}: {e}")

        if not X_list:
            logger.error(f"No data loaded for split {split}!")
            return {}

        out = {
            "X":    np.concatenate(X_list, axis=0),
            "y":    np.concatenate(y_list, axis=0),
            "w":    np.concatenate(w_list, axis=0),
            "m":    np.concatenate(m_list, axis=0),
            # keep proc as object to avoid weird unicode truncation surprises
            "proc": np.concatenate([np.asarray(p, dtype=object) for p in p_list], axis=0),
        }

        if max_entries is not None:
            N = out["X"].shape[0]
            n = min(int(max_entries), N)

            # random subset, no replacement
            idx = np.random.choice(N, size=n, replace=False)

            out["X"] = out["X"][idx]
            out["y"] = out["y"][idx]
            out["w"] = out["w"][idx]
            out["m"] = out["m"][idx]
            out["proc"] = out["proc"][idx]

        return out

    def reweight_signals(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Renormalize signal weights so each mass point contributes equally."""
        if 'm' not in data: return data

        w = data['w']
        m = data['m']
        unique_masses = np.unique(m, axis=0)
        if len(unique_masses) == 0: return data

        target_w = np.sum(w) / len(unique_masses)

        logger.info(f"Reweighting {len(unique_masses)} signal points to target weight {target_w:.2e}")

        for mx, my in unique_masses:
            mask = (m[:, 0] == mx) & (m[:, 1] == my)
            current_sum = np.sum(w[mask])
            if current_sum > 0:
                w[mask] *= (target_w / current_sum)

        data['w'] = w
        return data

    def downsample_for_tabpfn(self, X, y, w, limit=20000):
        """Probabilistic downsampling based on weights."""
        if len(X) <= limit: return X, y, w
        logger.info(f"TabPFN Downsampling: {len(X)} -> {limit}")

        prob = np.abs(w) / np.sum(np.abs(w))
        idx = np.random.choice(len(X), limit, replace=False, p=prob)
        return X[idx], y[idx], w[idx]

# ==========================================
# 3. Plotting Helpers
# ==========================================

def plot_overtraining(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, out_dir):
    """Checks score distribution for Train vs Validation to detect overtraining."""
    print(">>> Plotting Overtraining Check...")

    # Handle TabPFN or large datasets to prevent OOM during prediction
    if hasattr(model, "predict_proba"):
        # Downsample for plotting if dataset is too large (>20k)
        if len(X_tr) > 200000:
            idx = np.random.choice(len(X_tr), 200000, replace=False)
            X_tr, y_tr, w_tr = X_tr[idx], y_tr[idx], w_tr[idx]

        # XGBoost handles this fast, TabPFN needs small batches if not downsampled
        # --- FIX: Batch prediction for TabPFN to avoid CUDA errors ---
        # Check if the dataset is large (e.g. > 2000 samples)
        if len(X_tr) >50000:
            batch_size = 50000
            preds = []
            for i in range(0, len(X_tr), batch_size):
                # Predict in chunks
                preds.append(model.predict_proba(X_tr[i:i + batch_size])[:, 1])
            tr_scores = np.concatenate(preds)
        else:
            # Small dataset, run normally
            tr_scores = model.predict_proba(X_tr)[:, 1]

        # Repeat the same for Validation set if it is also large
        if len(X_val) > 50000:
            batch_size = 50000
            preds_val = []
            for i in range(0, len(X_val), batch_size):
                preds_val.append(model.predict_proba(X_val[i:i + batch_size])[:, 1])
            val_scores = np.concatenate(preds_val)
        else:
            val_scores = model.predict_proba(X_val)[:, 1]

    else:
        return

    plt.figure(figsize=(10, 8))
    bins = np.linspace(0, 1, 40)

    # Train (Filled Histogram)
    plt.hist(tr_scores[y_tr == 0], bins=bins, weights=w_tr[y_tr == 0], density=True,
             alpha=0.3, color='blue', label='Train Bkg')
    plt.hist(tr_scores[y_tr == 1], bins=bins, weights=w_tr[y_tr == 1], density=True,
             alpha=0.3, color='red', label='Train Sig')

    # Valid (Dots / Error bars)
    h_b, _ = np.histogram(val_scores[y_val == 0], bins=bins, weights=w_val[y_val == 0], density=True)
    h_s, _ = np.histogram(val_scores[y_val == 1], bins=bins, weights=w_val[y_val == 1], density=True)
    ct = (bins[:-1] + bins[1:]) / 2

    plt.plot(ct, h_b, 'o', color='blue', label='Valid Bkg')
    plt.plot(ct, h_s, 'o', color='red', label='Valid Sig')

    plt.xlabel("Model Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Overtraining Check")
    plt.savefig(out_dir / "overtraining.png")
    plt.close()





# ==========================================
# 3. Execution Flow
# ==========================================

def run_pipeline(args):
    # 1. Setup
    # --- output dir naming rule (match EveNet demo) ---
    if args.parameterize:
        mode_str = f"parametrized_reduce_factor_x_{args.param_mx_step}_y_{args.param_my_step}"
    else:
        mode_str = "individual"

    mass_target = "All" if args.parameterize else f"MX-{args.mX}_MY-{args.mY}"
    model_str = args.model  # "xgb" or "tabpfn"

    out_dir = Path(args.out_dir) / model_str / mode_str / mass_target
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Config & Discovery
    cfg = ConfigLoader(args.yaml_path, args.base_dir)
    all_datasets = cfg.discover_datasets()

    # Filter Signals based on args
    # Filter Signals based on args (train vs eval for sparse parametrization)
    if args.parameterize:
        sig_all = [d for d in all_datasets if d.is_signal]

        mx_vals = sorted({d.mx for d in sig_all})
        my_vals = sorted({d.my for d in sig_all})

        mx_keep = set(mx_vals[::max(1, args.param_mx_step)])
        my_keep = set(my_vals[::max(1, args.param_my_step)])

        sig_datasets_eval = sig_all
        sig_datasets_train = [d for d in sig_all if (d.mx in mx_keep and d.my in my_keep)]

        if not sig_datasets_train:
            logger.error("No signal datasets selected for training after applying sparse grid steps!")
            sys.exit(1)

        logger.info(
            f"Sparse parametrization: train on {len(sig_datasets_train)} / eval on {len(sig_datasets_eval)} signal points")
    else:
        sig_datasets_train = [d for d in all_datasets if d.is_signal and d.mx == args.mX and d.my == args.mY]
        sig_datasets_eval = sig_datasets_train
        if not sig_datasets_train:
            logger.error(f"Signal MX={args.mX}, MY={args.mY} not found!")
            sys.exit(1)

    bkg_datasets = [d for d in all_datasets if not d.is_signal]

    # Collect all available mass points for parametrization logic
    target_masses = np.array([[d.mx, d.my] for d in sig_datasets_train])

    # 3. Load Training Data
    dm = DatasetManager(cfg, parameterize=args.parameterize, features=args.features)
    model = None
    if "train" in args.stage:
        logger.info(">>> Loading Signal (Train)...")
        d_sig_tr = dm.load_data(sig_datasets_train, "train", lumi=args.lumi)
        d_sig_tr = dm.reweight_signals(d_sig_tr)



        logger.info(">>> Loading Background (Train)...")
        d_bkg_tr = dm.load_data(
            bkg_datasets, "train",
            target_masses=target_masses,
            lumi=args.lumi,
            max_entries=args.max_bkg_entries
        )
        # Global Balance: Sum(Bkg Weights) = Sum(Sig Weights)
        # ---- global balance: scale background to match total signal weight ----
        sig_sum = d_sig_tr["w"].sum()
        bkg_sum = d_bkg_tr["w"].sum()

        if bkg_sum > 0:
            num_bkg = d_bkg_tr["w"].shape[0]
            d_bkg_tr["w"] = d_bkg_tr["w"] * (num_bkg / bkg_sum)
            d_sig_tr["w"] = d_sig_tr["w"] * (num_bkg / sig_sum)
        #
        # scale = np.sum(d_sig_tr['w']) / np.sum(d_bkg_tr['w'])
        # d_bkg_tr['w'] *= scale

        # Merge
        X_full = np.concatenate([d_bkg_tr['X'], d_sig_tr['X']])
        y_full = np.concatenate([d_bkg_tr['y'], d_sig_tr['y']])
        w_full = np.concatenate([d_bkg_tr['w'], d_sig_tr['w']])

        w_full = abs(w_full)

        if args.parameterize:
            m_full = np.concatenate([d_bkg_tr['m'], d_sig_tr['m']])
            X_full = np.hstack([X_full, m_full])

        # 4. Training
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X_full, y_full, w_full, test_size=0.2, stratify=y_full, random_state=42
        )

        start_time = time.time()

        # positive_weight_mask = w_tr > 0
        #
        # X_tr = X_tr[positive_weight_mask]
        # y_tr = y_tr[positive_weight_mask]
        # w_tr = w_tr[positive_weight_mask]

        if args.model == 'xgb':
            use_gpu = os.environ.get('CUDA_VISIBLE_DEVICES') is not None
            logger.info("Training XGBoost...")
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=4000,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=5,
                subsample=0.7,
                colsample_bytree=0.7,
                gamma=0.0,
                reg_lambda=2.0,
                reg_alpha=0.0,
                tree_method="gpu_hist" if use_gpu else "hist",
                random_state=42,
                early_stopping_rounds = 200
            )
            model.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val],
                verbose=100
            )
            model.save_model(out_dir / "model.json")
            # -----------------------------
            # Feature importance plots
            # -----------------------------
            try:
                booster = model.get_booster()

                # Build feature names (best effort)
                base_names = dm.feature_names_loaded
                if base_names is None:
                    # fallback: f0,f1,...
                    n_base = X_tr.shape[1] - (2 if args.parameterize else 0)
                    base_names = [f"f{i}" for i in range(n_base)]
                else:
                    # apply selection if user passed --features or --features_yaml
                    if dm.feature_indices is not None:
                        base_names = [base_names[i] for i in dm.feature_indices]

                feat_names = list(base_names)
                if args.parameterize:
                    feat_names += ["MX", "MY"]

                booster.feature_names = feat_names

                for imp_type in ["gain", "weight"]:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    xgb.plot_importance(
                        booster,
                        importance_type=imp_type,
                        max_num_features=30,
                        show_values=False,
                        ax=ax
                    )
                    ax.set_title(f"XGBoost Feature Importance ({imp_type})")
                    fig.tight_layout()
                    fig.savefig(out_dir / f"feature_importance_{imp_type}.png")
                    plt.close(fig)

                logger.info("Saved feature importance plots to out_dir.")
            except Exception as e:
                logger.warning(f"Failed to plot feature importance: {e}")

        elif args.model == 'tabpfn':
            if not HAS_TABPFN:
                logger.error("TabPFN requested but not installed.")
                sys.exit(1)
            logger.info("Training TabPFN...")
            # downsampling background based on weights
            X_sig, y_sig, w_sig = X_tr[y_tr == 1], y_tr[y_tr == 1], w_tr[y_tr == 1]
            X_bkg, y_bkg, w_bkg = X_tr[y_tr == 0], y_tr[y_tr == 0], w_tr[y_tr == 0]
            if len(X_sig) > args.tabpfn_limit // 2:
                X_sig, y_sig, w_sig = dm.downsample_for_tabpfn(X_sig, y_sig, w_sig, limit=args.tabpfn_limit // 2)
            remaining_number = args.tabpfn_limit - len(X_sig)
            X_bkg_sub, y_bkg_sub, w_bkg_sub = dm.downsample_for_tabpfn(X_bkg, y_bkg, w_bkg, limit=remaining_number)
            X_sub = np.concatenate([X_sig, X_bkg_sub])
            y_sub =  np.concatenate([y_sig, y_bkg_sub])
            model = TabPFNClassifier(balance_probabilities=True) #'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
            print("cuda:", os.environ.get('CUDA_VISIBLE_DEVICES'))
            model.fit(X_sub, y_sub)
            # Save via Pickle ---
            model_path = out_dir / "model.tabpfn_fit"
            save_fitted_tabpfn_model(model, model_path)
            logger.info(f"TabPFN model saved to {out_dir / 'model.pkl'}")

        finish_time = time.time()
        fitting_time = finish_time - start_time

        # =========================================================
        # [INSERT 1] Plot Overtraining Check (Right after training)
        # =========================================================
        # plot_overtraining(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, out_dir)

    if "predict" in args.stage:
        # 5. Inference (Evaluation)
        if model is None:
            logger.info(">>> Loading Trained Model...")
            if args.model == 'xgb':
                model = xgb.XGBClassifier()
                model.load_model(out_dir / "model.json")
            elif args.model == 'tabpfn':
                if not HAS_TABPFN:
                    logger.error("TabPFN requested but not installed.")
                    sys.exit(1)
                model_path = out_dir / "model.tabpfn_fit"
                if not model_path.exists():
                    logger.error(f"TabPFN model not found at {model_path}")
                    sys.exit(1)

                device = "cpu" if os.environ.get('CUDA_VISIBLE_DEVICES') is None else "cuda"
                logger.info(f">>> Loading TabPFN model on {device}...")
                model = load_fitted_tabpfn_model(model_path, device=device)

        logger.info(">>> Loading Test Data...")
        d_sig_te = dm.load_data(sig_datasets_eval, "valid", lumi=args.lumi)
        d_bkg_te = dm.load_data(bkg_datasets, "valid", lumi=args.lumi)
        # d_sig_te = dm.reweight_signals(d_sig_te)

        # Prepare for parametrized inference loop
        unique_masses = np.unique(d_sig_te['m'], axis=0)

        for mx, my in unique_masses:
            # A. Get Signal Subset
            if args.mX is not None and (int(mx) != int(args.mX)):
                continue
            if args.mY is not None and (int(my) != int(args.mY)):
                continue
            mask_s = (d_sig_te['m'][:, 0] == mx) & (d_sig_te['m'][:, 1] == my)
            X_s = d_sig_te['X'][mask_s]
            # B. Get Background (Parameterize Injection)
            X_b = d_bkg_te['X'].copy()
            if args.parameterize:
                # Overwrite Bkg mass to current signal hypothesis
                m_b_inj = np.column_stack([np.full(len(X_b), mx), np.full(len(X_b), my)])
                m_s_act = np.column_stack([np.full(len(X_s), mx), np.full(len(X_s), my)])
                X_eval = np.concatenate([
                    np.hstack([X_b, m_b_inj]),
                    np.hstack([X_s, m_s_act])
                ])
            else:
                X_eval = np.concatenate([X_b, X_s])

            y_eval = np.concatenate([d_bkg_te['y'], d_sig_te['y'][mask_s]])
            w_eval = np.concatenate([d_bkg_te['w'], d_sig_te['w'][mask_s]])
            p_eval = np.concatenate([d_bkg_te['proc'], d_sig_te['proc'][mask_s]])

            # C. Predict
            if args.model == 'tabpfn' and len(X_eval) > 50000:
                # Batch prediction
                batch = 50000
                preds = []
                # Use tqdm for progress bar
                import tqdm
                logger.info(f"TabPFN large eval set detected ({len(X_eval)} samples). Using batch prediction...")
                for i in tqdm.tqdm(range(0, len(X_eval), batch)):
                    preds.append(model.predict_proba(X_eval[i:i + batch])[:, 1])
                y_pred = np.concatenate(preds)
            else:
                y_pred = model.predict_proba(X_eval)[:, 1]

            # Construct the filename
            filename = f"predictions_MX-{int(round(mx.item()))}_MY-{int(round(my.item()))}.npz"
            output_path = out_dir / filename

            # Save directly using keyword arguments.
            # No need for .tolist() or .item() here; numpy handles its own types best.
            np.savez_compressed(
                output_path,
                y_true=y_eval,
                y_pred=y_pred,
                w=w_eval,
                proc=p_eval,
                mx=mx,
                my=my
            )


    if "evaluate" in args.stage:
        logger.info(">>> Evaluating Predictions...")
        # Load predictions
        all_masses = [(d_sig.mx, d_sig.my) for d_sig in sig_datasets_eval]
        for mx, my in all_masses:
            if args.mX is not None and (int(mx) != int(args.mX)):
                continue
            if args.mY is not None and (int(my) != int(args.mY)):
                continue


            pred_file = out_dir / f"predictions_MX-{int(round(mx))}_MY-{int(round(my))}.npz"
            if not pred_file.exists():
                logger.warning(f"Prediction file not found: {pred_file}")
                continue

            with np.load(pred_file, allow_pickle=True) as data:

                # No need for np.array() casting; they are already loaded as ndarrays
                y_eval = data["y_true"]
                y_pred = data["y_pred"]
                w_eval = data["w"]
                p_eval = data["proc"]

            nevents_by_name = {ds.category: ds.nevents if ds.category != 'signal' else 1.0 for ds in all_datasets }
            nevents_eval = np.array([nevents_by_name[p] for p in p_eval])
            # w_eval = w_eval / nevents_eval

            # D. Metrics
            metrics = calculate_physics_metrics(
                y_pred, y_eval, w_eval, training=False,
                min_bkg_events=10,
                log_plots=True,
                bins=1000,
                # min_bkg_ratio=0.0001,
                f_name=f"{out_dir}/sic_MX-{int(mx)}_MY-{int(my)}.png",
                Zs=10,
                Zb=5,
                min_bkg_per_bin=3,
                min_mc_stats=0.2,
                include_signal_in_stat=False,
            )

            key = f"MX-{int(mx)}_MY-{int(my)}"
            results = {
                "auc": float(metrics['auc']),
                "max_sic": float(metrics['max_sic']),
                "max_sic_unc": float(metrics['max_sic_unc']),
                "trafo_bin_sig": float(metrics["trafo_bin_sig"]),
                "sic": metrics["sic"].tolist(),
                "sic_unc": metrics["sic_unc"].tolist(),
                "trafo_edge": metrics["trafo_edge"].tolist(),
                # "fitting_time": fitting_time
            }
            logger.info(
                f"Mass {key}: AUC={metrics['auc']:.4f}, Max SIC={metrics['max_sic']:.4f}, Bin SIG={metrics['trafo_bin_sig']:.4f}")


            plot_score_overlay(
                y_eval=y_eval,
                w_eval=w_eval,
                p_eval=p_eval,
                y_pred=y_pred,
                fname = out_dir / f"score_uniform_binning_MX-{int(mx)}_MY-{int(my)}.png"
            )
            plot_score_overlay(
                y_eval=y_eval,
                y_pred=y_pred,
                w_eval=w_eval,
                p_eval=p_eval,
                bins=metrics['trafo_edge'],
                uniform_bin_plot=True,
                fname = out_dir / f"score_auto_binning_flat_MX-{int(mx)}_MY-{int(my)}.png"
            )
            plot_score_overlay(
                y_eval=y_eval,
                y_pred=y_pred,
                w_eval=w_eval,
                p_eval=p_eval,
                bins=metrics['trafo_edge'],
                uniform_bin_plot=False,
                fname = out_dir / f"score_auto_binning_MX-{int(mx)}_MY-{int(my)}.png"
            )

            with open(out_dir / f"eval_metrics_MX-{int(mx)}_MY-{int(my)}.json", "w") as f:
                json.dump(results, f, indent=4)
    logger.info(f"Done. Results saved to {out_dir}")


# ==========================================
# 4. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Grid Search Trainer")

    # Data Selection
    parser.add_argument("--base_dir", type=str, default="/pscratch/sd/t/tihsu/database/GridStudy_v2")
    parser.add_argument("--yaml_path", type=str, default="sample.yaml")
    parser.add_argument("--features_yaml", type=str, default=None, help="YAML file specifying features to use")
    parser.add_argument("--mX", type=float, default=None)
    parser.add_argument("--mY", type=float, default=None)
    parser.add_argument("--lumi", type=float, default=300000)
    parser.add_argument("--param-mx-step", type=int, default=1, help="Sparse grid step for MX in parametrized training")
    parser.add_argument("--param-my-step", type=int, default=1, help="Sparse grid step for MY in parametrized training")

    # Model Config
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "tabpfn"])
    parser.add_argument("--parameterize", action="store_true", help="Include Mass as input")
    parser.add_argument("--features", nargs="+", help="Explicit list of features to use")
    parser.add_argument("--tabpfn_limit", type=int, default=50000)
    parser.add_argument("--max_bkg_entries", type=int, default=None, help="Max entries to load for training")
    # IO
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--stage", type=str, default=["train", "predict", "evaluate"], nargs="+", help="Pipeline stages to run")

    args = parser.parse_args()

    if not args.parameterize and (args.mX is None or args.mY is None):
        parser.error("Specify -mX and -mY, or use --parameterize for mass parameterization.")

    if args.features_yaml:
        with open(args.features_yaml) as f:
            feat_cfg = yaml.safe_load(f)
            args.features = feat_cfg.get('features', [])

    run_pipeline(args)