#!/usr/bin/env python3
import re
import json
import yaml
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("make_mixed_npz_per_mass")


@dataclass
class DatasetInfo:
    name: str
    category: str
    path: Path
    is_signal: bool
    xsec: float = 1.0
    nevents: float = 1.0
    mx: float = 0.0
    my: float = 0.0


def process_mass_group(
    mx: int,
    my: int,
    sig_list: List[DatasetInfo],
    bkg_ds: List[DatasetInfo],
    out_dir: str,
    lumi: float,
    chunk_size: int,
    compress: bool,
    abs_weight: bool,
    scale2: float,
) -> None:
    out_base = Path(out_dir)
    mass_dir = out_base / f"MX-{mx}_MY-{my}"
    log.info(f"Building mass group: MX={mx} MY={my} -> {mass_dir}")

    for split in ["train", "valid"]:
        writer = ShardWriter(mass_dir / split, prefix="mixed", chunk_size=chunk_size, compress=compress)

        # Add all backgrounds
        for ds in bkg_ds:
            files = iter_pt_files(ds, split)
            if not files:
                continue
            for fp in files:
                loaded = load_one_pt(fp)
                if loaded is None:
                    continue
                x, cond, x_mask, raw_w = loaded
                x, x_mask = fix_x_and_mask(x, x_mask, n_obj=18, n_feat=7)

                nevt = float(ds.nevents) if float(ds.nevents) != 0.0 else 1.0
                w = raw_w * (float(ds.xsec) * float(lumi) / nevt) * scale2
                if abs_weight:
                    w = w.abs()

                writer.add(x, x_mask, cond, w, y=0)

        # Add signals for this mass
        for ds in sig_list:
            files = iter_pt_files(ds, split)
            if not files:
                continue
            for fp in files:
                loaded = load_one_pt(fp)
                if loaded is None:
                    continue
                x, cond, x_mask, raw_w = loaded
                x, x_mask = fix_x_and_mask(x, x_mask, n_obj=18, n_feat=7)

                nevt = float(ds.nevents) if float(ds.nevents) != 0.0 else 1.0
                w = raw_w * (float(ds.xsec) * float(lumi) / nevt) * scale2
                if abs_weight:
                    w = w.abs()

                writer.add(x, x_mask, cond, w, y=1)

        writer.flush()

class ConfigLoader:
    def __init__(self, yaml_path: str, base_data_dir: str):
        self.yaml_path = Path(yaml_path)
        self.base_dir = Path(base_data_dir)
        self.signal_config: Dict[str, Any] = {}
        self.bkg_config: Dict[str, Any] = {}
        self._load_yaml()

    def _load_yaml(self) -> None:
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML config not found: {self.yaml_path}")
        with open(self.yaml_path) as f:
            raw = yaml.safe_load(f)
        self.signal_config = raw.get("signal", {}) if isinstance(raw, dict) else {}
        self.bkg_config = raw.get("background", {}) if isinstance(raw, dict) else {}

    @staticmethod
    def parse_mass(folder_name: str) -> Tuple[float, float]:
        m = re.search(r"MX-(\d+)_MY-(\d+)", folder_name)
        if m:
            return float(m.group(1)), float(m.group(2))
        return 0.0, 0.0

    def discover_datasets(self) -> List[DatasetInfo]:
        found: List[DatasetInfo] = []
        existing_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]

        # Backgrounds from YAML
        for name, cfg in (self.bkg_config or {}).items():
            matched = [f for f in existing_folders if str(name) in f.name]
            if not matched:
                log.warning(f"Background '{name}' not found under {self.base_dir}")
                continue
            target = matched[0]
            xsec = float(cfg.get("xsec", 1.0))

            cutflow_json_first = target / "../cutflow.json"
            if cutflow_json_first.exists():
                with open(cutflow_json_first, "r") as f:
                    cutflow = json.load(f)
                nevents = cutflow[name].get("total", None)
                if nevents is None:
                    raise ValueError(f"Missing 'all' in {cutflow_json_first}")
                nevents = float(nevents)
                if nevents == 0.0:
                    nevents = 1.0
                print("Using cutflow from", cutflow_json_first)
            else:
                nevents = float(cfg.get("nEvent", 1.0))

            if nevents == 0.0:
                nevents = 1.0
                
            found.append(
                DatasetInfo(
                    name=str(name),
                    category=str(cfg.get("name", "background")),
                    path=target,
                    is_signal=False,
                    xsec=xsec,
                    nevents=nevents,
                )
            )

        # Signals by folder naming
        for folder in existing_folders:
            if "MX-" not in folder.name:
                continue
            mx, my = self.parse_mass(folder.name)

            cutflow_json_first = folder / "../cutflow.json"
            if cutflow_json_first.exists():
                with open(cutflow_json_first, "r") as f:
                    cutflow = json.load(f)
                nevents = cutflow[folder.name].get("total", None)
                if nevents is None:
                    raise ValueError(f"Missing 'all' in {cutflow_json_first}")
                nevents = float(nevents)
                if nevents == 0.0:
                    nevents = 1.0
                print("Using cutflow from", cutflow_json_first)
            else:
                cutflow_json = folder / "cutflow.json"
                if cutflow_json.exists():
                    with open(cutflow_json, "r") as f:
                        cutflow = json.load(f)
                    nevents = cutflow.get("all", None)
                    if nevents is None:
                        raise ValueError(f"Missing 'all' in {cutflow_json}")
                    nevents = float(nevents)
                    if nevents == 0.0:
                        nevents = 1.0
                else:
                    nevents = 184000.0



            found.append(
                DatasetInfo(
                    name=folder.name,
                    category="signal",
                    path=folder,
                    is_signal=True,
                    xsec=0.01,
                    nevents=nevents,
                    mx=mx,
                    my=my,
                )
            )

        log.info(f"Discovered {len(found)} datasets ({sum(d.is_signal for d in found)} signal).")
        return found


def iter_pt_files(ds: DatasetInfo, split: str) -> List[Path]:
    d = ds.path / "evenet" / split
    if not d.exists():
        return []
    return sorted(d.glob("*.pt"))


def torch_load_cpu(fp: Path) -> Dict[str, Any]:
    try:
        return torch.load(fp, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(fp, map_location="cpu")


def load_one_pt(fp: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    d = torch_load_cpu(fp)

    x = d.get("x", None)
    if x is None:
        return None

    g = d.get("global", None)
    if g is None:
        g = d.get("globals", None)

    x_mask = d.get("x_mask", None)
    if g is None or x_mask is None:
        return None

    x = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    g = torch.as_tensor(g, dtype=torch.float32, device="cpu")
    x_mask = torch.as_tensor(x_mask, device="cpu")

    w = d.get("weights", None)
    if w is None:
        w = torch.ones((x.shape[0],), dtype=torch.float32, device="cpu")
    else:
        w = torch.as_tensor(w, dtype=torch.float32, device="cpu").view(-1)
        if w.shape[0] != x.shape[0]:
            raise ValueError(f"weights length mismatch in {fp}: {w.shape[0]} vs N={x.shape[0]}")

    if g.ndim == 1:
        g = g.unsqueeze(1)
    elif g.ndim != 2:
        raise ValueError(f"globals must be [N,C] or [N], got {tuple(g.shape)}")

    return x, g, x_mask, w


def fix_x_and_mask(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    n_obj: int = 18,
    n_feat: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError(f"x must be [N,M,F], got {tuple(x.shape)}")
    n, m, f = x.shape

    if f > n_feat:
        x = x[:, :, :n_feat]
    elif f < n_feat:
        pad = torch.zeros((n, m, n_feat - f), dtype=x.dtype)
        x = torch.cat([x, pad], dim=2)

    x_mask = (x_mask > 0.5)
    if x_mask.ndim != 2:
        raise ValueError(f"x_mask must be [N,M], got {tuple(x_mask.shape)}")

    if m > n_obj:
        x = x[:, :n_obj, :]
        x_mask = x_mask[:, :n_obj]
    elif m < n_obj:
        pad_x = torch.zeros((n, n_obj - m, n_feat), dtype=x.dtype)
        pad_m = torch.zeros((n, n_obj - m), dtype=torch.bool)
        x = torch.cat([x, pad_x], dim=1)
        x_mask = torch.cat([x_mask, pad_m], dim=1)

    return x, x_mask


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, chunk_size: int, compress: bool = True):
        self.out_dir = out_dir
        self.prefix = prefix
        self.chunk_size = int(chunk_size)
        self.compress = compress
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.reset()
        self.shard_idx = 0

    def reset(self) -> None:
        self.buf_x: List[torch.Tensor] = []
        self.buf_m: List[torch.Tensor] = []
        self.buf_c: List[torch.Tensor] = []
        self.buf_w: List[torch.Tensor] = []
        self.buf_y: List[torch.Tensor] = []
        self.n = 0

    def add(self, x: torch.Tensor, x_mask: torch.Tensor, cond: torch.Tensor, w: torch.Tensor, y: int) -> None:
        n = x.shape[0]
        self.buf_x.append(x)
        self.buf_m.append(x_mask)
        self.buf_c.append(cond)
        self.buf_w.append(w)
        self.buf_y.append(torch.full((n,), int(y), dtype=torch.int32))
        self.n += n
        if self.n >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if self.n == 0:
            return

        x = torch.cat(self.buf_x, dim=0).numpy().astype(np.float32)
        x_mask = torch.cat(self.buf_m, dim=0).numpy().astype(bool)
        conditions = torch.cat(self.buf_c, dim=0).numpy().astype(np.float32)
        event_weight = torch.cat(self.buf_w, dim=0).numpy().astype(np.float32)
        classification = torch.cat(self.buf_y, dim=0).numpy().astype(np.int32)

        n = x.shape[0]
        conditions_mask = np.ones((n, 1), dtype=bool)

        num_sequential_vectors = x_mask.sum(axis=1).astype(np.int32)
        num_vectors = (num_sequential_vectors + 1).astype(np.int32)

        out_fp = self.out_dir / f"{self.prefix}_{self.shard_idx:04d}.npz"
        saver = np.savez_compressed if self.compress else np.savez
        saver(
            out_fp,
            num_vectors=num_vectors,
            num_sequential_vectors=num_sequential_vectors,
            x=x,
            x_mask=x_mask,
            conditions=conditions,
            conditions_mask=conditions_mask,
            classification=classification,
            event_weight=event_weight,
        )
        log.info(f"Wrote {out_fp} (N={n})")

        self.shard_idx += 1
        self.reset()


def build_mass_groups(signals: List[DatasetInfo]) -> Dict[Tuple[int, int], List[DatasetInfo]]:
    groups: Dict[Tuple[int, int], List[DatasetInfo]] = {}
    for ds in signals:
        key = (int(round(ds.mx)), int(round(ds.my)))
        groups.setdefault(key, []).append(ds)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser("Make mixed NPZ per signal mass; keep train/valid separate.")
    ap.add_argument("--base_dir", required=True, type=str)
    ap.add_argument("--yaml_path", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--lumi", type=float, default=300000.0)
    ap.add_argument("--chunk_size", type=int, default=200_000)
    ap.add_argument("--no_compress", action="store_true")
    ap.add_argument("--abs_weight", action="store_true")
    ap.add_argument("--split_factor2", action="store_true", help="Multiply weights by 2 for train/valid halves.")
    ap.add_argument("--mx", type=int, default=None)
    ap.add_argument("--my", type=int, default=None)
    ap.add_argument("--nproc", type=int, default=1, help="Number of CPU processes over mass points.")

    args = ap.parse_args()

    cfg = ConfigLoader(args.yaml_path, args.base_dir)
    all_ds = cfg.discover_datasets()

    bkg_ds = [d for d in all_ds if not d.is_signal]
    sig_ds = [d for d in all_ds if d.is_signal]

    # Optional mass filter
    if args.mx is not None:
        sig_ds = [d for d in sig_ds if int(round(d.mx)) == int(args.mx)]
    if args.my is not None:
        sig_ds = [d for d in sig_ds if int(round(d.my)) == int(args.my)]

    if not sig_ds:
        raise SystemExit("No signal datasets selected. Check --mx/--my filters.")

    mass_groups = build_mass_groups(sig_ds)

    out_base = Path(args.out_dir)
    compress = not args.no_compress
    scale2 = 2.0 if args.split_factor2 else 1.0
    mass_items = sorted(mass_groups.items())

    if args.nproc > 1 and len(mass_items) > 1:
        log.info(f"Parallel mode: nproc={args.nproc}, mass_points={len(mass_items)}")
        with ProcessPoolExecutor(max_workers=args.nproc) as ex:
            futs = {}
            for (mx, my), sig_list in mass_items:
                fut = ex.submit(
                    process_mass_group,
                    mx, my, sig_list, bkg_ds,
                    str(out_base),
                    float(args.lumi),
                    int(args.chunk_size),
                    bool(compress),
                    bool(args.abs_weight),
                    float(scale2),
                )
                futs[fut] = (mx, my)

            for fut in as_completed(futs):
                mx, my = futs[fut]
                fut.result()
                log.info(f"Finished MX={mx} MY={my}")
    else:
        # serial fallback (also used when only 1 mass point)
        for (mx, my), sig_list in mass_items:
            process_mass_group(
                mx, my, sig_list, bkg_ds,
                str(out_base),
                float(args.lumi),
                int(args.chunk_size),
                bool(compress),
                bool(args.abs_weight),
                float(scale2),
            )

    log.info(f"Done. Output: {out_base}")


if __name__ == "__main__":
    main()
