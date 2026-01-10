import sys
import os
import argparse
import re
import json
import yaml
import numpy as np
import logging


from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time

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
    max_events: int = None


class ConfigLoader:
    def __init__(self, yaml_path: str, base_data_dir: str):
        self.yaml_path = Path(yaml_path)
        self.base_dir = Path(base_data_dir)
        self.signal_config = {}
        self.bkg_config = {}
        self._load_yaml()

    def _load_yaml(self):
        if not self.yaml_path.exists():
            sys.exit(1)
        with open(self.yaml_path) as f:
            raw = yaml.safe_load(f)
            self.signal_config = raw.get('signal', {})
            self.bkg_config = raw.get('background', {})

    def parse_mass(self, folder_name: str) -> Tuple[float, float]:
        match = re.search(r"MX-(\d+)_MY-(\d+)", folder_name)
        if match:
            return float(match.group(1)), float(match.group(2))
        return 0.0, 0.0

    def discover_datasets(self) -> List[DatasetInfo]:
        found_datasets = []
        # Note: Assuming structure base_dir/process_name/...
        existing_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]

        # 1. Discover Backgrounds
        for name, cfg in self.bkg_config.items():
            matched = [f for f in existing_folders if name in f.name]
            if not matched:
                continue

            target = matched[0]

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
                cutflow_json = target / "cutflow.json"
                if cutflow_json.exists():
                    with open(cutflow_json, "r") as f:
                        cutflow = json.load(f)
                    nevents = cutflow.get("all", None)
                    if nevents is None:
                        raise ValueError(f"Missing 'all' in {cutflow_json}")
                    nevents = float(nevents)
                    print("Using cutflow from", cutflow_json)
                    if nevents == 0.0:
                        nevents = 1.0
                else:
                    nevents = cfg.get('nEvent', 1.0)

            found_datasets.append(DatasetInfo(
                name=name,
                path=target,
                is_signal=False,
                xsec=cfg.get('xsec', 1.0),
                nevents=nevents,
                category=cfg.get("name", "background"),
                max_events=cfg.get('max_events', None)
            ))

        # 2. Discover Signals
        for folder in existing_folders:
            if "MX-" in folder.name:
                mx, my = self.parse_mass(folder.name)
                cutflow_json = folder / "cutflow.json"
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
                found_datasets.append(DatasetInfo(
                    name=folder.name,
                    path=folder,
                    is_signal=True,
                    xsec=0.01, # 10 fb
                    nevents=nevents,  # TODO: make configurable, now hardcoded
                    mx=mx,
                    my=my,
                    category="signal",
                    max_events=cfg.get('max_events', None)
                ))

        print(
            f"Discovered {len(found_datasets)} datasets ({len([d for d in found_datasets if d.is_signal])} Signal).")
        return found_datasets
