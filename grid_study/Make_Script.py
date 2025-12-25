import json
import re
import os
import stat
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training scripts from sample_list.json")
    parser.add_argument(
        "--farm_dir",
        type=str,
        required=True,
        help="The directory (Farm) where the .sh scripts will be created"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="sample_list.json",
        help="Path to the sample_list.json file"
    )
    return parser.parse_args()


def generate_shell_scripts(args):
    # 1. Configuration
    # ----------------
    base_cmd_pc = (
        "python3 train_pc_mva.py "
        "--base_dir /pscratch/sd/t/tihsu/database/GridStudy_v2/ "
        "--yaml_path config/sample.yaml "
        "--mX {mX} --mY {mY} "
        "--epochs 20 --batch_size 1024 "
        "--out_dir /pscratch/sd/t/tihsu/database/GridStudy_v2/method/ "
        "--ensemble 3"
    )

    base_cmd_xgb = (
        "python3 train_tabular_mva.py "
        "--base_dir /pscratch/sd/t/tihsu/database/GridStudy_v2/ "
        "--yaml_path config/sample.yaml "
        "--features_yaml config/feature.yaml "
        "--mX {mX} --mY {mY} "
        "--out_dir /pscratch/sd/t/tihsu/database/GridStudy_v2/method/ "
        "--model xgb"
    )

    # 2. Setup Farm Directory
    # -----------------------
    farm_dir = args.farm_dir
    if not os.path.exists(farm_dir):
        print(f"Creating Farm directory: {farm_dir}")
        os.makedirs(farm_dir, exist_ok=True)

    # 3. Parse JSON
    # -------------
    if not os.path.exists(args.json_file):
        print(f"Error: {args.json_file} not found.")
        return

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    if "signal" not in data:
        print("Error: Key 'signal' not found in JSON.")
        return

    pattern = re.compile(r"MX-(\d+)_MY-(\d+)")
    keys = sorted(data["signal"].keys())

    # 4. Open output files inside the Farm directory
    # ----------------------------------------------
    filenames = {
        "scratch": os.path.join(farm_dir, "run_scratch.sh"),
        "pretrain": os.path.join(farm_dir, "run_pretrain.sh"),
        "xgboost": os.path.join(farm_dir, "run_xgboost.sh")
    }

    # Open all files
    files = {k: open(v, "w") for k, v in filenames.items()}

    # Write headers
    for f in files.values():
        f.write("#!/bin/bash\n\n")

    count = 0
    for key in keys:
        match = pattern.search(key)
        if match:
            mX = match.group(1)
            mY = match.group(2)

            # --- Write Commands ---
            # 1. Scratch
            files["scratch"].write(base_cmd_pc.format(mX=mX, mY=mY) + "\n")

            # 2. Pretrain
            files["pretrain"].write(base_cmd_pc.format(mX=mX, mY=mY) + " --pretrain\n")

            # 3. XGBoost
            files["xgboost"].write(base_cmd_xgb.format(mX=mX, mY=mY) + "\n")

            count += 1

    # 5. Cleanup and Permissions
    # --------------------------
    for key, f in files.items():
        f.close()
        filepath = filenames[key]

        # Make executable
        st = os.stat(filepath)
        os.chmod(filepath, st.st_mode | stat.S_IEXEC)
        print(f"Generated: {filepath}")

    print(f"\nSuccess! {count} jobs written to {farm_dir}/")


if __name__ == "__main__":
    args = parse_args()
    generate_shell_scripts(args)