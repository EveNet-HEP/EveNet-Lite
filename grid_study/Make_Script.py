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
        "shifter python3 train_pc_mva.py "
        "--base_dir /pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_v2/data "
        "--yaml_path config/sample.yaml "
        "--mX {mX} --mY {mY} "
        "--out_dir /pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_v2/method/ "
        "--ensemble 3 --learning_rate 0.0005 "
    )

    base_cmd_xgb = (
        "python3 train_tabular_mva.py "
        "--base_dir /pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_v2/data "
        "--yaml_path config/sample.yaml "
        "--features_yaml config/feature.yaml "
        "--out_dir /pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_v2/method/ "
        "--model xgb "
    )


    base_cmd_tabpfn = (
        "python3 train_tabular_mva.py "
        "--base_dir /pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_v2/data "
        "--yaml_path config/sample.yaml "
        "--features_yaml config/feature.yaml "
        "--out_dir /pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_v2/method/ "
        "--model tabpfn "
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
    filenames_raw = {
        "scratch": os.path.join(farm_dir, "run_scratch.sh"),
        "pretrain": os.path.join(farm_dir, "run_pretrain.sh"),
        "xgboost": os.path.join(farm_dir, "run_xgboost.sh"),
        "tabpfn":  os.path.join(farm_dir, "run_tabpfn.sh")
    }

    filenames = {
        f"{k}_{mode}": v.replace(".sh", f"_{mode}.sh")
        for k, v in filenames_raw.items()
        for mode in ["train", "predict", "evaluate"]
    }

    # Open all files
    files = {k: open(v, "w") for k, v in filenames.items()}

    # Write headers
    for f in files.values():
        f.write("#!/bin/bash\n\n")

    count = 0
    for mode in ["train", "predict", "evaluate"]:
        for key in keys:
            match = pattern.search(key)
            if match:
                mX = match.group(1)
                mY = match.group(2)

                # --- Write Commands ---
                # 1. Scratch
                files[f"scratch_{mode}"].write(base_cmd_pc.format(mX=mX, mY=mY) + f" --stage {mode} --batch_size 1024 --gamma 0.5 --epochs 20 \n")

                # 2. Pretrain
                files[f"pretrain_{mode}"].write(base_cmd_pc.format(mX=mX, mY=mY) + f" --pretrain --stage {mode}  --use_adapter --batch_size 2048 --gamma 0.5 --epochs 20 \n")

                # 3. XGBoost
                files[f"xgboost_{mode}"].write(
                    base_cmd_xgb + f"--mX {mX} --mY {mY} --stage {mode}\n"
                )
                files[f"tabpfn_{mode}"].write(
                    base_cmd_tabpfn + f"--mX {mX} --mY {mY} --stage {mode}\n"
                )
                count += 1

    with open(os.path.join(farm_dir, f"run_param_train_pretrain.sh"), "w") as f:
        for num_sparse in [1, 2]:
        # train
            mX = 500 # deosn't matter
            mY = 90 # deosn't matter
            cmd = (base_cmd_pc.format(mX=mX, mY=mY) + f" --stage train --parameterize  --pretrain  --param-mx-step {num_sparse} --param-my-step {num_sparse} --batch_size 1024 --gamma 0.3 --epochs 30  ")
            f.write(f'bash -c "source ../NERSC/export_DDP_vars.sh && {cmd}"\n')
    with open(os.path.join(farm_dir, f"run_param_predict_pretrain.sh"), "w") as f:
        # predict, eval
        for num_sparse in [1, 2]:
            for key in keys:
                match = pattern.search(key)
                if match:
                    mX = match.group(1)
                    mY = match.group(2)
                    f.write(base_cmd_pc.format(mX=mX, mY=mY) + f" --stage predict --parameterize --pretrain  --param-mx-step {num_sparse} --param-my-step {num_sparse} --batch_size 1024\n")

    with open(os.path.join(farm_dir, f"run_param_evaluate_pretrain.sh"), "w") as f:
        # predict, eval
        for num_sparse in [1, 2]:
            for key in keys:
                match = pattern.search(key)
                if match:
                    mX = match.group(1)
                    mY = match.group(2)
                    f.write(base_cmd_pc.format(mX=mX, mY=mY) + f" --stage evaluate --parameterize --pretrain --param-mx-step {num_sparse} --param-my-step {num_sparse} --batch_size 1024\n")

    with open(os.path.join(farm_dir, f"run_param_train_scratch.sh"), "w") as f:
        for num_sparse in [1, 2]:
            # train
            mX = 500  # deosn't matter
            mY = 90  # deosn't matter
            cmd = (base_cmd_pc.format(mX=mX, mY=mY) + f" --stage train --parameterize --param-mx-step {num_sparse} --param-my-step {num_sparse} --batch_size 1024 --gamma 0.3 ")
            cmd = cmd.replace("--epochs 20", "--epochs 40")
            f.write(f'bash -c "source ../NERSC/export_DDP_vars.sh && {cmd}"\n')
    with open(os.path.join(farm_dir, f"run_param_predict_scratch.sh"), "w") as f:
        # predict, eval
        for num_sparse in [1, 2]:
            for key in keys:
                match = pattern.search(key)
                if match:
                    mX = match.group(1)
                    mY = match.group(2)
                    f.write(base_cmd_pc.format(mX=mX, mY=mY) + f" --stage predict --parameterize --param-mx-step {num_sparse} --param-my-step {num_sparse} --batch_size 1024\n")

    with open(os.path.join(farm_dir, f"run_param_evaluate_scratch.sh"), "w") as f:
        # predict, eval
        for num_sparse in [1, 2]:
            for key in keys:
                match = pattern.search(key)
                if match:
                    mX = match.group(1)
                    mY = match.group(2)
                    f.write(base_cmd_pc.format(mX=mX,
                                                mY=mY) + f" --stage evaluate --parameterize --param-mx-step {num_sparse} --param-my-step {num_sparse} --batch_size 1024\n")

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
