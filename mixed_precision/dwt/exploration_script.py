import os
import json
import itertools
import subprocess
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== Configuration ==========
base_path = os.path.dirname(os.path.abspath(__file__))
dwt_script = os.path.join(base_path, "data_generator.py")

# Float types and scaling methods to explore
float_inputs = ["FP8_CUSTOM", "FP16", "FP32"]
scaling_methods = ["multiplicative", "standardize"]  # 'normalize' handled separately below
manual_scales = [0.0025, 0.5]  # optionally include fixed scales
normalize_ranges = [(0.0, 1.0), (-1.0, 1.0)]  # Use floating-point values with decimal point to maintain precision and consistency in folder naming
mantissa_bits_options = [1, 2, 3, 4, 5, 6]  # Mantissa bits for FP8_CUSTOM; other types will use 0

# ========= Argument Parser ==========
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=256, help='Input signal length')
parser.add_argument('--levels', type=int, default=4, help='Wavelet decomposition levels')
parser.add_argument('--mode', type=str, default="sym4", help='Wavelet family name')
parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
args = parser.parse_args()

input_size = args.input_size
levels = args.levels
family = args.mode
max_workers = args.max_workers

# ========= Build Commands ==========
commands_to_run = []
info_for_each_run = []

# Handle normalization ranges separately
for (norm_min, norm_max) in normalize_ranges:
    norm_tag = f"{norm_min}_{norm_max}"
    scale_tag = f"normalize/{norm_tag}"
    for float_combo in itertools.product(float_inputs, repeat=3):
        float_m, float_n, float_p = float_combo
        float_str = f"{float_m},{float_n},{float_p}"

        mantissa_list = mantissa_bits_options if "FP8_CUSTOM" in (float_m, float_n, float_p) else [0]

        for mantissa in mantissa_list:
            cmd = f"/bin/python3 {dwt_script} " \
                  f"--input_size={input_size} --levels={levels} --mode={family} " \
                  f"--float_type={float_str} --mantissa_bits={mantissa} " \
                  f"--scaling_method=normalize --target_range={norm_min},{norm_max} --exploration_flag=true"

            info_for_each_run.append({
                "float_str": float_str,
                "float_m": float_m,
                "float_n": float_n,
                "float_p": float_p,
                "mantissa": mantissa,
                "scaling_method": "normalize",
                "scale_tag": scale_tag,
                "target_range": f"{norm_min},{norm_max}",
                "scale_value": None
            })
            commands_to_run.append(cmd)

# Add other scaling methods (none, standardize, and manual scales)
for method in scaling_methods + [f"scale_{s}" for s in manual_scales]:
    scale_value = None
    scaling_method = method

    if method.startswith("scale_"):
        scale_value = float(method.replace("scale_", ""))
        scaling_method = "multiplicative"
        scale_tag = f"multiplicative/scale_{scale_value}"
    elif method == "multiplicative":
        for scale_value in manual_scales:
            scale_tag = f"multiplicative/scale_{scale_value}"
            for float_combo in itertools.product(float_inputs, repeat=3):
                float_m, float_n, float_p = float_combo
                float_str = f"{float_m},{float_n},{float_p}"

                mantissa_list = mantissa_bits_options if "FP8_CUSTOM" in (float_m, float_n, float_p) else [0]

                for mantissa in mantissa_list:
                    cmd = f"/bin/python3 {dwt_script} " \
                          f"--input_size={input_size} --levels={levels} --mode={family} " \
                          f"--float_type={float_str} --mantissa_bits={mantissa} " \
                          f"--scaling_method=multiplicative --scale_value={scale_value} --exploration_flag=true"

                    info_for_each_run.append({
                        "float_str": float_str,
                        "float_m": float_m,
                        "float_n": float_n,
                        "float_p": float_p,
                        "mantissa": mantissa,
                        "scaling_method": "multiplicative",
                        "scale_tag": scale_tag,
                        "target_range": None,
                        "scale_value": scale_value
                    })
                    commands_to_run.append(cmd)
        continue
    elif method == "standardize":
        scale_tag = "standardize/minax_0_mean_0_std_1"
    else:
        scale_tag = method

    for float_combo in itertools.product(float_inputs, repeat=3):
        float_m, float_n, float_p = float_combo
        float_str = f"{float_m},{float_n},{float_p}"

        mantissa_list = mantissa_bits_options if "FP8_CUSTOM" in (float_m, float_n, float_p) else [0]

        for mantissa in mantissa_list:
            cmd = f"/bin/python3 {dwt_script} " \
                  f"--input_size={input_size} --levels={levels} --mode={family} " \
                  f"--float_type={float_str} --mantissa_bits={mantissa} " \
                  f"--scaling_method={scaling_method} --exploration_flag=true"

            if scale_value:
                cmd += f" --scale_value={scale_value}"

            info_for_each_run.append({
                "float_str": float_str,
                "float_m": float_m,
                "float_n": float_n,
                "float_p": float_p,
                "mantissa": mantissa,
                "scaling_method": method,
                "scale_tag": scale_tag,
                "target_range": None,
                "scale_value": scale_value
            })
            commands_to_run.append(cmd)

# ========= Run and Collect ==========
metrics_list = []

def run_and_collect(index_and_command):
    idx, command = index_and_command
    info = info_for_each_run[idx]

    try:
        subprocess.run(command, shell=True, check=True)
        folder = os.path.join("exploration", info["scale_tag"], f"mode_{family}", f"levels_{levels}")
        json_file = f"error_metric__{input_size}_{info['float_m']}_{info['float_n']}_{info['float_p']}_{info['mantissa']}.json"
        json_path = os.path.join(folder, json_file)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                metrics = json.load(f)
                metrics.update({
                    "float_type": info["float_str"],
                    "mantissa_bits": info["mantissa"],
                    "scaling_method": info["scaling_method"],
                    "scale_tag": info["scale_tag"],
                    "scale_value": info["scale_value"],
                    "target_range": info["target_range"],
                    "input_size": input_size,
                    "levels": levels,
                    "mode": family
                })
                return metrics
        else:
            print(f"Warning: Missing JSON {json_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return None

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_and_collect, (i, cmd)) for i, cmd in enumerate(commands_to_run)]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Exploring DWT Kernel"):
        res = f.result()
        if res:
            metrics_list.append(res)

# ========= Save and Visualize ==========
print("\n✅ DWT Exploration Finished")
df = pd.DataFrame(metrics_list)
outfile = os.path.join("exploration", "dwt_exploration_results.xlsx")
df.to_excel(outfile, index=False)
print(f"Saved results to {outfile}")
print(df.head())

try:
    import matplotlib.pyplot as plt
    df.plot(x="float_type", y=["MAE", "MSE", "RMSE"], kind="bar", figsize=(14, 6))
    plt.title("DWT Error Metrics Across Float Types and Scaling Methods")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed; skipping plot.")
