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
kmeans_script = os.path.join(base_path, "data_generator.py")

# Float types and scaling methods to explore
float_inputs = ["FP8_CUSTOM", "FP16", "FP32"]
scaling_methods = ["multiplicative", "standardize"]  # 'normalize' handled separately below
manual_scales = [0.0025, 0.5]  # optionally include fixed scales
normalize_ranges = [(0.0, 1.0), (-1.0, 1.0)]  # normalize ranges
mantissa_bits_options = [1, 2, 3, 4, 5, 6]

# ========= Argument Parser ==========
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=32, help='Number of data points')
parser.add_argument('--features', type=int, default=8, help='Number of features per point')
parser.add_argument('--num_clusters', type=int, default=9, help='Number of clusters')
parser.add_argument('--cores', type=int, default=1, help='Number of parallel cores to simulate')
parser.add_argument('--max_workers', type=int, default=2, help='Number of parallel workers')
args = parser.parse_args()

input_size = args.input_size
num_features = args.features
num_clusters = args.num_clusters
cores = args.cores
max_workers = args.max_workers

# ========= Build Commands ==========
commands_to_run = []
info_for_each_run = []

# ---- Handle normalization cases ----
for (norm_min, norm_max) in normalize_ranges:
    norm_tag = f"{norm_min}_{norm_max}"
    scale_tag = norm_tag

    for float_combo in itertools.product(float_inputs, repeat=2):
        float_m, float_p = float_combo
        float_str = f"{float_m},{float_p}"

        mantissa_list = mantissa_bits_options if "FP8_CUSTOM" in (float_m, float_p) else [0]

        for mantissa in mantissa_list:
            cmd = f"/bin/python3 {kmeans_script} " \
                  f"--input_size={input_size} --features={num_features} --num_clusters={num_clusters} --cores={cores} " \
                  f"--float_type={float_str} --mantissa_bits={mantissa} " \
                  f"--scaling_method=normalize --target_range={norm_min},{norm_max} --exploration_flag=true"

            info_for_each_run.append({
                "float_str": float_str,
                "float_m": float_m,
                "float_p": float_p,
                "mantissa": mantissa,
                "scaling_method": "normalize",
                "scale_tag": scale_tag,
                "target_range": f"{norm_min},{norm_max}",
                "scale_value": None
            })
            commands_to_run.append(cmd)

# ---- Handle multiplicative and standardize cases ----
for method in scaling_methods + [f"scale_{s}" for s in manual_scales]:
    scale_value = None
    scaling_method = method

    if method.startswith("scale_"):
        scale_value = float(method.replace("scale_", ""))
        scaling_method = "multiplicative"
        scale_tag = f"scale_{scale_value}"
    elif method == "multiplicative":
        continue  # already handled via manual_scales
    elif method == "standardize":
        scale_tag = "minax_0_mean_0_std_1"
    else:
        scale_tag = method

    for float_combo in itertools.product(float_inputs, repeat=2):
        float_m, float_p = float_combo
        float_str = f"{float_m},{float_p}"

        mantissa_list = mantissa_bits_options if "FP8_CUSTOM" in (float_m, float_p) else [0]

        for mantissa in mantissa_list:
            cmd = f"/bin/python3 {kmeans_script} " \
                  f"--input_size={input_size} --features={num_features} --num_clusters={num_clusters} --cores={cores} " \
                  f"--float_type={float_str} --mantissa_bits={mantissa} " \
                  f"--scaling_method={scaling_method} --exploration_flag=true"

            if scale_value:
                cmd += f" --scale_value={scale_value}"

            info_for_each_run.append({
                "float_str": float_str,
                "float_m": float_m,
                "float_p": float_p,
                "mantissa": mantissa,
                "scaling_method": scaling_method,
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
        folder = os.path.join("exploration", info["scaling_method"], info["scale_tag"])
        json_file = f"error_metric__{input_size}_{info['float_m']}_{info['float_p']}_{info['mantissa']}.json"
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
                    "num_features": num_features,
                    "num_clusters": num_clusters
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
    for f in tqdm(as_completed(futures), total=len(futures), desc="Exploring KMeans Kernel"):
        res = f.result()
        if res:
            metrics_list.append(res)

# ========= Save and Visualize ==========
print("\n✅ KMeans Exploration Finished")
df = pd.DataFrame(metrics_list)
outfile = os.path.join("exploration", "kmeans_exploration_results.xlsx")
df.to_excel(outfile, index=False)
print(f"Saved results to {outfile}")
print(df.head())

try:
    import matplotlib.pyplot as plt
    df.plot(x="float_type", y=["MAE", "MSE", "RMSE"], kind="bar", figsize=(14, 6))
    plt.title("KMeans Error Metrics Across Float Types and Scaling Methods")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed; skipping plot.")
