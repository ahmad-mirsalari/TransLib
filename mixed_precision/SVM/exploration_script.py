import itertools
import subprocess
import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== Configuration ==========
base_path = os.path.dirname(os.path.abspath(__file__))
svm_script = os.path.join(base_path, "data_generator.py")

# Single clean command template
command_template = (
    f"/bin/python3 {svm_script} --input_size={{}} --kernel={{}} --dataset={{}} "
    f"--float_type={{}} --mantissa_bits={{}} --mac_flag=false --vec_flag=false --exploration_flag=true"
)

# Float types to explore
float_inputs = ["FP8_CUSTOM", "FP16","FP16ALT", "FP32"]

# ========= Argument Parser ==========
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=64, help='Input feature vector size')
parser.add_argument('--kernel', type=str, default="linear", help='Kernel type (e.g., linear, rbf)')
parser.add_argument('--dataset', type=str, default="bill", help='Dataset name (e.g., cancer, bill)')
parser.add_argument('--max_workers', type=int, default=1, help='Number of parallel workers')
args = parser.parse_args()

input_size = args.input_size
kernel = args.kernel
dataset = args.dataset
max_workers = args.max_workers



# ========= Prepare All Commands ==========
commands_to_run = []
info_for_each_run = []

for float_combo in itertools.product(float_inputs, repeat=3):
    float_m, float_n, float_p = float_combo
    float_str = f"{float_m},{float_n},{float_p}"

    if "FP8_CUSTOM" in (float_m, float_n, float_p):
        mantissa_bits_list = [1, 2, 3, 4, 5, 6]
    else:
        mantissa_bits_list = [0]

    for mantissa_bit in mantissa_bits_list:
        command = command_template.format(input_size, kernel, dataset, float_str, mantissa_bit)
        commands_to_run.append(command)
        info_for_each_run.append({
            "float_str": float_str,
            "float_m": float_m,
            "float_n": float_n,
            "float_p": float_p,
            "mantissa_bit": mantissa_bit,
        })

# ========= Run Commands in Parallel ==========
metrics_list = []

def run_and_collect(command_info):
    idx, command = command_info
    info = info_for_each_run[idx]

    try:
        subprocess.run(command, shell=True, check=True)

        output_folder = os.path.join(os.getcwd(), "exploration", kernel, dataset)
        json_filename = f"error_metric__{input_size}_{info['float_m']}_{info['float_n']}_{info['float_p']}_{info['mantissa_bit']}.json"
        json_path = os.path.join(output_folder, json_filename)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                metrics = json.load(f)

            metrics.update({
                "float_type": info["float_str"],
                "mantissa_bits": info["mantissa_bit"],
                "input_size": input_size,
                "kernel": kernel,
                "dataset": dataset
            })

            return metrics
        else:
            print(f"Warning: JSON not found: {json_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return None

# Create thread pool
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_and_collect, (idx, cmd)) for idx, cmd in enumerate(commands_to_run)]

    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing SVM Exploration"):
        result = f.result()
        if result:
            metrics_list.append(result)

# ========= After All Runs ==========
print("\n==== Parallel SVM Exploration Finished! ====")

# Save results
df = pd.DataFrame(metrics_list)
output_excel_path = os.path.join(os.getcwd(), "exploration",f"{kernel}_{dataset}_svm_exploration_results_parallel.xlsx")
df.to_excel(output_excel_path, index=False)

print(f"Saved SVM exploration results to {output_excel_path}")
print(df.head())

# Optional Plot
try:
    import matplotlib.pyplot as plt
    df.plot(x="float_type", y="ACC", kind="bar", figsize=(12, 6))
    plt.title("SVM Accuracy Across Float Types and Mantissa Bits")
    plt.xlabel("Float Format Combinations")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed, skipping plots.")
