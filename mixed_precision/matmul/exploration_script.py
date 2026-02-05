import itertools
import subprocess
import argparse
import os
import sys
import json
import glob
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

exploration_dir = Path.cwd() / "exploration"
if exploration_dir.exists() and exploration_dir.is_dir():
    shutil.rmtree(exploration_dir)
# ========== Configuration ==========
BASE_PATH = Path(__file__).resolve().parent
MATRIX_SCRIPT = BASE_PATH / "data_generator.py"

# Use the exact interpreter you're running now
PY = sys.executable

# Command template (list of args, no shell)
def build_cmd(m: int, n: int, p: int, std: str, float_str: str, mantissa_bits: int) -> List[str]:
    return [
        PY, str(MATRIX_SCRIPT),
        f"--m={m}", f"--n={n}", f"--p={p}",
        f"--std={std}",
        f"--float_type={float_str}",
        f"--mantissa_bits={mantissa_bits}",
        "--mac_flag=true",
        "--exploration_flag=true",
    ]

# Possible float and std inputs
FLOAT_INPUTS = ["FP8_CUSTOM", "FP16", "FP16ALT", "FP32"]
STD_INPUTS = ["0.01", "0.5", "1"]

# ========= Argument Parser ==========
parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=16, help='Value for m')
parser.add_argument('--n', type=int, default=16, help='Value for n')
parser.add_argument('--p', type=int, default=16, help='Value for p')
parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
parser.add_argument('--suppress_torch_numpy_warning', action='store_true',
                    help='Silence the "Failed to initialize NumPy" UserWarning from torch.')
args = parser.parse_args()

m_input = args.m
n_input = args.n
p_input = args.p
max_workers = args.max_workers

# ========= Prepare All Commands ==========
commands_to_run: List[List[str]] = []
info_for_each_run: List[Dict[str, Any]] = []

for std_input in STD_INPUTS:
    for float_combo in itertools.product(FLOAT_INPUTS, repeat=3):
        float_m, float_n, float_p = float_combo
        float_str = f"{float_m},{float_n},{float_p}"

        if "FP8_CUSTOM" in (float_m, float_n, float_p):
            mantissa_bits_list = [1, 2, 3, 4, 5, 6]
        else:
            mantissa_bits_list = [0]

        for mantissa_bit in mantissa_bits_list:
            cmd = build_cmd(m_input, n_input, p_input, std_input, float_str, mantissa_bit)
            commands_to_run.append(cmd)
            info_for_each_run.append({
                "std_input": std_input,
                "float_m": float_m,
                "float_n": float_n,
                "float_p": float_p,
                "float_str": float_str,
                "mantissa_bit": mantissa_bit,
            })

# ========= Helpers ==========
def expected_json_path(info: Dict[str, Any]) -> Path:
    std_dir = Path.cwd() / "exploration" / str(float(info["std_input"]))
    json_name = f"error_metric__{m_input}__{n_input}__{p_input}_{info['float_m']}_{info['float_n']}_{info['float_p']}_{info['mantissa_bit']}_{float(info['std_input'])}.json"
    return std_dir / json_name

def find_json_fallback(info: Dict[str, Any]) -> Optional[Path]:
    """
    In case the filename spelling changes slightly (e.g. FP16ALT vs BF16),
    try to glob a close match in the std directory.
    """
    std_dir = Path.cwd() / "exploration" / str(float(info["std_input"]))
    if not std_dir.exists():
        return None
    pat = f"error_metric__{m_input}__{n_input}__{p_input}_*_*_*_{info['mantissa_bit']}_{float(info['std_input'])}.json"
    candidates = sorted(std_dir.glob(pat))
    # Attempt to pick one that actually mentions the three float tokens if possible
    for c in candidates:
        name = c.name
        if info['float_m'] in name and info['float_n'] in name and info['float_p'] in name:
            return c
    return candidates[0] if candidates else None

def collect_metrics(json_path: Path, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        with open(json_path, "r") as f:
            metrics = json.load(f)
        metrics.update({
            "float_type": info["float_str"],
            "mantissa_bits": info["mantissa_bit"],
            "std": float(info["std_input"]),
            "m": m_input,
            "n": n_input,
            "p": p_input
        })
        return metrics
    except Exception as e:
        print(f"Warning: failed reading metrics {json_path}: {e}")
        return None

# ========= Run Commands in Parallel ==========
metrics_list: List[Dict[str, Any]] = []

def run_and_collect(idx: int, cmd: List[str]) -> Optional[Dict[str, Any]]:
    info = info_for_each_run[idx]

    # Optional: suppress that torch UserWarning inside the child process only
    env = os.environ.copy()
    if args.suppress_torch_numpy_warning:
        env["PYTHONWARNINGS"] = "ignore"  # broad; keeps stdout clean

    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Show a compact message (stderr can be long)
        print(f"Command failed ({' '.join(cmd)}): returncode={e.returncode}")
        if e.stderr:
            # Print the last 10 lines of stderr to help debugging
            tail = b"\n".join(e.stderr.splitlines()[-10:])
            print(tail.decode(errors="ignore"))
        return None

    # Collect expected file; if not found, try fallback glob
    jpath = expected_json_path(info)
    if jpath.exists():
        return collect_metrics(jpath, info)
    else:
        # Fallback discovery
        alt = find_json_fallback(info)
        if alt and alt.exists():
            return collect_metrics(alt, info)
        print(f"Warning: JSON not found: {jpath}")
        return None

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_and_collect, idx, cmd) for idx, cmd in enumerate(commands_to_run)]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing Exploration"):
        result = f.result()
        if result:
            metrics_list.append(result)

print("\n==== Parallel Exploration Finished! ====")

# Save results
df = pd.DataFrame(metrics_list)
out_dir = Path.cwd() / "exploration"
out_dir.mkdir(parents=True, exist_ok=True)
output_excel_path = out_dir / "exploration_results_parallel.xlsx"
df.to_excel(output_excel_path, index=False)

print(f"Saved parallel exploration results to {output_excel_path}")
print(df.head())
print(f"Total configurations run: {len(df)}")