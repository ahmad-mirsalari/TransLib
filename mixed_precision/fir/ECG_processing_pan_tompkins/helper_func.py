# Description: Utility functions for ECG processing.
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import find_peaks
from scipy import signal

# Description: Utility functions for ECG processing.
from typing import Union, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import find_peaks
from scipy import signal

# DKind import (matches your FIR/DWT helpers)
from utils.helper_functions import DKind, fp8_quantizer, _quant_mbits_tensor_like, _final_round_tensor
# ----------------------------
# Small dtype adapter
# ----------------------------
def _as_torch_dtype(dt_or_kind: Union[DKind, torch.dtype, None]) -> torch.dtype:
    """Accept DKind or torch.dtype and return a torch.dtype (default fp32)."""
    if isinstance(dt_or_kind, DKind):
        td = dt_or_kind.to_torch()
        return torch.float32 if td is None else td
    if isinstance(dt_or_kind, torch.dtype):
        return dt_or_kind
    return torch.float32

def finalize_out_dtype(t: torch.Tensor, d_out: DKind = DKind.FP32, mantissa_bits: int = 2) -> torch.Tensor:
    """
    Convert/quantize intermediate to declared output dtype with correct rounding.
    """
    if d_out is DKind.FP8_CUSTOM:
        return fp8_quantizer(t, _quant_mbits_tensor_like(t, mantissa_bits))
    if d_out in (DKind.FP16, DKind.BF16, DKind.FP32):
        return _final_round_tensor(t, d_out)
    return t
# ----------------------------
# Filter coefficients
# ----------------------------

def lowpass_coef(NC_Lo):
        b = torch.tensor([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
        a = [1, -2, 1] 

        # Generate an impulse signal of length 13
        impulse = np.zeros(NC_Lo)
        impulse[0] = 1  # First sample = 1, rest are zeros

        # Compute the filter's impulse response
        h_LP = signal.lfilter(b, a, impulse)
        h_LP = torch.tensor(h_LP, dtype=torch.float32)

        return h_LP

def highpass_coef(NC_Hi):
    b = torch.tensor([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            32, -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)

    a = [1, -1]  # As in MATLAB
    impulse = np.zeros(NC_Hi)
    impulse[0] = 1  # First sample = 1, rest are zeros
    h_HP = signal.lfilter(b, a, impulse)
    h_HP = torch.tensor(h_HP, dtype=torch.float32)

    return h_HP

# ----------------------------
# Minimal header writer (kept)
# ----------------------------
def save_data_into_hfile(
    length: int,
    order: int,
    res: torch.Tensor,
    filter_conv: torch.Tensor,
    input_conv: torch.Tensor,
) -> None:
    """
    Saves data into a header file.

    Args:
        length (int): The length of the data.
        order (int): The order of the filter.
        res (torch.Tensor): The result matrix to be saved.
        filter_conv (torch.Tensor): The filter convolution matrix to be saved.
        input_conv (torch.Tensor): The input convolution matrix to be saved.

    Returns:
        None
    """
    # Generate header file
    with open("data.h", "w", encoding="utf-8") as f:
        f.write('#include "config.h"\n\n')
        f.write(f"#define LENGTH {length}\n")
        f.write(f"#define ORDER {order}\n\n")

# ----------------------------
# I/O helpers
# ----------------------------
def read_txt_file(filename):
    """
    Reads numerical data from a text file and returns it as a NumPy array.
    """
    try:
        data = np.loadtxt(filename, dtype=int)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# ----------------------------
# Streaming preproc (DC cancel + normalize)
# ----------------------------
def dc_cancellation_and_normalize_python(
    sample_idx: int,
    ecg_signal: torch.Tensor,
    max_value: torch.Tensor,
    signal_data: List[float],
    input_dt: Union[DKind, torch.dtype] = torch.float32,
    mantissa_bits: int = 2,
) -> float:
    """
    Simulate real-time DC component cancellation and normalization.
    Returns a Python float (keeps your ring buffers purely numeric).

    input_dt can be either DKind or torch.dtype; it’s used only to mirror
    precision behavior locally (result is cast then returned as float).
    """

    if 1 <= sample_idx < (len(ecg_signal) - 1):
        # Use scalar math to avoid tensor-in-list issues
        diff = float(ecg_signal[sample_idx + 1].item() - ecg_signal[sample_idx].item())
        diff = torch.as_tensor(diff, dtype=torch.float32)
        diff = finalize_out_dtype(diff, d_out=input_dt, mantissa_bits=mantissa_bits)
        prev = float(signal_data[-1])
        prev = torch.as_tensor(prev, dtype=torch.float32)
        prev = finalize_out_dtype(prev, d_out=input_dt, mantissa_bits=mantissa_bits)
        mul = float(0.995 * prev.item())
        mul = torch.as_tensor(mul, dtype=torch.float32)
        mul = finalize_out_dtype(mul, d_out=input_dt, mantissa_bits=mantissa_bits)
        val = diff + mul
        val = finalize_out_dtype(val, d_out=input_dt, mantissa_bits=mantissa_bits)
        mv = float(max_value.item() if isinstance(max_value, torch.Tensor) else max_value)
        mv = torch.as_tensor(mv, dtype=torch.float32)
        mv = finalize_out_dtype(mv, d_out=input_dt, mantissa_bits=mantissa_bits)
        if mv != 0.0:
            val = val / mv
            val = finalize_out_dtype(val, d_out=input_dt, mantissa_bits=mantissa_bits)

        # emulate dtype locally, then return as float to keep buffers simple
        tval = torch.as_tensor(val, dtype=torch.float32)
        tval = finalize_out_dtype(tval, d_out=input_dt, mantissa_bits=mantissa_bits)
        return float(tval.item())
    else:
        return 0.0


# ----------------------------
# Misc math/plot helpers (unchanged behavior)
# ----------------------------
def find_local_max(signal, left, right):
    """Finds the index of the maximum value within a range."""
    return left + torch.argmax(signal[left:right+1]).item()


def normalize_range(data):
    """
    Normalizes data to the range [0,1] (like MATLAB's `normalize(x, 'range')`).
    
    Args:
        data (torch.Tensor or np.ndarray): Input signal.
    
    Returns:
        torch.Tensor: Normalized signal in [0,1] range.
    """
    min_val = torch.min(data)
    max_val = torch.max(data)
    return (data - min_val) / (max_val - min_val + 1e-10)  # Add small value to avoid division by zero

def plot_figure(input_data, title, name, fs=128):
    if fs is not None:
        # Convert sample indices to time (seconds)
        t = np.arange(len(input_data)) / fs
    else:
        t = np.arange(len(input_data))
    plt.figure(figsize=(10, 4))
    plt.plot(t,input_data, label="ECG Signal")
    plt.title(title)
    if fs is not None:
        plt.xlabel('Time (seconds)')  # Time axis in seconds
    else:
        plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f'figure/{name}.png')

    # Show the plot
    plt.show()


def detect_qrs_variant1(integrated_data, window_data, fs):
    """Implements Variant 1 of QRS Detection (Thresholding Approach)."""
    WINDOW_SIZE = len(integrated_data)

    # Normalize integrated signal
    max_val = torch.max(torch.abs(integrated_data))
    integrated_data = normalize_range(integrated_data)

    # Compute mean threshold value
    mean_val = torch.mean(integrated_data)
    max_val = torch.max(torch.abs(integrated_data))
    threshold = mean_val * max_val  # Thresholding logic

    # Find regions above the threshold
    left = []
    right = []
    prev_check = integrated_data[0] > threshold

    if prev_check:
        left.append(0)

    for i in range(1, WINDOW_SIZE):
        current_check = integrated_data[i] > threshold
        if current_check and not prev_check:
            left.append(i)
        if not current_check and prev_check:
            right.append(i - 1)
        prev_check = current_check

    if prev_check:
        right.append(WINDOW_SIZE - 1)

    # Find R-peaks in detected regions
    R_loc = [find_local_max(window_data, l, r) for l, r in zip(left, right)]

    return R_loc, integrated_data

def detect_qrs_variant2(integrated_data, fs):
    """Implements Variant 2 of QRS Detection (Search-Back Logic)."""
    WINDOW_SIZE = len(integrated_data)

    # Initialize adaptive thresholding variables
    peaki = spki = npki = 0.0
    threshold1 = threshold2 = 0.0
    previous_peak = 0
    searchback = searchback_end = 0
    peak_counter = 0

    min_rr_width = int(0.2 * fs)  # Minimum RR-interval in samples
    max_rr_width = int(2.0 * fs)  # Maximum RR-interval in samples

    R_loc = []

    for i in range(2, WINDOW_SIZE - 2):
        if i - previous_peak > max_rr_width and i - searchback_end > max_rr_width:
            searchback = 1
            searchback_end = i
            i = previous_peak + 1
            continue

        if searchback and i == searchback_end:
            searchback = 0
            continue

        # Check for QRS detection
        peaki = integrated_data[i]
        if peaki < integrated_data[i - 2] or peaki <= integrated_data[i + 2]:
            continue

        is_qrs = False
        if searchback and (peaki > threshold2):
            spki = 0.750 * spki + 0.250 * peaki
            is_qrs = True
        elif peaki > threshold1:
            spki = 0.875 * spki + 0.125 * peaki
            is_qrs = True

        if is_qrs:
            if peak_counter == 0 or i - previous_peak >= min_rr_width:
                R_loc.append(i)
                peak_counter += 1
            elif integrated_data[previous_peak] < peaki:
                R_loc[-1] = i

            previous_peak = i
        else:
            npki = 0.875 * npki + 0.125 * peaki

        # Update thresholds
        threshold1 = npki + 0.25 * (spki - npki)
        threshold2 = 0.5 * threshold1

    return R_loc, integrated_data#normalize_range(integrated_data)


def non_streaming_peaks(mwi_ecg, fs, quantize, ecg_signal, plot=True):
    # Apply the selected QRS detection method
    VARIANT = 2  # Change this to 2 for Variant 2
    if VARIANT == 1:
        R_loc, integrated_data = detect_qrs_variant1(mwi_ecg, ecg_signal, fs)
    else:
        R_loc, integrated_data = detect_qrs_variant2(mwi_ecg, fs)

    if quantize:
        suffix="_quantized"
    else:
        suffix=""
    np.savetxt(f"debug/R_loc{suffix}.txt", R_loc, fmt="%d", header="Detected R-Peaks (Sample Indices)")
    np.savetxt(f"debug/integrated_data{suffix}.txt", integrated_data, fmt="%.4f", header="Integrated ECG Signal")
    
    if plot:
        # 📊 **Plot Results**
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(ecg_signal.numpy(), label="Original ECG")
        plt.title("Original ECG Signal")

        plt.subplot(3, 1, 2)
        plt.plot(integrated_data, label="Moving Window Integrated Signal")
        plt.title("Moving Window Integration")

        plt.subplot(3, 1, 3)
        plt.plot(integrated_data, label="Integrated ECG", alpha=0.6)
        plt.scatter(R_loc, integrated_data[R_loc], color='red', label="Detected R Peaks")
        plt.title("Detected QRS Complexes")
        plt.legend()
        # Save the final plot
        plt.savefig(f'figure/final_results{suffix}.png')
        plt.show()

    return R_loc, integrated_data


def checks(R_loc, mwi_ecg, fs, name="fp32", plot=True):
    # Compute RR intervals (in samples)
    RR_intervals_samples = np.diff(R_loc)  

    # Convert RR intervals to seconds for BPM calculation
    RR_intervals_sec = RR_intervals_samples / fs  

    # Compute Heart Rate (BPM)
    HR = 60 / RR_intervals_sec  

    # Save RR intervals in samples and seconds
    np.savetxt(f"debug/RR_intervals_samples_{name}.txt", RR_intervals_samples, fmt="%d", header="RR Intervals (samples)")
    np.savetxt(f"debug/RR_intervals_sec_{name}.txt", RR_intervals_sec, fmt="%.4f", header="RR Intervals (seconds)")
    np.savetxt(f"debug/Heart_Rate_BPM_{name}.txt", HR, fmt="%.2f", header="Heart Rate (BPM)")

    # Convert MWI signal to numpy
    mwi_ecg_np = mwi_ecg

    # Detect R-peaks using `find_peaks()` on the **processed** MWI signal
    ground_truth_peaks, _ = find_peaks(mwi_ecg_np, height=0.2 * np.max(mwi_ecg_np), distance=fs//2, prominence=0.1)

    # Compute Ground Truth RR intervals (in samples)
    ground_truth_intervals = np.diff(ground_truth_peaks)

    # Save Ground Truth RR intervals to a file
    np.savetxt("debug/Ground_Truth_Intervals.txt", ground_truth_intervals, fmt="%d", header="Ground Truth intervals")

    # Convert peak indices to seconds
    ground_truth_sec = ground_truth_peaks / fs
    detected_sec = np.array(R_loc) / fs

    # Set tolerance window (150 ms = 0.15 seconds)
    tolerance = 0.2  

    TP = 0
    FP = 0
    FN = 0

    # Convert lists to NumPy arrays for fast operations
    ground_truth_sec = np.array(ground_truth_sec)
    detected_sec = np.array(detected_sec)

    # Initialize matching arrays
    matched_gt = np.zeros(len(ground_truth_sec), dtype=bool)  # Tracks matched ground truth peaks
    matched_detected = np.zeros(len(detected_sec), dtype=bool)  # Tracks matched detected peaks

    np.savetxt(f"debug/detected_sec_{name}.txt", detected_sec, fmt="%.4f", header="Detected R-Peaks (seconds)")
    np.savetxt(f"debug/detected_sec_ground_truth.txt", ground_truth_sec, fmt="%.4f", header="Ground Truth Peaks (seconds)")
    # Iterate over detected peaks and check if they match ground truth
    for i, det in enumerate(detected_sec):
        # Find the closest ground truth peak within the tolerance window
        match_idx = np.where(np.abs(ground_truth_sec - det) <= tolerance)[0]
        with open(f"debug/peak_matching_{name}.txt", "a") as f:
            f.write(f"Detected Peak: {det:.2f} sec\n")
            f.write(f"Matched Ground Truth Peaks: {ground_truth_sec[match_idx]} sec\n")
        if len(match_idx) > 0 :  # If a match is found
            TP += 1
            matched_detected[i] = True  # Mark detected peak as matched
            matched_gt[match_idx[0]] = True  # Mark ground truth peak as matched (only first match)
    
    print(f"Peak matching information saved in debug/peak_matching_{name}.txt")
    # Count False Negatives (ground truth peaks that were NOT matched)
    FN = np.sum(~matched_gt)

    # Count False Positives (detected peaks that were NOT matched)
    FP = np.sum(~matched_detected)

    if TP == 0:
        return 0.0, 100.0  # If no true positives, F1 is zero, and error rate is 100%.

    # Compute F1₁ Score
    F1_1 = TP / (TP + 0.5 * (FP + FN))

    # Compute ErrRate%
    ErrRate = (1 - F1_1) * 100

    print(f"F1₁ Score: {F1_1:.2f}")
    print(f"Error Rate: {ErrRate:.2f}%")
    print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")

    if plot:
        # Visualization
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(mwi_ecg_np)) / fs, mwi_ecg_np, label="ECG Signal")
        plt.scatter(detected_sec, mwi_ecg_np[R_loc], color="blue", label="Your R Peaks")
        plt.scatter(ground_truth_sec, mwi_ecg_np[ground_truth_peaks], color="red", marker="x", label="Ground Truth Peaks")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Comparison of Your R-Peaks vs Ground Truth (find_peaks)")
        plt.legend()
        plt.grid()
        plt.savefig(f'figure/peaks_comparison_{name}.png')
        plt.show()



def compute_accuracy(name="fp32"):
    """
    Computes the RMSD and Accuracy of RR intervals compared to the ground truth.

    Args:
        name (str): Name of the file to load predicted RR intervals.

    Returns:
        rmsd (float): Root Mean Square Deviation.
        accuracy (float): Accuracy score based on RMSD.
    """
    
    # Load Ground Truth RR intervals
    rr_intervals_gt = np.loadtxt("debug/Ground_Truth_Intervals.txt", dtype=int)

    # Load Predicted RR intervals
    rr_intervals_pred = np.loadtxt(f"debug/RR_intervals_samples_{name}.txt", dtype=int)
    # Ensure both arrays have the same length
    n = min(len(rr_intervals_gt), len(rr_intervals_pred))

    # print(f"Length of GT: {len(rr_intervals_gt)}")
    # print(f"Length of Pred: {len(rr_intervals_pred)}")

    rr_intervals_gt = rr_intervals_gt[:n]
    rr_intervals_pred = rr_intervals_pred[:n]

    total_diff = 0
    for i in range(n):
        total_diff += (rr_intervals_gt[i] - rr_intervals_pred[i]) ** 2

        
    # Compute RMSD
    rmsd = np.sqrt(np.mean((rr_intervals_gt - rr_intervals_pred) ** 2))

    # Compute accuracy
    x_max = np.max(rr_intervals_gt)
    x_min = np.min(rr_intervals_gt)

    accuracy = 100 - (rmsd / (x_max - x_min))
    print(f"RMSD: {rmsd:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")
    return rmsd, accuracy
