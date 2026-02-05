import torch
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import firwin
import scipy.signal as signal
from scipy.signal import find_peaks

import sys

sys.path.append("..")  # Add the parent directory
from data_generator import (
    convolve,
    matrix_init,
    error_metric,
    check_cast,
    select_dtypes,
)  # Import your FIR kernel


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
    return (data - min_val) / (
        max_val - min_val + 1e-10
    )  # Add small value to avoid division by zero


def plot_figure(input_data, title, name, fs=128):
    if fs is not None:
        # Convert sample indices to time (seconds)
        t = np.arange(len(input_data)) / fs
    else:
        t = np.arange(len(input_data))
    plt.figure(figsize=(10, 4))
    plt.plot(t, input_data, label="ECG Signal")
    plt.title(title)
    if fs is not None:
        plt.xlabel("Time (seconds)")  # Time axis in seconds
    else:
        plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f"figure/{name}.png")

    # Show the plot
    plt.show()


def find_local_max(signal, left, right):
    """Finds the index of the maximum value within a range."""
    return left + torch.argmax(signal[left : right + 1]).item()


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

    return R_loc, normalize_range(integrated_data)


def pan_tompkins(filter_configs, quantize, plot_flag, fs=128):
    
    if quantize:
        suffix = "_quantized"
    else:
        suffix = ""
        
    # Ensure the debug directory exists
    os.makedirs("debug", exist_ok=True)

    # 1️⃣ **Load ECG Signal from MATLAB File**
    # Create the 'fig' directory if it doesn't exist
    if not os.path.exists("figure"):
        os.makedirs("figure")

    # Example usage
    filename = "bartolini.txt"  # Change to your actual file path
    data = read_txt_file(filename)

    ecg_signal = data.flatten()  # Convert to 1D NumPy array

    # Remove first 500 samples (to match MATLAB code)
    ecg_signal = ecg_signal[500:]

    # Convert to Torch tensor
    ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)
    N = len(ecg_signal)
    # t = np.arange(N) / fs  # Time in seconds
    # Plot ECG Signal with Time Axis**
    if plot_flag:
        plot_figure(
            ecg_signal.numpy(), title="ECG row Signal", name="row_signal", fs=None
        )

    # 2️⃣ **Remove DC Component & Normalize Signal**
    ecg_signal_normalized = normalize_range(ecg_signal)
    # Plot ECG Signal with Time Axis**
    if plot_flag:
        plot_figure(
            ecg_signal_normalized.numpy(),
            title="ECG Signal after cancellation DC drift and normalization",
            name="dc_cancelation",
            fs=fs,
        )

    if plot_flag:
        # Plot the distribution of the normalized ECG signal
        plt.figure(figsize=(10, 4))
        plt.hist(
            ecg_signal_normalized.numpy(),
            bins=50,
            alpha=0.7,
            label="Normalized ECG Signal",
        )
        plt.title("Distribution of ECG Signal")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig("figure/ecg_signal_distribution.png")
        plt.show()

    # 3️⃣ **Low-Pass FIR Filtering (Mimicking MATLAB h_LP)**
    num_taps_LP = 13
    b = torch.tensor([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    a = [1, -2, 1]

    # Generate an impulse signal of length 13
    impulse = np.zeros(13)
    impulse[0] = 1  # First sample = 1, rest are zeros

    # Compute the filter's impulse response
    h_LP = signal.lfilter(b, a, impulse)
    h_LP = torch.tensor(h_LP, dtype=torch.float32)

    if quantize:
        datatypes = filter_configs["lowpass"]["datatypes"]
        mac_flag = filter_configs["lowpass"]["mac_flag"]
        vec_flag = filter_configs["lowpass"]["vec_flag"]
        cast_flag = filter_configs["lowpass"]["cast_flag"]
        cast_to = filter_configs["lowpass"]["cast_to"]
        mantissa_bits = filter_configs["lowpass"]["mantissa_bits"]
    else:
        datatypes = [torch.float32, torch.float32, torch.float32]
        mac_flag = "false"
        vec_flag = "false"
        cast_flag = "false"
        cast_to = "FP32"
        mantissa_bits = 2

    ecg_signal_normalized = matrix_init(
        ecg_signal_normalized, datatypes[0], mantissa_bits=mantissa_bits
    )
    h_LP = matrix_init(h_LP, datatypes[1], mantissa_bits=mantissa_bits)
    lowpass_ecg = convolve(
        ecg_signal_normalized,
        h_LP,
        datatypes,
        N - num_taps_LP,
        mac_flag=mac_flag,
        vec_flag=vec_flag,
        cast_flag=cast_flag,
        cast_to=cast_to,
        mantissa_bits=mantissa_bits,
    )

    # Plot ECG Signal with Time Axis**
    if plot_flag:
        plot_figure(
            lowpass_ecg, title="ECG Signal after LPF", name=f"LPF{suffix}", fs=fs
        )

    # Plot the distribution of the normalized ECG signal
    if plot_flag:
        plt.figure(figsize=(10, 4))
        plt.hist(lowpass_ecg.numpy(), bins=50, alpha=0.7, label="low pass ECG Signal")
        plt.title("Distribution of ECG Signal")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig("figure/ecg_signal_distribution.png")
        plt.show()
    # 4️⃣ **High-Pass FIR Filtering (Mimicking MATLAB h_HP)**
    num_taps_HP = 33
    b = torch.tensor(
        [
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            32,
            -32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
        ],
        dtype=torch.float32,
    )

    a = [1, -1]  # As in MATLAB
    impulse = np.zeros(num_taps_HP)
    impulse[0] = 1  # First sample = 1, rest are zeros
    h_HP = signal.lfilter(b, a, impulse)
    h_HP = torch.tensor(h_HP, dtype=torch.float32)

    if quantize:
        datatypes = filter_configs["highpass"]["datatypes"]
        mac_flag = filter_configs["highpass"]["mac_flag"]
        vec_flag = filter_configs["highpass"]["vec_flag"]
        cast_flag = filter_configs["highpass"]["cast_flag"]
        cast_to = filter_configs["highpass"]["cast_to"]
        mantissa_bits = filter_configs["highpass"]["mantissa_bits"]
    else:
        datatypes = datatypes = [torch.float32, torch.float32, torch.float32]
        mac_flag = "false"
        vec_flag = "false"
        cast_flag = "false"
        cast_to = "FP32"
        mantissa_bits = 2
    lowpass_ecg = matrix_init(lowpass_ecg, datatypes[0], mantissa_bits=mantissa_bits)
    h_HP = matrix_init(h_HP, datatypes[1], mantissa_bits=mantissa_bits)

    highpass_ecg = convolve(
        lowpass_ecg,
        h_HP,
        datatypes,
        len(lowpass_ecg) - num_taps_HP,
        mac_flag=mac_flag,
        vec_flag=vec_flag,
        cast_flag=cast_flag,
        cast_to=cast_to,
        mantissa_bits=mantissa_bits,
    )

    # Plot the distribution of the normalized ECG signal
    if plot_flag:
        plt.figure(figsize=(10, 4))
        plt.hist(highpass_ecg.numpy(), bins=50, alpha=0.7, label="high pass ECG Signal")
        plt.title("Distribution of ECG Signal")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig("figure/ecg_signal_distribution.png")
        plt.show()
    # Plot ECG Signal with Time Axis**
    if plot_flag:
        plot_figure(
            highpass_ecg.numpy(),
            title="ECG Signal after HPF",
            name=f"HPF{suffix}",
            fs=fs,
        )

    # 5️⃣ **Compute Derivative (Mimicking MATLAB Derivative)**
    derivative_kernel = torch.tensor(
        [-1 / 8, -2 / 8, 0, 2 / 8, 1 / 8], dtype=torch.float32
    )

    if quantize:
        datatypes = filter_configs["derivative"]["datatypes"]
        mac_flag = filter_configs["derivative"]["mac_flag"]
        vec_flag = filter_configs["derivative"]["vec_flag"]
        cast_flag = filter_configs["derivative"]["cast_flag"]
        cast_to = filter_configs["derivative"]["cast_to"]
        mantissa_bits = filter_configs["derivative"]["mantissa_bits"]

    else:
        datatypes = datatypes = [torch.float32, torch.float32, torch.float32]
        mac_flag = "false"
        vec_flag = "false"
        cast_flag = "false"
        cast_to = "FP32"
        mantissa_bits = 2

    highpass_ecg = matrix_init(highpass_ecg, datatypes[0], mantissa_bits=mantissa_bits)
    derivative_kernel = matrix_init(
        derivative_kernel, datatypes[1], mantissa_bits=mantissa_bits
    )
    derivative_ecg = convolve(
        highpass_ecg,
        derivative_kernel,
        datatypes,
        len(highpass_ecg) - len(derivative_kernel),
        mac_flag=mac_flag,
        vec_flag=vec_flag,
        cast_flag=cast_flag,
        cast_to=cast_to,
        mantissa_bits=mantissa_bits,
    )

    # Plot the distribution of the normalized ECG signal
    if plot_flag:
        plt.figure(figsize=(10, 4))
        plt.hist(
            derivative_ecg.numpy(), bins=50, alpha=0.7, label="derivative ECG Signal"
        )
        plt.title("Distribution of ECG Signal")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig("figure/ecg_signal_distribution.png")
        plt.show()
    # Plot ECG Signal with Time Axis**
    if plot_flag:
        plot_figure(
            derivative_ecg.numpy(),
            title="ECG Signal after Derivative",
            name=f"Derivative{suffix}",
            fs=fs,
        )

    # 6️⃣ **Squaring Function**
    if quantize:
        datatypes = filter_configs["squared"]["datatypes"]
    else:
        datatypes = datatypes = [torch.float32]
    derivative_ecg = matrix_init(
        derivative_ecg, datatypes[0], mantissa_bits=mantissa_bits
    )

    squared_ecg = derivative_ecg ** 2
    # Plot the distribution of the normalized ECG signal
    if plot_flag:
        plt.figure(figsize=(10, 4))
        plt.hist(squared_ecg.numpy(), bins=50, alpha=0.7, label="squared ECG Signal")
        plt.title("Distribution of ECG Signal")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig("figure/ecg_signal_distribution.png")
        plt.show()
    # Plot ECG Signal with Time Axis**
    if plot_flag:
        plot_figure(
            squared_ecg.numpy(),
            title="ECG Signal after Squaring",
            name=f"Squaring{suffix}",
            fs=fs,
        )

    # 7️⃣ **Moving Window Integration (MWI)**
    window_size = 31  # As in MATLAB
    mwi_kernel = torch.ones(window_size, dtype=torch.float32) / window_size

    if quantize:
        datatypes = filter_configs["mwi"]["datatypes"]
        mac_flag = filter_configs["mwi"]["mac_flag"]
        vec_flag = filter_configs["mwi"]["vec_flag"]
        cast_flag = filter_configs["mwi"]["cast_flag"]
        cast_to = filter_configs["mwi"]["cast_to"]
        mantissa_bits = filter_configs["mwi"]["mantissa_bits"]
    else:
        datatypes = [torch.float32, torch.float32, torch.float32]
        mac_flag = "false"
        vec_flag = "false"
        cast_flag = "false"
        cast_to = "FP32"
        mantissa_bits = 2

    mwi_kernel = matrix_init(mwi_kernel, datatypes[1], mantissa_bits=mantissa_bits)
    squared_ecg = matrix_init(squared_ecg, datatypes[0], mantissa_bits=mantissa_bits)
    mwi_ecg = convolve(
        squared_ecg,
        mwi_kernel,
        datatypes,
        len(squared_ecg) - window_size,
        mac_flag=mac_flag,
        vec_flag=vec_flag,
        cast_flag=cast_flag,
        cast_to=cast_to,
        mantissa_bits=mantissa_bits,
    )
    np.savetxt(
        f"debug/mwi_ecg{suffix}.txt",
        mwi_ecg,
        fmt="%.4f",
        header="Moving Window Integrated ECG Signal",
    )
    # Plot ECG Signal with Time Axis**
    if plot_flag:
        plot_figure(
            mwi_ecg.numpy(), title="ECG Signal after MWI", name=f"MWI{suffix}", fs=fs
        )

    # 8️⃣ **Adaptive Thresholding for R-Peak Detection**
    # Apply the selected QRS detection method
    VARIANT = 2  # Change this to 2 for Variant 2
    if VARIANT == 1:
        R_loc, integrated_data = detect_qrs_variant1(mwi_ecg, ecg_signal, fs)
    else:
        R_loc, integrated_data = detect_qrs_variant2(mwi_ecg, fs)
    np.savetxt(
        f"debug/R_loc{datatypes[0]}.txt",
        R_loc,
        fmt="%d",
        header="Detected R-Peaks (Sample Indices)",
    )
    np.savetxt(
        f"debug/integrated_data{datatypes[0]}.txt",
        integrated_data,
        fmt="%.4f",
        header="Integrated ECG Signal",
    )
    # 📊 **Plot Results**
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(ecg_signal.numpy(), label="Original ECG")
    plt.title("Original ECG Signal")

    plt.subplot(3, 1, 2)
    plt.plot(integrated_data.numpy(), label="Moving Window Integrated Signal")
    plt.title("Moving Window Integration")

    plt.subplot(3, 1, 3)
    plt.plot(integrated_data.numpy(), label="Integrated ECG", alpha=0.6)
    plt.scatter(
        R_loc, integrated_data.numpy()[R_loc], color="red", label="Detected R Peaks"
    )
    plt.title("Detected QRS Complexes")
    plt.legend()
    # Save the final plot
    plt.savefig(f"figure/final_results{suffix}.png")
    plt.show()

    return R_loc, integrated_data, ecg_signal


def checks(R_loc, mwi_ecg, ecg_signal, fs, name="fp32"):
    # Compute RR intervals (in samples)
    RR_intervals_samples = np.diff(R_loc)

    # Convert RR intervals to seconds for BPM calculation
    RR_intervals_sec = RR_intervals_samples / fs

    # Compute Heart Rate (BPM)
    HR = 60 / RR_intervals_sec

    # Save RR intervals in samples and seconds
    np.savetxt(
        f"debug/RR_intervals_samples_{name}.txt",
        RR_intervals_samples,
        fmt="%d",
        header="RR Intervals (samples)",
    )
    np.savetxt(
        f"debug/RR_intervals_sec_{name}.txt",
        RR_intervals_sec,
        fmt="%.4f",
        header="RR Intervals (seconds)",
    )
    np.savetxt(f"debug/Heart_Rate_BPM_{name}.txt", HR, fmt="%.2f", header="Heart Rate (BPM)")

    # Convert MWI signal to numpy
    mwi_ecg_np = mwi_ecg.numpy()

    # Detect R-peaks using `find_peaks()` on the **processed** MWI signal
    ground_truth_peaks, _ = find_peaks(
        mwi_ecg_np, height=0.2 * np.max(mwi_ecg_np), distance=fs // 2, prominence=0.1
    )

    # Compute Ground Truth RR intervals (in samples)
    ground_truth_intervals = np.diff(ground_truth_peaks)

    # Save Ground Truth RR intervals to a file
    np.savetxt(
        "debug/Ground_Truth_Intervals.txt",
        ground_truth_intervals,
        fmt="%d",
        header="Ground Truth intervals",
    )

    # Convert peak indices to seconds
    ground_truth_sec = ground_truth_peaks / fs
    detected_sec = np.array(R_loc) / fs

    # Set tolerance window (150 ms = 0.15 seconds)
    tolerance = 0.15

    TP = 0
    FP = 0
    FN = 0

    # Convert lists to NumPy arrays for fast operations
    ground_truth_sec = np.array(ground_truth_sec)
    detected_sec = np.array(detected_sec)

    # Initialize matching arrays
    matched_gt = np.zeros(
        len(ground_truth_sec), dtype=bool
    )  # Tracks matched ground truth peaks
    matched_detected = np.zeros(
        len(detected_sec), dtype=bool
    )  # Tracks matched detected peaks

    np.savetxt(
        f"debug/detected_sec_debug.txt",
        detected_sec,
        fmt="%.4f",
        header="Detected R-Peaks (seconds)",
    )
    np.savetxt(
        f"debug/ground_truth_sec_debug.txt",
        ground_truth_sec,
        fmt="%.4f",
        header="Ground Truth Peaks (seconds)",
    )
    # Iterate over detected peaks and check if they match ground truth
    for i, det in enumerate(detected_sec):
        print(f"Detected Peak: {det:.2f} sec")
        # Find the closest ground truth peak within the tolerance window
        match_idx = np.where(np.abs(ground_truth_sec - det) <= tolerance)[0]
        print(f"Matched Ground Truth Peaks: {ground_truth_sec[match_idx]} sec")
        if len(match_idx) > 0:  # If a match is found
            TP += 1
            matched_detected[i] = True  # Mark detected peak as matched
            matched_gt[
                match_idx[0]
            ] = True  # Mark ground truth peak as matched (only first match)

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

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(mwi_ecg_np)) / fs, mwi_ecg_np, label="ECG Signal")
    plt.scatter(detected_sec, mwi_ecg_np[R_loc], color="blue", label="Your R Peaks")
    plt.scatter(
        ground_truth_sec,
        mwi_ecg_np[ground_truth_peaks],
        color="red",
        marker="x",
        label="Ground Truth Peaks",
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Comparison of Your R-Peaks vs Ground Truth (find_peaks)")
    plt.legend()
    plt.grid()
    plt.savefig(f"figure/peaks_comparison_{name}.png")
    plt.show()


def compute_accuracy(rr_intervals_gt, rr_intervals_pred):
    """
    Computes the RMSD and Accuracy of RR intervals compared to the ground truth.

    Args:
        rr_intervals_gt (np.array): Ground truth RR intervals (from find_peaks).
        rr_intervals_pred (np.array): RR intervals detected by your method.

    Returns:
        rmsd (float): Root Mean Square Deviation.
        accuracy (float): Accuracy score based on RMSD.
    """
    # Ensure both arrays have the same length
    n = min(len(rr_intervals_gt), len(rr_intervals_pred))

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

    return rmsd, accuracy

def main():
    
    
    plot_flag = False # Set to True to plot figures

    
    fs = 128  # Sampling frequency (as in MATLAB)
    
    # First, run the FP32 version of the algorithm
    quantize = False

    R_loc_fp32, mwi_ecg_fp32, ecg_signal_fp32 = pan_tompkins(
        None, quantize, plot_flag, fs=fs
    )

    checks(R_loc_fp32, mwi_ecg_fp32, ecg_signal_fp32, fs, name="fp32")

    # Now, run the Quantized version of the algorithm
    quantize = True
    # Define datatype settings for each filter
    filter_datatypes = {
        "lowpass": ["FP8_CUSTOM", "FP8_CUSTOM", "FP8_CUSTOM"],
        "highpass": ["FP8_CUSTOM", "FP8_CUSTOM", "FP8_CUSTOM"],
        "derivative": ["FP8_CUSTOM", "FP8_CUSTOM", "FP8_CUSTOM"],
        "squared": ["FP8_CUSTOM"], 
        "mwi": ["FP8_CUSTOM", "FP8_CUSTOM", "FP8_CUSTOM"]
    }

    # Define mac_flag settings for each filter
    filter_mac_flags = {
        "lowpass": "false",
        "highpass": "false",
        "derivative": "false",
        "squared": "false",
        "mwi": "false",
    }

    # Define vec_flag settings for each filter
    filter_vec_flags = {
        "lowpass": "false",
        "highpass": "false",
        "derivative": "false",
        "squared": "false",
        "mwi": "false",
    }

    filter_mantissa_bits = {
        "lowpass": 2,
        "highpass": 2,
        "derivative": 2,
        "squared": 2,
        "mwi": 2,
    }

    # Dictionary to store the configurations for each filter
    filter_configs = {}

    # Assign values manually using dictionary lookups
    for filter_name in filter_datatypes:
        bits = filter_datatypes[filter_name]
        datatypes = select_dtypes(bits, len(bits))
        cast_flag = check_cast(datatypes[0 : len(datatypes) - 1])
        cast_to = "FP16ALT" if "FP16ALT" in bits else "FP16"

        # Assign manually set mac_flag and vec_flag values
        mac_flag = filter_mac_flags[filter_name]
        vec_flag = filter_vec_flags[filter_name]
        mantissa_bits = filter_mantissa_bits[filter_name]

        # Store everything in a dictionary
        filter_configs[filter_name] = {
            "datatypes": datatypes,
            "cast_flag": cast_flag,
            "cast_to": cast_to,
            "mac_flag": mac_flag,
            "vec_flag": vec_flag,
            "mantissa_bits": mantissa_bits,
        }

    
    R_loc, mwi_ecg, ecg_signal = pan_tompkins(filter_configs, quantize, plot_flag, fs=fs)

    checks(R_loc, mwi_ecg_fp32, ecg_signal, fs, name="quantized")


    # Example Usage

    rr_intervals_gt = np.loadtxt(
        "debug/Ground_Truth_Intervals.txt"
    )  # Replace with ground truth file
    rr_intervals_pred = np.loadtxt(
        "debug/RR_intervals_samples_quantized.txt"
    )  # Your detected RR intervals

    rmsd, accuracy = compute_accuracy(rr_intervals_gt, rr_intervals_pred)

    print(f"RMSD: {rmsd:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    print(f"Max RR Interval: {np.max(rr_intervals_gt)}")
    print(f"Min RR Interval: {np.min(rr_intervals_gt)}")
    print(f"RR Interval Range: {np.max(rr_intervals_gt) - np.min(rr_intervals_gt)}")


    print("Ground Truth RR Intervals:", rr_intervals_gt[:10])
    print("Predicted RR Intervals:", rr_intervals_pred[:10])


    # Plot Absolute Errors
    plt.plot(np.abs(rr_intervals_gt - rr_intervals_pred), marker="o")
    plt.title("Absolute Errors in RR Intervals")
    plt.xlabel("Interval Index")
    plt.ylabel("Error (samples)")
    plt.grid()
    plt.savefig("figure/absolute_errors.png")
    plt.show()


if __name__ == "__main__":
    main()