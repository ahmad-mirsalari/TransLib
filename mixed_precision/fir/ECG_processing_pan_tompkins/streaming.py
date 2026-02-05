import logging
import torch
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import firwin
import scipy.signal as signal
from scipy.signal import find_peaks
from collections import deque
from pathlib import Path
import sys
sys.path.append("..")  # Add the parent directory
from data_generator import convolve, write_matrix  # Import your FIR kernel
from helper_func import (
    read_txt_file,
    dc_cancellation_and_normalize_python,
    compute_accuracy,
    checks,
    lowpass_coef,
    highpass_coef,
    finalize_out_dtype,
)


# repo root: translib_jr (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- cfg/dtype helpers (same as FIR/DWT style) ----
from utils.helper_functions import (
    DKind,
    dkind_name,
    select_dtypes,
    check_cast,
    matrix_init_like as matrix_init,
)

# ----------------------------
# Processor
# ----------------------------

class PanTompkinsProcessor:
    def __init__(self, filter_configs, quantize: bool, fs: int = 128):
        self.filter_configs = filter_configs
        self.quantize = quantize
        self.FS = fs  # Sampling frequency (Hz)
        # Define filter coefficients (example values)
        self.NC_Lo = 13
        self.NC_Hi = 33
        self.NC_Der = 5
        self.NC_Int = 31

        # ---------------- GLOBAL VARIABLES ----------------
        self.BUFFER_SIZE = 205  # Similar to C code
        self.N_AVG = 8
        self.MAX_SAMPLES = 65536

        self.min_rr_width = int(0.2 * self.FS)  # Minimum RR interval (200 ms in samples)
        self.max_rr_width = self.BUFFER_SIZE  # Maximum RR interval
        
        self.suffix = "_quantized" if quantize else ""


        # Initialize Buffers
        self.init_buffers()
        self.ensure_dirs()


        # --------------------------------------------------
    def init_buffers(self):
        """Initializes signal buffers."""
        
        self.signal_data = [0.0] * self.NC_Lo
        self.lowpass_data = [0.0] * self.NC_Hi
        self.highpass_data = [0.0] * self.NC_Der
        self.derivative_data = [0.0] * self.NC_Int
        self.squaring_data = [0.0] * self.NC_Int
        self.integrated_data = [0.0] * self.BUFFER_SIZE

        # Current indices for each filter
        self.current_signal = self.current_lo = self.current_hi = self.current_der = self.current_int = 0
    

        self.peaki = self.spki = self.npki = self.threshold1 = self.threshold2 = 0.0
        self.previous_peak = 0
        self.peak_counter = 0
        self.searchback_end = 0
        self.peak_processing = 0
        self.heart_rate = 0.0
        self.R_loc = [0] * self.N_AVG
        self.R_loc_all = []

    def ensure_dirs(self):
            for d in ("figure", "debug"):
                os.makedirs(d, exist_ok=True)

    # ----------------------------
    # Per-filter config access
    # ----------------------------
    
    def get_config(self, filter_name):
        """
        Returns a tuple:
        (datatypes[List[DKind]], mac_flag[bool], vec_flag[bool],
        cast_flag[bool], cast_to[str|None], mantissa_bits[List[int]])
        """
        if self.quantize:
            datatypes    = self.filter_configs[filter_name]["datatypes"]       # List[DKind]
            mac_flag     = bool(self.filter_configs[filter_name]["mac_flag"])
            vec_flag     = bool(self.filter_configs[filter_name]["vec_flag"])
            cast_flag    = bool(self.filter_configs[filter_name]["cast_flag"])
            cast_to      = self.filter_configs[filter_name]["cast_to"]         # e.g., "FP16", "FP16ALT" or None
            mantissa_bits = self.filter_configs[filter_name]["mantissa_bits"]  # [in, filt, out] or per-filter list
        else:
            # FP32 everywhere (no quantization)
            datatypes     = [DKind.FP32, DKind.FP32, DKind.FP32]
            mac_flag      = False
            vec_flag      = False
            cast_flag     = False
            cast_to       = None
            mantissa_bits = [0, 0, 0]
        return datatypes, mac_flag, vec_flag, cast_flag, cast_to, mantissa_bits
    
    # ----------------------------
    # Circular-buffer helper (streaming alignment)
    # ----------------------------
    
    def update_buffer(self, buffer, sample, max_len, initialized, threshold):
        """
        Updates a buffer by shifting its elements if the sample count exceeds the threshold.
        This replicates the behavior in the C code.
        """
        if initialized or sample >= threshold:
            buffer[:-1] = buffer[1:]  # Left shift: remove from left, add to right
            return max_len - 1
        else:
            return sample
    # ----------------------------
    # Filters
    # ----------------------------
    def lowpass_filter(self, signal_tensor, current_signal):

        # # Simulate initialization after the first BUFFER_SIZE samples
        # 3️⃣ **Low-Pass FIR Filtering (Mimicking MATLAB h_LP)**

        h_LP = lowpass_coef(self.NC_Lo)

        datatypes, mac_flag, vec_flag, cast_flag, cast_to, mantissa_bits = self.get_config("lowpass")
        
        
        signal_tensor = matrix_init(signal_tensor, datatypes[0],mantissa_bits=mantissa_bits[0])
        h_LP = matrix_init(h_LP, datatypes[1],mantissa_bits=mantissa_bits[1])

        if len(signal_tensor) - self.NC_Lo == 0:
            len_lp = 1
        else:
            len_lp = len(signal_tensor) - self.NC_Lo
        
        
        return convolve(
            signal_tensor,
            h_LP[:current_signal+1],
            datatypes,
            len_lp,
            mac_flag=mac_flag,
            vec_flag=vec_flag,
            cast_flag=cast_flag,
            cast_to=cast_to,
            mantissa_bits=mantissa_bits[2],
            streaming=True
        )


    def highpass_filter(self, lowpass_tensor, current_lo):

        h_HP = highpass_coef(self.NC_Hi)
        datatypes, mac_flag, vec_flag, cast_flag, cast_to, mantissa_bits = self.get_config("highpass")

        lowpass_tensor = matrix_init(lowpass_tensor, datatypes[0],mantissa_bits=mantissa_bits[0])
        h_HP = matrix_init(h_HP, datatypes[1],mantissa_bits=mantissa_bits[1])
            
        len_lp = 1 if len(lowpass_tensor) - self.NC_Hi == 0 else len(lowpass_tensor) - self.NC_Hi
        
        
        return  convolve(lowpass_tensor,
                        h_HP[:current_lo+1],
                        datatypes,
                        len_lp,
                        mac_flag=mac_flag,
                        vec_flag=vec_flag,
                        cast_flag=cast_flag,
                        cast_to=cast_to,
                        mantissa_bits=mantissa_bits[2],
                        streaming=True
                    )


    def derivative_filter(self, highpass_tensor, current_hi):
        derivative_kernel = torch.tensor([-1/8, -2/8, 0, 2/8, 1/8], dtype=torch.float32)

        datatypes, mac_flag, vec_flag, cast_flag, cast_to, mantissa_bits = self.get_config("derivative")
        
        highpass_tensor = matrix_init(highpass_tensor, datatypes[0],mantissa_bits=mantissa_bits[0])
        derivative_kernel = matrix_init(derivative_kernel, datatypes[1],mantissa_bits=mantissa_bits[1])
            
        len_hp = 1 if len(highpass_tensor) - self.NC_Der == 0 else len(highpass_tensor) - self.NC_Der
        
        
        return convolve(highpass_tensor,
                        derivative_kernel[:current_hi+1],
                        datatypes,
                        len_hp,
                        mac_flag=mac_flag,
                        vec_flag=vec_flag,
                        cast_flag=cast_flag,
                        cast_to=cast_to,
                        mantissa_bits=mantissa_bits[2],
                        streaming=True
                    )

    def mwi_filter(self, derivative_tensor, current_der):
        window_size = 31  # As in MATLAB
        mwi_kernel = torch.ones(window_size, dtype=torch.float32) / window_size
        datatypes, mac_flag, vec_flag, cast_flag, cast_to, mantissa_bits = self.get_config("mwi")
        
        mwi_kernel = matrix_init(mwi_kernel, datatypes[1],mantissa_bits=mantissa_bits[1])
        derivative_tensor = matrix_init(derivative_tensor, datatypes[0],mantissa_bits=mantissa_bits[0])
            
        len_mwp = 1 if len(derivative_tensor) - window_size == 0 else len(derivative_tensor) - window_size


        return convolve(derivative_tensor,
                        mwi_kernel[:current_der+1],
                        datatypes,
                        len_mwp,
                        mac_flag=mac_flag,
                        vec_flag=vec_flag,
                        cast_flag=cast_flag,
                        cast_to=cast_to,
                        mantissa_bits=mantissa_bits[2],
                        streaming=True
                    )

    # ----------------------------
    # I/O & streaming wrapper
    # ----------------------------
    def load_ecg_signal(self, filename):
        """Loads and processes ECG signal from a text file."""
        data = read_txt_file(filename)
        ecg_signal = torch.tensor(data.flatten()[500:], dtype=torch.float32)
        max_value = torch.max(ecg_signal)
        return ecg_signal, max_value    


    # ----------------------------
    # R-peak detection (unchanged logic)
    # ----------------------------
    def detect_r_peaks(
        self, sample, current_int, last_index = False, qrs_dtype: DKind = DKind.FP32, qrs_mantissa_bits: int = 2,
        hr_dtype: DKind = DKind.FP32, hr_mantissa_bits: int = 2):
        """
        Implements adaptive thresholding with search-back logic as in the original C implementation.

        Args:
            sample (int): Current sample index
            current_int (int): Current index in the integrated data
            last_index (bool): Flag to indicate if it's the last index
            qrs_dtype (DKind): Data type for QRS detection
            qrs_mantissa_bits (int): Mantissa bits for QRS detection

        Returns:
            None
        """
        # Compute time since last detected peak
        dist_previous_peak = sample - self.previous_peak
        dist_searchback_end = sample - self.searchback_end

        # Handle circular index wrapping
        if dist_previous_peak < 0:
            dist_previous_peak += self.MAX_SAMPLES
        if dist_searchback_end < 0:
            dist_searchback_end += self.MAX_SAMPLES

        is_qrs = False

        # **Check if search-back is required**
        if dist_previous_peak > self.max_rr_width and dist_searchback_end > self.max_rr_width:
            
            self.searchback_end = sample
            for i in range(current_int):
                temp = self.integrated_data[i]
                temp = torch.as_tensor(temp, dtype=torch.float32)
                self.peaki = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                is_qrs = False

                if self.peaki > self.threshold2:
                    temp = 0.750 * self.spki
                    temp = torch.as_tensor(temp, dtype=torch.float32)
                    temp = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                    temp2 = 0.250 * self.peaki
                    temp2 = torch.as_tensor(temp2, dtype=torch.float32)
                    temp2 = finalize_out_dtype(temp2, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                    temp = temp + temp2
                    temp = torch.as_tensor(temp, dtype=torch.float32)
                    self.spki = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                    # Update SPKI with new value
                    # self.spki = 0.750 * self.spki + 0.250 * self.peaki
                    is_qrs = True

                if is_qrs:
                    if self.peak_counter == 0 or dist_previous_peak >= self.min_rr_width:
                        self.peak_counter += 1
                        self.R_loc[(self.peak_counter-1)%self.N_AVG] = sample
                        self.R_loc_all.append(self.R_loc[(self.peak_counter - 2) % self.N_AVG])
                        
                    elif(self.integrated_data[0] < self.peaki):
                        self.R_loc[(self.peak_counter-1)%self.N_AVG] = sample  # Replace last detected peak
                    
                    self.previous_peak = sample
                else:
                    temp = 0.875 * self.npki
                    temp = torch.as_tensor(temp, dtype=torch.float32)
                    temp = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                    temp2 = 0.125 * self.peaki
                    temp2 = torch.as_tensor(temp2, dtype=torch.float32)
                    temp2 = finalize_out_dtype(temp2, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                    temp = temp + temp2
                    temp = torch.as_tensor(temp, dtype=torch.float32)
                    self.npki = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                    # Update NPKI with new value
                    # self.npki = 0.875 * self.npki + 0.125 * self.peaki

                # Update thresholds
                temp = (self.spki - self.npki)
                temp = torch.as_tensor(temp, dtype=torch.float32)
                temp = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                temp2 = 0.25 * temp
                temp2 = torch.as_tensor(temp2, dtype=torch.float32)
                temp2 = finalize_out_dtype(temp2, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                temp = self.npki + temp2
                temp = torch.as_tensor(temp, dtype=torch.float32)
                self.threshold1 = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                # self.threshold1 = self.npki + 0.25 * (self.spki - self.npki)
                temp = 0.5 * self.threshold1
                temp = torch.as_tensor(temp, dtype=torch.float32)
                self.threshold2 = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                # self.threshold2 = 0.5 * self.threshold1

        # **Normal QRS detection without search-back**
        else:
            
            self.peaki = self.integrated_data[current_int]
            is_qrs = False
            if self.peaki > self.threshold1:
                temp = 0.875 * self.spki
                temp = torch.as_tensor(temp, dtype=torch.float32)
                temp = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                temp2 = 0.125 * self.peaki
                temp2 = torch.as_tensor(temp2, dtype=torch.float32)
                temp2 = finalize_out_dtype(temp2, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                temp = temp + temp2
                temp = torch.as_tensor(temp, dtype=torch.float32)
                self.spki = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                # Update SPKI with new value
                # self.spki = 0.875 * self.spki + 0.125 * self.peaki
                is_qrs = True

            if is_qrs:
                prev_peak_idx = current_int - dist_previous_peak
                if prev_peak_idx < 0:
                    prev_peak_idx = 0

                if self.peak_counter == 0 or dist_previous_peak >= self.min_rr_width:
                    self.peak_counter += 1
                    self.R_loc[(self.peak_counter-1)%self.N_AVG] = sample 
                    self.R_loc_all.append(self.R_loc[(self.peak_counter - 2) % self.N_AVG])
                    

                elif self.integrated_data[prev_peak_idx] < self.peaki:
                    self.R_loc[(self.peak_counter-1)%self.N_AVG] = sample # Replace last detected peak

                self.previous_peak = sample
        
            else:
                temp = 0.875 * self.npki
                temp = torch.as_tensor(temp, dtype=torch.float32)
                temp = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                temp2 = 0.125 * self.peaki
                temp2 = torch.as_tensor(temp2, dtype=torch.float32)
                temp2 = finalize_out_dtype(temp2, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                temp = temp + temp2
                temp = torch.as_tensor(temp, dtype=torch.float32)
                self.npki = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
                # Update NPKI with new value
                # self.npki = 0.875 * self.npki + 0.125 * self.peaki

            # Update thresholds
            temp = (self.spki - self.npki)
            temp = torch.as_tensor(temp, dtype=torch.float32)
            temp = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
            temp2 = 0.25 * temp
            temp2 = torch.as_tensor(temp2, dtype=torch.float32)
            temp2 = finalize_out_dtype(temp2, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
            temp = self.npki + temp2
            temp = torch.as_tensor(temp, dtype=torch.float32)
            self.threshold1 = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
            # self.threshold1 = self.npki + 0.25 * (self.spki - self.npki)
            temp = 0.5 * self.threshold1
            temp = torch.as_tensor(temp, dtype=torch.float32)
            self.threshold2 = finalize_out_dtype(temp, qrs_dtype, mantissa_bits=qrs_mantissa_bits)
            # self.threshold2 = 0.5 * self.threshold1

        # **Enable heart rate computation after enough peaks**
        if self.peak_counter >= self.N_AVG:
            self.peak_processing = 1
            
        # **Compute Heart Rate (HR) in BPM**
        hr = 0.0
        if self.peak_processing:
            for i in range(1, self.N_AVG):

                idx = (((self.peak_counter - 1) % self.N_AVG) + 1 + i) % self.N_AVG
                temp = self.R_loc[idx] - self.R_loc[(idx + self.N_AVG - 1) % self.N_AVG]
                if temp < 0:
                    temp += self.MAX_SAMPLES
                hr += temp
                hr = torch.as_tensor(hr, dtype=torch.float32)
                hr = finalize_out_dtype(hr, hr_dtype, mantissa_bits=hr_mantissa_bits)

            hr /= (self.N_AVG - 1)
            hr = torch.as_tensor(hr, dtype=torch.float32)
            hr = finalize_out_dtype(hr, hr_dtype, mantissa_bits=hr_mantissa_bits)
            temp = hr / self.FS
            temp = torch.as_tensor(temp, dtype=torch.float32)
            temp = finalize_out_dtype(temp, hr_dtype, mantissa_bits=hr_mantissa_bits)
            temp2 = 60.0 / temp
            temp2 = torch.as_tensor(temp2, dtype=torch.float32)
            self.heart_rate = finalize_out_dtype(temp2, hr_dtype, mantissa_bits=hr_mantissa_bits)

        # **Handle last index case**
        if last_index:
            if self.peak_counter > 0:
                self.R_loc_all.append(self.R_loc[(self.peak_counter - 1) % self.N_AVG])
            else:
                self.R_loc_all.append(self.R_loc[0])

    # ----------------------------
    # Main pipeline (streaming)
    # ----------------------------
    def pan_tompkins(self, filename=None, input_dt=DKind.FP32, dc_mantissa_bits: int = 2, qrs_dtype: DKind = DKind.FP32, qrs_mantissa_bits: int = 2, hr_dtype: DKind = DKind.FP32, hr_mantissa_bits: int = 2):


            
        initialized = False  # Ensure it's initialized

        # 1️⃣ **Load ECG Signal from MATLAB File**

        """Processes an ECG signal using the Pan-Tompkins algorithm."""
        ecg_signal, max_value = self.load_ecg_signal(filename)
        # Convert to Torch tensor
        # ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)
        ecg_signal = ecg_signal.clone().detach().to(dtype=torch.float32)

        N = len(ecg_signal)

        ############# Defining arrays to save the results #############
        # Initialize the heart rate array
        hr_array = np.zeros(len(ecg_signal))
        mwi_array = np.zeros(len(ecg_signal))
        lp_debug = np.zeros(len(ecg_signal))
        
        # Input dtype init (map torch dtype to DKind)
        # def _torch_to_dkind(td: torch.dtype) -> DKind:
        #     if td == torch.float32:   return DKind.FP32
        #     if td == torch.float16:   return DKind.FP16
        #     if td == torch.bfloat16:  return DKind.BF16
        #     # For FP8/custom, pass DKind.FP8_CUSTOM manually from caller if needed
        #     return DKind.FP32  # Default fallback

        
        # input_kind = _torch_to_dkind(input_dt)
        print(f"Input dtype: {input_dt}, Mapped kind: {input_dt}, Mantissa bits: {dc_mantissa_bits}")
        ecg_signal = matrix_init(ecg_signal, input_dt, mantissa_bits=dc_mantissa_bits)  # Initialize the input signal


        # Dump inputs (FP-only run)
        if not self.quantize:
            with open("data.h", "w", encoding="utf-8") as f:
                f.write(f"#define LENGTH {N}\n")
                f.write(f"#define BUFFER_SIZE {self.BUFFER_SIZE}\n")
                f.write(f"#define MAX_SAMPLES {self.MAX_SAMPLES}\n")
                f.write(f"#define N_AVG {self.N_AVG}\n")
                f.write(f"#define MAX_VALUE {int(max_value)}\n")
                f.write(f"#define FS {self.FS}\n")

                f.write("#ifndef DATA_H\n")
                f.write("#define DATA_H\n\n")
                write_matrix(ecg_signal, "input_data", N, f)
                f.write("#endif\n")


        # Streaming Processing
        for sample_idx, sample in enumerate(ecg_signal):

            # Update circular buffers

            current_signal = self.update_buffer(self.signal_data, sample_idx, self.NC_Lo, initialized, self.NC_Lo)
            current_lo = self.update_buffer(self.lowpass_data, sample_idx, self.NC_Hi, initialized, self.NC_Hi)
            current_hi = self.update_buffer(self.highpass_data, sample_idx, self.NC_Der, initialized, self.NC_Der)
            current_der = self.update_buffer(self.derivative_data, sample_idx, self.NC_Int, initialized, self.NC_Int)
            current_int = self.update_buffer(self.integrated_data, sample_idx, self.BUFFER_SIZE, initialized, self.BUFFER_SIZE)


            # 2️⃣ **Remove DC Component & Normalize Signal**
            # Simulate streaming by passing only the current sample and the previous one
            # Provide two consecutive samples for DC cancellation as in the C code
            result = dc_cancellation_and_normalize_python(sample_idx, ecg_signal, max_value, self.signal_data, input_dt, mantissa_bits=dc_mantissa_bits)

            # Update the signal_data buffer safely
            self.signal_data[current_signal] = result # Store the value in the correct index   

            # **Terminate if we cannot apply the first filter**
            if current_lo < self.NC_Lo - 1:
                continue

            # Convert Buffers to Tensors for Convolution
            signal_tensor = torch.tensor(list(self.signal_data), dtype=torch.float32)


            # 3️⃣ **Low-Pass FIR Filtering (Mimicking MATLAB h_LP)**
            lowpass_ecg = self.lowpass_filter(signal_tensor, current_signal)
            lp_debug[sample_idx] = lowpass_ecg[-1]
            self.lowpass_data[current_lo] = lowpass_ecg[-1]  # Store last computed value 
            lowpass_tensor = torch.tensor(list(self.lowpass_data), dtype=torch.float32)


            # 4️⃣ **High-Pass FIR Filtering (Mimicking MATLAB h_HP)**
            highpass_ecg = self.highpass_filter(lowpass_tensor, current_lo)
            self.highpass_data[current_hi] = highpass_ecg[-1]  # Store last computed value
            highpass_tensor = torch.tensor(list(self.highpass_data), dtype=torch.float32)

            # print("Derivative filter")
            # 5️⃣ **Compute Derivative (Mimicking MATLAB Derivative)**

            derivative_ecg = self.derivative_filter(highpass_tensor, current_hi)

            # 6️⃣ **Squaring Function**
            if self.quantize:
                datatypes = self.filter_configs["squared"]["datatypes"]
                mantissa_bits = self.filter_configs["squared"]["mantissa_bits"]
            else:
                datatypes = [DKind.FP32]
                mantissa_bits = [2,2]
            derivative_ecg = matrix_init(derivative_ecg, datatypes[0],mantissa_bits=mantissa_bits[0])
            squared_ecg = derivative_ecg ** 2
            self.derivative_data[current_der] = squared_ecg[-1]    # Store last computed value
            derivative_tensor = torch.tensor(list(self.derivative_data), dtype=torch.float32)


            # 7️⃣ **Moving Window Integration (MWI)**
            mwi_ecg = self.mwi_filter(derivative_tensor, current_der)    
            self.integrated_data[current_int] = mwi_ecg[-1]  # Store last computed value
            integrated_tensor = torch.tensor(list(self.integrated_data), dtype=torch.float32)
            # ✅ Call the R-peak detection function
            # **Detect R-Peaks**
            last_index = True if sample_idx == len(ecg_signal) - 1 else False
            self.detect_r_peaks(sample_idx, current_int, last_index=last_index, qrs_dtype=qrs_dtype, qrs_mantissa_bits=qrs_mantissa_bits, hr_dtype=hr_dtype, hr_mantissa_bits=hr_mantissa_bits)
            hr_array[sample_idx] = self.heart_rate
            mwi_array[sample_idx] = mwi_ecg[-1]
        return hr_array, ecg_signal, mwi_array, lp_debug    # Return the heart rate array and the ECG signal

# ----------------------------
# Main
# ----------------------------
def main():
    
    plot_flag = True # flag to plot the intermediate results
    FS = 128 # Set the sampling frequency
    
    # First, run the FP32 version of the algorithm
    print("Running FP32 version of the algorithm...")
    quantize = False
    fp32_processor = PanTompkinsProcessor(None, quantize, fs=FS)
    hr_fp32, ecg_signal_fp32, mwi_fp32, lp_fp32 = fp32_processor.pan_tompkins(filename="bartolini.txt")
    
    # R_loc_fp32, mwi_ecg_fp32 = non_streaming_peaks(mwi_fp32, FS, quantize, ecg_signal_fp32, plot=plot_flag)
    # checks(R_loc_fp32, mwi_ecg_fp32, FS, name="fp32", plot=plot_flag)
    # print(f"R_loc_fp32: {R_loc_fp32}")
    

    checks(fp32_processor.R_loc_all[1:], mwi_fp32, FS, name="fp32", plot=plot_flag)
    compute_accuracy(name="fp32")
    # Run the quantized version of the algorithm
    quantize = True

    filter_datatypes = {
        "lowpass": ["FP16ALT", "FP16ALT", "FP16ALT"],
        "highpass": ["FP16ALT", "FP16ALT", "FP16ALT"],
        "derivative": ["FP16ALT", "FP16ALT", "FP16ALT"],
        "squared": ["FP16ALT"],
        "mwi": ["FP16ALT", "FP16ALT", "FP16ALT"]
    }

    # Define mac_flag settings for each filter
    filter_mac_flags = {
        "lowpass": True,
        "highpass": True,
        "derivative": True,
        "squared": True,
        "mwi": True
    }

    # Define vec_flag settings for each filter
    filter_vec_flags = {
        "lowpass": True,
        "highpass": True,
        "derivative": True,
        "squared": True,
        "mwi": True
    }

    filter_mantissa_bits = {
        "lowpass": [2, 2, 2], # input, filter, output
        "highpass": [2, 2, 2],
        "derivative": [2, 2, 2],
        "squared": [2, 2, 2],
        "mwi": [2, 2, 2]
    }

    # Dictionary to store the configurations for each filter
    filter_configs = {}

    # Assign values manually using dictionary lookups
    for filter_name in filter_datatypes:
        bits = filter_datatypes[filter_name]
        
        datatypes = select_dtypes(bits, len(bits))
        cast_flag = check_cast(datatypes)
        cast_to = bits[-1] if cast_flag else None
        
        print(f"Filter: {filter_name}, Datatypes: {[dkind_name(dt) for dt in datatypes]}, Cast: {cast_flag}, Cast To: {cast_to}")

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
            "mantissa_bits": mantissa_bits
        }
    # Print the filter configurations
    print()
    print("Running Quantized version of the algorithm...")
    
    input_dt = DKind.BF16  # Input data type (can be changed to test different types)
    dc_mantissa_bits = 2  # Mantissa bits for DC cancellation (can be adjusted)

    qrs_dtype = DKind.BF16  # Data type for QRS detection
    qrs_mantissa_bits = 2  # Mantissa bits for QRS detection

    hr_dtype = DKind.BF16  # Data type for heart rate
    hr_mantissa_bits = 2  # Mantissa bits for heart rate

    q_processor = PanTompkinsProcessor(filter_configs, quantize, fs=FS)
    hr_q, ecg_signal, mwi_q, lp_q = q_processor.pan_tompkins(filename="bartolini.txt", input_dt=input_dt, dc_mantissa_bits=dc_mantissa_bits, qrs_dtype=qrs_dtype, qrs_mantissa_bits=qrs_mantissa_bits, hr_dtype=hr_dtype, hr_mantissa_bits=hr_mantissa_bits)

    # R_loc_q, mwi_ecg_q = non_streaming_peaks(mwi_q, FS, quantize, ecg_signal, plot=plot_flag)
    # checks(R_loc_q, mwi_ecg_fp32, FS, name="quantized", plot=plot_flag)
    
    # I am using mwi_fp32 because I want to compare the results with the FP32 version
    checks(q_processor.R_loc_all[1:], mwi_fp32, FS, name="quantized", plot=plot_flag)
    compute_accuracy(name="quantized")
    

    with open("debug/comparison_hr.txt", "w", encoding="utf-8") as f:
        for i in range(len(hr_fp32)):
            f.write(f"Sample {i}: HR_FP32 = {hr_fp32[i]:.2f} BPM, HR_Q = {hr_q[i]:.2f} BPM\n")
    print("Heart rate comparison information saved in debug/comparison_hr.txt")


    # Only compute max_diff where hr_q is not zero
    valid_indices = np.where((hr_q != 0) & (hr_fp32 != 0))[0]
    if len(valid_indices) > 0:
        max_diff = np.max(np.abs(hr_fp32[valid_indices] - hr_q[valid_indices]))
        max_diff_index = valid_indices[np.argmax(np.abs(hr_fp32[valid_indices] - hr_q[valid_indices]))]
        avg_error = np.mean(np.abs(hr_fp32[valid_indices] - hr_q[valid_indices]))
    else:
        max_diff = 0
        max_diff_index = -1
        avg_error = 0
    # max_diff_index = np.argmax(np.abs(hr_fp32 - hr_q))
    # avg_error = np.mean(np.abs(hr_fp32 - hr_q))
    
    print(f"Max difference between FP32 and Quantized: {max_diff:.2f} BPM at index {max_diff_index}")
    print(f"Average error between FP32 and Quantized: {avg_error:.2f} BPM")

    hr_q = torch.tensor(hr_q, dtype=torch.float32)
    if quantize:
        with open("ref.h", "w", encoding="utf-8") as f:
            write_matrix(hr_q, "reference", len(hr_q), f)
            

if __name__ == "__main__":
    main()
    # Run the main function