## Pan-Tompkins ECG Processing with Quantization Support
This repository contains a modular implementation of the Pan-Tompkins algorithm in Python. It supports both full-precision (FP32) and quantized (e.g., FP16, FP16ALT, FP8) modes with streaming-based signal processing, allowing for real-time and embedded system compatibility.

## 📋 Features

- ✅ Modular implementation of all Pan-Tompkins processing stages:
  - Low-pass filter
  - High-pass filter
  - Derivative filter
  - Squaring
  - Moving Window Integration (MWI)
- ✅ Streaming buffer logic to mimic embedded behavior
- ✅ Full-precision (FP32) and quantized (FP16, FP8_CUSTOM) data support
- ✅ Export of signal and reference data to `.h` headers for C-based environments
- ✅ Heart rate computation and R-peak detection via adaptive thresholding
- ✅ Performance evaluation using RR intervals and F1 score
- ✅ Optional visualization and debugging utilities

---

## 🧠 Algorithm Overview

This implementation closely follows the Pan-Tompkins algorithm:

1. DC removal and normalization
2. Low-pass filtering
3. High-pass filtering
4. Derivative filtering
5. Signal squaring
6. Moving window integration
7. Adaptive thresholding + search-back logic
8. Heart rate estimation


---

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have Python ≥ 3.8. Then install required packages:

```bash
pip install numpy torch matplotlib scipy
```
### 2. Run the Pipeline

```bash
python streaming.py
```

This will:
- Load a `.txt` file
- Run both FP32 and quantized versions
- Compute and compare R-peak detection performance
- Export C headers with input and HR output(As reference for C output)

#### Dataset

The [dataset](./dataset/) folder contains `.txt` files with real ECG signals. These files can be used as input for the pipeline to test and evaluate the algorithm's performance.

---

#### ⚙️ Configuration

#### Filter Precision Configuration

- `filter_datatypes`: A dictionary to specify the precision for each stage of the filters. 
    - The first data type represents the precision for the input.
    - The second data type represents the precision for the coefficients.
    - The third data type represents the precision for the output (summation).

#### Mantissa Bits Configuration

- `filter_mantissa_bits`: A dictionary to control the number of mantissa bits for **only quantized float8** datatype. 
    - Each value in the list corresponds to the mantissa bits for the respective stage of the filter.

### Optional Simulation Flags

- `vec_flag`: A boolean flag to enable or disable emulation of SIMD vector instructions for filters.
- `mac_flag`: A boolean flag to enable or disable emulation of multiply-and-accumulate operations for embedded DSP systems.

These configurations allow fine-tuning of precision, mantissa bits, and performance for different stages of the Pan-Tompkins algorithm.


---

## 📊 Evaluation

The `checks()` function compares quantized detected R-peaks against a ground-truth method using:

- True Positives (TP), False Positives (FP), False Negatives (FN)
- F1₁ Score
- Error Rate (%)
- RR Interval deviation
- RMSD and accuracy

Results are saved in `debug/` and plotted in `figure/`.

---

## 🧪 Visualization

Set `plot_flag = True` in `main()` to enable plotting for:

- Peak comparison overlays

---

## 🛠 Embedded Integration

Use the generated headers:

- `data.h` — contains ECG input signal (`input_data`)
- `ref.h` — contains expected HR output (`reference`)

---

After setting up the platform and SDK, you can run the test with:

~~~~~shell
make clean all run FABRIC=1
~~~~~


There are several flags useful to activate some functionalities:

- `fmt=FP_FMT` Specify the floating-point format (FP32, FP8, FP16, or FP16ALT).
<!-- For mixed-precision, specify:
~~~~~shell
fmt_INP=FP_INP   fmt_FIL=FP_FIL   fmt_OUT=FP_OUT
~~~~~
where `FP_INP`, `FP_FIL`, and `FP_OUT` are the data types for the input, filter, and output matrices, respectively. -->

- `vec=1`  Enable vectorial (SIMD) format (only available for FP16, FP16ALT, and FP8).
- `debug=1` Enable result checking against the golden model.
- `stats=1` Enable performance measurement.

## 📌 Notes

- Uses buffer-based logic for all FIR filters to simulate streaming
- Quantized settings (e.g., FP16, FP8) are configurable

---


## Reference

This implementation is based on the algorithm described in the following paper:

> Benedetta Mazzoni, Giuseppe Tagliavini, Luca Benini, Simone Benatti 
> "**An Optimized Heart Rate Detection System Based on Low-Power Microcontroller Platforms for Biosignal Processing**,"  
> In International Conference on System-Integrated Intelligence (pp. 160-170). Cham: Springer International Publishing.
<!-- > [https://doi.org/10.1007/978-3-031-16281-7_16](https://doi.org/10.1007/978-3-031-16281-7_16) -->

