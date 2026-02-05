# FIR filter test and Floating-Point Format Exploration

---
This test performs signal processing through a FIR (Finite Impulse Response) filter.

## FIR Test

This test performs a FIR filter on various floating-point formats:
- `FP32`
- `FP16`
- `FP16ALT`
- `FP8_e5m2`

It can also be used to **measure performance**.

In this folder, you can find **pre-generated golden models** for validation.

---

## Running a test

### 1-Generating the golden model
If you want to regenerate a golden model to check the output of the C code, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
python3 data_generator.py --length=length --order=order --float_type=fmt --mac_flag=MAC_FLAG --vec_flag=VEC_FLAG --reversed=Reversed_flag
~~~~~
- `float_type` can be FP32, FP16, FP16ALT, or FP8_CUSTOM.

    For mixed-precision, use:

    ~~~~~shell
    --float_type=FP_INP,FP_FIL,FP_OUT
    ~~~~~

    where `FP_INP`, `FP_FIL`, and `FP_OUT` are the data types for the input, filter, and output matrices, respectively.

- `hwmixed_flag --> true|false`  emulates the `-mfaux` fused mixed-precision accumulate; 

    > ⚠️ **Note:**    
        > it works only when `FP_INP`, and `FP_FIL` are **the same type** (e.g., FP8,FP8,FP32✅).

- `mac_flag --> true|false` is used to emulate the multiply-and-add operator available on most DSP instruction sets for embedded devices.

- `vec_flag --> true|false` to emulate SIMD vector instructions. Vectorization is available for fixed precision and mixed precision.

    > ⚠️ **Note:**  
    > Mixed-precision vectorization is only supported when the two multiplicands use the same floating-point format (A == B). The accumulator/result format may differ.  
    > 
    > Examples (A × B → C):  
    > 
    > - FP8 × FP8 → FP32 ✅  
    > - FP16 × FP16 → FP16ALT ✅  
    > - FP16ALT × FP16ALT → FP16 ✅  
    > - FP8 × FP16 → FP32 ❌ (not supported: mismatched multiplicand formats)


- `reversed --> true|false` reverses the FIR filter coefficients to align with the C implementation’s memory access pattern for vectorized execution, avoiding runtime shuffling; this flag is automatically saved by the data generator to ensure the correct C code is executed.

In the data_generator, the mantissa bits for float8 are configured to 2, aligning with the float8_e5m2 format, which is supported by PULP platforms.
##
The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT/FP8_e5m2):

### 2- Running the C code
After setting up the platform and SDK, you can run the test with:

~~~~~shell
make clean all run
~~~~~


There are several flags useful to activate some functionalities:

- `cores=N_CORES` Set the number of cores (N_CORES) used for execution (default: 1 core).
Alternatively, you can run on the Fabric Controller using `FABRIC=1` instead of `cores=N_CORE`.

- `fmt=FP_FMT` Specify the floating-point format (FP32, FP8, FP16, or FP16ALT).
    For mixed-precision, specify:
    ~~~~~shell
    fmt_INP=FP_INP   fmt_FIL=FP_FIL   fmt_OUT=FP_OUT
    ~~~~~
    where `FP_INP`, `FP_FIL`, and `FP_OUT` are the data types for the input, filter, and output matrices, respectively.

- `hwmixed=1` To enable the fused mixed-precision accumulation

- `vec=1`  Enable vectorial (SIMD) format.

- `no_fmadd=1` Disable all Fused MAC (If `mac_flag==false` in the data generator)

- `check=1` Enable result checking against the golden model.

- `verbose=1` Print incorrect result details.

- `stats=1` Enable performance measurement.

- `print_results=1` Print output matrices to console.

- `target=TARGET` To distinguish between PULP and GAP9 toolchains, a makefile flag is available:

    target=pulp (default)

    target=gap9 (for running the code on GAP9)

### 🛠️ CMake Support

This kernel also supports execution using **CMake**, which is required by newer versions of the **GAP SDK**.  
A `CMakeLists.txt` file has been added to ensure compatibility with CMake-based toolchains.


## Exploring floating point formats and different data distribution
You can use [exploration_script.py](./exploration_script.py) to analyze the behavior of various floating-point formats, including:
- `FP16`
- `FP16ALT`
- `FP8_CUSTOM`
- `FP32`

This script evaluates FIR filter across different data distributions by specifying the **standard deviation** (`std`) of the random matrices.

You can modify the list of standard deviations inside [exploration_script.py](./exploration_script.py).


---

### ✨ How It Works:

- The script explores **fixed-precision** and **mixed-precision** floating-point settings.
- It evaluates different combinations of input and output data type formats (`input length`, `order`) and **mantissa bits** (for `FP8_CUSTOM`).
- The exploration is **parallelized** for faster execution (you can configure the number of parallel workers).
- **Error metrics** such as **MAE**, **MSE**, **RMSE**, **R²**, and **RAE** are collected automatically.

---

### 📄 Output:

- All results are collected into a **single Excel file**.
- The Excel file (`exploration_results_parallel.xlsx`) is saved inside the **exploration folder**.

Each row in the Excel file includes:
- Dimensions (`length`, `order`)
- Data types used
- Standard deviation
- Mantissa bits
- Error metrics (MAE, MSE, RMSE, R², RAE)

---

### 🚀 Usage Example:

~~~~~shell
python3 exploration_script.py --length=1024 --order=64 --max_workers=8
~~~~~

## Test Runner

This repository contains a small pipeline to **generate data** with `data_generator.py` and then **compile & run C code** with `make`.  
The pipeline is driven by a `config.sh` file and a `run_pipeline.sh` script.

---

### Usage

1. **Edit the configuration**  
   Define your runs in `config.sh`. Each run has two parts:  
   - arguments for the Python data generator (`PY:`)  
   - arguments for the C/Make step (`MAKE:`)  

   Example:
   ```bash
   RUNS=(
     "PY: --length=128 --order=35 --float_type=FP16 --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 vec=1"
   )

2. **Run the pipeline**  
Make the script executable and launch it:

    ```bash
    chmod +x run_pipeline.sh
    ./run_pipeline.sh
    ```


This will:

- run the Python generator with your chosen options,

- run make with the specified arguments,

- print everything to the terminal and save logs into ./logs/.

3. **Check logs**  
For each run you’ll get two log files:

    ```bash
    logs/run1_python_<timestamp>.log
    logs/run1_make_<timestamp>.log
    ```
    These contain the full terminal output for the Python and C steps.

4. **Debug mode (optional)**  
To see detailed execution steps:

    ```bash
    DEBUG=1 ./run_pipeline.sh
    ```

    This prints each command before running it (helpful for troubleshooting).

