# FFT Test and Floating-Point Format Exploration
This test performs an FFT (Fast Fourier Transform)across various floating-point formats:

- `FP32`
- `FP16`
- `FP16ALT`
- `FP8_e5m2`

It can also be used to **measure performance**.

In this folder, you can find **pre-generated golden models** for validation.

**The C code only supports fixed-precision.**

## Running a test

~~~~~shell
cd /fft_mixed_2_8/
~~~~~
### 1-Running the golden model
If you want to regenerate a golden model to check the output of the C code, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
python3 data_generator.py --input_size=INPUT_SIZE --float_type=fmt --scaling_method=method --scale_value=value --target_range=range --mac_flag=MAC_FLAG --vec_flag=VEC_FLAG
~~~~~
- `input_size` Number of features in the input data.

    The following arguments allow you to control how the input data is preprocessed before FFT execution.
    - `scaling_method` Method to scale the input data. One of:
        - `normalize` – scale to a specified range (e.g., [0, 1])
        - `standardize` – zero-mean, unit-variance
        -  `multiplicative` – simple scalar multiplication
        
        Default: normalize
    - `scale_value` 	Scaling factor applied when `scaling_method=multiplicative`. For example, `scale_value=0.0025` multiplies each sample by 0.0025.
    
        Default: 0.0025
    
    - `target_range` Used **only when `scaling_method=normalize`** to define the minimum and maximum range of the output values. For example, `--target_range=-1,1` scales input data to lie within [-1, 1].<br><br>This parameter is **ignored** if the scaling method is `standardize` or `multiplicative`.

        Default: "0,1"

- `float_type` can be FP32, FP16, FP16ALT, or FP8_CUSTOM. 

    For mixed-precision, use:

    ~~~~~shell
    --float_type=FP_INP,FP_FIL,FP_OUT
    ~~~~~

    where `FP_INP`, `FP_FIL`, and `FP_OUT` are the data types for the input, filter, and output matrices, respectively.
    **But consider that the C code only supports fixed-precision.**

    In the data_generator, the mantissa bits for float8 are configured to 2, aligning with the float8_e5m2 format, which is supported by PULP platforms.

- `mac_flag --> true|false` is used to emulate the multiply-and-add operator available on most DSP instruction sets for embedded devices.

- `vec_flag --> true|false` to emulate SIMD vector instructions. Vectorization is available for fixed precision and mixed precision.

The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT).

**It should be mentioned that for half-precision floating point (FP16 and FP16ALT) there is a small error based on some optimizations in the C code.**


The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT).

### 2-Running the C code
After setting up the platform and SDK, you can run the test with:

~~~~~shell
make clean all run
~~~~~

There are several flags useful to activate some functionalities:

- `cores=N_CORES` Set the number of cores (N_CORES) used for execution (default: 1 core).
Alternatively, you can run on the Fabric Controller using `FABRIC=1` instead of `cores=N_CORE`.

- `fmt=FP_FMT` specifies the floating-point format for data, by default, it is set to `FP32` but you can also choose `FP16` and `FP16ALT` formats

- `vec=1`  Enable vectorial (SIMD) format (only available for FP16 and FP16ALT).

- `no_fmadd=1` Disable all Fused MAC (If `mac_flag==false` in the data generator)

- `stats=1` Enable performance measurement.

- `print_results=1` Print output matrices to console.

- `target=TARGET` To distinguish between PULP and GAP9 toolchains, a makefile flag is available:

    target=pulp (default)

    target=gap9 (for running the code on GAP9)

Note: The current C implementation of the FFT kernel does not support FP8 format; only fixed-precision types like FP32, FP16, and FP16ALT are supported for execution on the target hardware.
### 🛠️ CMake Support

This kernel also supports execution using **CMake**, which is required by newer versions of the **GAP SDK**.  
A `CMakeLists.txt` file has been added to ensure compatibility with CMake-based toolchains.


## Exploring Floating-Point Formats and Signal Scaling for FFT
You can use [exploration_script.py](./exploration_script.py) to analyze the behavior of the FFT kernel under different floating-point representations and input scaling techniques.

This script helps you evaluate precision loss and format tradeoffs when simulating embedded implementations of DWT using:
- `FP16`
- `FP16ALT`
- `FP8_CUSTOM`
- `FP32`

It also assesses the impact of scaling methods (normalization, standardization, and multiplicative) on numerical stability and output quality.


---

### ✨ How It Works:

- The script explores **fixed-precision** and **mixed-precision** floating-point settings.
- It evaluates different combinations of input, filter and output data type formats and **mantissa bits** (for `FP8_CUSTOM`).
- Multiple scaling strategies:
    - normalize to ranges like [0, 1] and [-1, 1]

    - standardize to zero-mean, unit-variance

    - multiplicative scaling (e.g., by 0.0025)
- The exploration is **parallelized** for faster execution (you can configure the number of parallel workers).
- **Error metrics** such as **MAE**, **MSE**, **RMSE**, **R²**, and **RAE** are collected automatically.

You can modify the list of normalization ranges, multiplicative scaling values, and other configuration parameters directly inside [exploration_script.py](./exploration_script.py) to customize the exploration space to your needs.

---

### 📄 Output:

- All results are collected into a **single Excel file**.
- The Excel file (`exploration_results_parallel.xlsx`) is saved inside the **exploration folder**.

Each row in the Excel file includes:
- Float format configuration (input, filter, output)

- Mantissa bits (if using FP8_CUSTOM)

- Scaling method and parameters

- Input size

- All error metrics

---

### 🚀 Usage Example:

~~~~~shell
python3 exploration_script.py --input_size=256 --max_workers=8

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
    "PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 vec=1 print_results=1"
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

