# SVM Test and Floating-Point Format Exploration
This test performs an SVM (Support Vector Machine) classificationacross various floating-point formats:

- `FP32`
- `FP16`
- `FP16ALT`
- `FP8_e5m2`

It can also be used to **measure performance**.

In this folder, you can find **pre-generated golden models** for validation.


## Running a test

### 1-Running the golden model
If you want to regenerate a golden model to check the output of the C code, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --input_size=INPUT_SIZE --kernel=KERNEL  --dataset=DATASET --float_type=FP** --mac_flag=MAC_FLAG --vec_flag=VEC_FLAG
~~~~~
- `input_size` Number of features in the input data.

- `kernel`: Type of kernel to use, either linear or rbf.

- `dataset`: Specify which dataset to use. Options include:

    - `bill`: Loads the Bill Authentication dataset.

    - `cancer`: Loads the Breast Cancer dataset.

    - `custom`: Generates a synthetic dataset with random values. For the custom option, you can modify the number of features by adjusting the `num_features` parameter in the `read_dataset` function within the [data_generator.py](./data_generator.py) script. By default, this is set to `48`.

    Predefined datasets are located in the [dataset](./dataset) folder.

- `float_type`: Floating-point precision. Defaults to `FP32`, but can be set to:
    - `FP16`
    - `FP16ALT`
    - `FP8_CUSTOM`

    For mixed-precision, use:

    ~~~~~shell
    --float_type=FP_INP,FP_FIL,FP_OUT
    ~~~~~

    where `FP_INP`, `FP_FIL`, and `FP_OUT` represent the input, filter, and output formats respectively.

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

The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT/FP8_e5m2).

> ⚠️ **Note:**  
> This version of the SVM kernel supports **binary classification only** (i.e., 2-class problems).


In the data_generator, the mantissa bits for float8 are configured to 2, aligning with the float8_e5m2 format, which is supported by PULP platforms.

### 2-Running the C code
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

- `vec=1` Activates vectorial format

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
## Floating-Point Format Exploration: 

You can use [exploration_script.py](./exploration_script.py) to analyze the behavior of various floating-point formats, including:
- `FP16`
- `FP16ALT`
- `FP8_CUSTOM`
- `FP32`

### ✨ How It Works:

- The script explores **fixed-precision** and **mixed-precision** floating-point settings.
- It evaluates different combinations of input, filter and output data type formats and **mantissa bits** (for `FP8_CUSTOM`).
- The exploration is **parallelized** for faster execution (you can configure the number of parallel workers).
- **Accuracy metric** is collected automatically.

---

### 📄 Output:

- All results are collected into a **single Excel file**.
- The Excel file (`sv_exploration_results_parallel.xlsx`) is saved inside the **exploration folder**.

Each row in the Excel file includes:
- Dimensions
- Data types used
- Mantissa bits
- Accuracy metric
---

### 🚀 Usage Example:

~~~~~shell
python3 exploration_script.py --input_size=30 --kernel=linear --dataset=custom --max_workers=8
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
     "PY: --input_size=64 --kernel=linear --dataset=custom  --float_type=FP16 --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16  vec=1"

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

