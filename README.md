# TransBench
This repository contains a set of kernel for the PULP cluster or FABRIC.
We performed a comparative analysis of seven standard kernels used in IoT end-node processing to evaluate our proposed method.


## Setup
These tests requires the [PULP-SDK](https://github.com/pulp-platform/pulp-sdk). Once you cloned the PULP-SDK repository and you have the [RISC-V GNU Compiler Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain) installed,

1. If you want to execute these tests on RTL, source the vsim file from the RTL platform folder

~~~~~shell
source setup/vsim.sh
~~~~~

2. Export the path to the toolchain with the following command

~~~~~shell
export PULP_RISCV_GCC_TOOLCHAIN=<INSTALL_DIR>
~~~~~

3. Source the file corresponding to the desired configuration:

~~~~~shell
cd pulp-sdk
source configs/pulp-open.sh
~~~~~

## Repository organization
In this repository there are some of the most common applications which exploit floating-point numbers. Each folder contains a specific test with the golden model generator and a brief description of how to run the test. 
Since some kernels don't support mixed-precision in current version, we have divided the kernels to [fixed-precision test ](./fixed-precision/) and [mixed-precision test](./mixed-precision/).
These are the developed tests:

- [Matrix Multiplication test](./mixed-precision/matmul/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [Convolution test](./mixed-precision/convolution/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [DWT test](./fixed-precision/DWT) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [FFT test](./fixed-precision/FFT) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [FIR filter test](./mixed-precision/fir) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [K-means algorithm test](./fixed-precision/kmeans) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [SVM classification test](./mixed-precision/SVM/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
