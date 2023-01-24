
# TransLib
TransLib, an open-source kernel library based on transprecision computing principles, which provides knobs to exploit different FP data types (i.e., float, float16, and bfloat16), also considering the trade-off between homogeneous and mixed-precision solutions. 

We demonstrate the capabilities of the proposed library on PULP, a 32-bit microcontroller (MCU) coupled with a parallel, programmable accelerator. We performed a comparative analysis of seven standard kernels used in IoT end-node processing to evaluate our proposed method.


## Get Started
Each kernel design includes a Python model and a C program. The Python model generates the input dataset, computes the kernel output as a golden reference, and assesses the accuracy using a customizable error metric. Each folder contains a specific test with the golden model generator and a brief description of how to run the test.  
### Prerequisites 
#### Python
These golden models are built on top of PyTorch data types. The following packages needed to be installed:
~~~~~shell
pip install pandas torch matplotlib pywavelets scipy
~~~~~
#### PULP-SDK
These tests requires the [PULP-SDK](https://github.com/pulp-platform/pulp-sdk). Once you cloned the PULP-SDK repository and you have the [RISC-V GNU Compiler Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain) installed, you need to compile [GVSOC](https://github.com/pulp-platform/pulp-sdk#gvsoc). **Please refer to the links to correctly setup your working environment.**

Here is my suggestion:

1-  First install and compile the [RISC-V GNU Compiler Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#risc-v-gnu-compiler-toolchain).

Follow the next steps in the [RISC-V GNU Compiler Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#risc-v-gnu-compiler-toolchain) repository.

- [Getting the sources](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#getting-the-sources)
- [Prerequisites](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#prerequisites)
- [Installation (Pulp)](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#installation-pulp)

2- Install and compile [PULP-SDK](https://github.com/pulp-platform/pulp-sdk#pulp-sdk).

Please follow the next setups in the [PULP-SDK](https://github.com/pulp-platform/pulp-sdk#pulp-sdk) repository
- [Getting started](https://github.com/pulp-platform/pulp-sdk#getting-started)
- [GVSoC](https://github.com/pulp-platform/pulp-sdk#gvsoc)

3- Finally, test the installation according to [Test execution](https://github.com/pulp-platform/pulp-sdk#test-execution)


**Don't forgot to source the file corresponding to the desired configuration when you want to use the TransLib again** :

~~~~~shell
cd pulp-sdk
source configs/pulp-open.sh
~~~~~
## Repository Organization
In this repository there are some of the most common applications which exploit floating-point numbers. Each folder contains a specific test with the golden model generator and a brief description of how to run the test.  

Since some kernels don't support mixed-precision in current version, we have divided the kernels to [fixed-precision test ](./fixed_precision/) and [mixed-precision test](./mixed_precision/).
These are the developed tests:

- [Matrix Multiplication test](./mixed_precision/matmul/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [Convolution test](./mixed_precision/convolutioncl/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [DWT test](./fixed_precision/dwt) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [FFT test](./fixed_precision/fft-memsave) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [FIR filter test](./mixed_precision/fir) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [K-means algorithm test](./fixed_precision/kmeans) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [SVM classification test](./mixed_precision/SVM/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point

## Acknowledgements
This work was supported by the [APROPOS](https://projects.tuni.fi/apropos/) project (g.a. no. 956090), founded by the European Union’s Horizon 2020 research and innovation program. 


## Contributors
- [Giuseppe Tagliavini](https://github.com/gtagliavini), University of Bologna,[E-mail](giuseppe.tagliavini@unibo.it)
