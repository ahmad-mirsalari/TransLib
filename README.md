
# TransLib
TransLib, an open-source kernel library based on transprecision computing principles, provides knobs to exploit different FP data types (i.e., float, float16, float8, and bfloat16), also considering the trade-off between homogeneous and mixed-precision solutions. 

We demonstrate the capabilities of the proposed library on **PULP** (a 32-bit microcontroller (MCU) coupled with a parallel, programmable accelerator) and **GAP9** (Ultra-low-power RISC-V SoC with compute cluster). We performed a comparative analysis of seven standard kernels used in IoT end-node processing to evaluate our proposed method.

## Reference
This library is based on the research outlined in the following paper:

- Mirsalari, S.A., Tagliavini, G., Rossi, D., and Benini, L., "TransLib: A Library to Explore Transprecision Floating-Point Arithmetic on Multi-Core IoT End-Nodes", 2023 Design, Automation & Test in Europe Conference & Exhibition (DATE), 2023, [Link to the paper](https://ieeexplore.ieee.org/abstract/document/10136916)

If you find this library useful in your research, please consider citing the paper:

> ```
> @INPROCEEDINGS{10136916,
>   author={Mirsalari, Seyed Ahmad and Tagliavini, Giuseppe and Rossi, Davide and Benini, Luca},
>   booktitle={2023 Design, Automation & Test in Europe Conference & Exhibition (DATE)},
>   title={TransLib: A Library to Explore Transprecision Floating-Point Arithmetic on Multi-Core IoT End-Nodes},
>   year={2023},
>   volume={},
>   number={},
>   pages={1-2},
>   doi={10.23919/DATE56975.2023.10136916}}
> ```

## Get Started
Each kernel design includes a Python model and a C program. The Python model generates the input dataset, computes the kernel output as a golden reference, and assesses the accuracy using a customizable error metric. Each folder contains a specific test with the golden model generator and a brief description of how to run the test.  
### Prerequisites 
#### Python
These golden models are built on top of PyTorch data types. The following packages need to be installed:
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


**Don't forget to source the file corresponding to the desired configuration when you want to use the TransLib again** :

~~~~~shell
cd pulp-sdk
source configs/pulp-open.sh
~~~~~
## Repository Organization
Each folder contains a specific test with the golden model generator and a brief description of how to run the test.  

Since some kernels don't support mixed-precision in the current version, we have divided the kernels into [fixed-precision test ](./fixed_precision/) and [mixed-precision test](./mixed_precision/).
These are the developed tests:

- [Matrix Multiplication test](./mixed_precision/matmul/)
- [Convolution test](./mixed_precision/convolutioncl/)
- [DWT test](./mixed_precision/dwt)
- [FFT test](./fixed_precision/fft-memsave)
- [FIR filter test](./mixed_precision/fir)
- [K-means algorithm test](./mixed_precision/kmeans)
- [SVM classification test](./mixed_precision/SVM/)


## Roadmap

- Expand the library by adding additional kernels 

## License 
 TransLib is released under Apache 2.0, see the [LICENSE](./LICENSE.md) file in the root of this repository for details.

## Acknowledgements
This work was supported by the [APROPOS](https://projects.tuni.fi/apropos/) project (g.a. no. 956090), founded by the European Union’s Horizon 2020 research and innovation program. 


## Contributors
- [Seyed Ahmad Mirsalari](https://github.com/ahmad-mirsalari), University of Bologna,[E-mail](mailto:seyedahmad.mirsalar2@unibo.it)
- [Giuseppe Tagliavini](https://github.com/gtagliavini), University of Bologna,[E-mail](mailto:giuseppe.tagliavini@unibo.it)


## 🚀 Contact Me
- [Email](mailto:seyedahmad.mirsalar2@unibo.it)
- [LinkedIn](https://www.linkedin.com/in/ahmad-mirsalari/)
- [Twitter](https://twitter.com/ahmad_mirsalari)
- [APROPOS](https://projects.tuni.fi/apropos/news/pr_esr_3/)
