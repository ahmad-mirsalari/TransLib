# MatMul test
This test performs a matrix multiplication on FP32/FP16/FP16ALT data and can also be used to measure performances.
In this folder you can find pre-generated golden models for all the supported floating-point formats.

## Running a test
After the platform and the SDK setup you can run the test:

~~~~~shell
make clean all [platform=rtl] run
~~~~~

If you want to run this test on RTL, remember to specify the platform which is gvsoc by default.
There are several flags useful to activate some functionalities:

- `cores=N_CORES` set the number of cores used for the execution to `N_CORES`, by default `cores=1`
- `fmt=FP_FMT` specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` or `FP16ALT` formats
- `vec=1` activates vectorial format **only for half-precision floating point (FP16 and FP16ALT)**
- `check=1` activates the result check
- `verbose=1` prints the wrong results
- `stats=1` activates performance measurement

## Generating the golden model
If you want to re-generate a golden model, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --M=m --N=n --P=p --float_type=fmt
~~~~~

The script will generate three floating-point matrices fo format `fmt` (FP32/FP16/FP16ALT):
- A_mat[mxn] input matrix
- B_mat[nxp] input matrix
- ref[mxp] output matrix

The generated header file will be written in the [references](./references) folder.