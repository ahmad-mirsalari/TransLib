# FIR filter test
This test performs signal processing through a FIR (Finite Impulse Response) filter.
In this folder you can find pre-generated golden models.

## Running a test
After the platform and the SDK setup you can run the test:

~~~~~shell
make clean all [platform=rtl] run
~~~~~

If you want to run this test on RTL, remeber to specify the platform which is gvsoc by default.
There are several flags useful to activate some functionalities:

- `cores=N_CORES` set the number of cores used for the execution to `N_CORES`, by default `cores=1`. There is also the ability to run on the Fabric controller by using `FABRIC=1` instead of `cores=N_CORE`.
- `fmt=FP_FMT` specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` and `FP16ALT` formats. **For this application you can use mixed-precision in the C code by using `fmt_INP=FP_INP fmt_FIL=FP_FIL fmt_OUT=FP_OUT` instead of `fmt`.**
- `vec=1` activates vectorial format **only for half-precision floating point (FP16 and FP16ALT)**
- `check=1` activates results checking
- `verbose=1` activates wrong results printing
- `PRINT_RESULTS=1` print outputs of C code
- `stats=1` activates performance measurement


## Generating the golden model
If you want to re-generate a golden model, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --LENGTH=length --ORDER=order --float_type=fmt --MAC_flag=MAC_FLAG --vec_flag=VEC_FLAG
~~~~~

- specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` and `FP16ALT` formats. **Also, you can run the mixed-precision golden model by using `--float_type=FP_INP,FP_FIL,FP_OUT` (input,filter,output).**
- `MAC_flag` is used to emulate the multiply-and-add operator available on most DSP instruction sets for embedded devices. It can be true or false. To emulate `FP16` and `FP16ALT` behavior on PULP, true this flag.
- `vec_flag` to emulate SIMD vector instructions. It can be true or false. To emulate vectorized `FP16` and `FP16ALT` behavior on PULP, true this flag.
The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT):
