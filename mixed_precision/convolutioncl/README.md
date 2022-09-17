# Convolution test
This test performs a convolution bewteen a ones matrix and FP32/FP16/FP16ALT filter matrix and can also be used to measure performances.
In this folder you can find pre-generated golden models.

## Running a test
After the platform and the SDK setup you can run the test:

~~~~~shell
make clean all [platform=rtl] run
~~~~~

If you want to run this test on RTL, remember to specify the platform which is gvsoc by default.
There are several flags useful to activate some functionalities:

- `cores=N_CORES` set the number of cores used for the execution to `N_CORES`, by default `cores=1`
- `fmt=FP_FMT` specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` or `FP16ALT` formats. **For this application you can use mixed-precision in the C code by using fmt_INP=FP_FMT_INPU fmt_FIL=FP_FMT_FILTER fmt_OUT=FP_FMT_OUT.**
- `vec=1` activates vectorial format **only for half-precision floating point (FP16 and FP16ALT)**
- `check=1` activates the result check
- `verbose=1` prints the wrong results
- `stats=1` activates performance measurement
- `PRINT_RESULTS=1` print outputs of C code

- ** for Vectorization, there are only optimized C functions for FILT_WIN = 3 and FILT_WIN = 5. But you can easily copy one of this functions and make some minor changes to supprt your desired filter size. It should be mentioned that for non-vectorized version there is no restriction on the Filter size. **

## Generating the golden model
If you want to re-generate a golden model, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --IMG_WIDTH=length --FILT_WIN=fw --STRIDE=stride --PADDING=padding --float_type=fmt --MAC_flag=MAC_FLAG --vec_flag=VEC_FLAG
~~~~~
- padding could be 'same' or 'valid'

- specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` and `FP16ALT` formats. Also, you can run the mixed-precision golden model by using --float_type=FP**,FP**,FP** (input,filter,output).
- MAC_flag is used to emulate the multiply-and-add operator available on most DSP instruction sets for embedded devices. It can be true or false. To emulate FP16 and FP16ALT behavior on PULP, true this flag.
- vector flag to emulate SIMD vector instructions. It can be true or false. To emulate vectorized FP16 and FP16ALT behavior on PULP, true this flag.
The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT):

