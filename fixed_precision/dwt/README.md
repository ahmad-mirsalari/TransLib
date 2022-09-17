# DWT test
This test performs a DWT (Discrete Wavelet Transformation).
In this folder you can find golden model and optimized c code for FP32, FP16, and FP16ALT formats.
The C code only supports fixed-precision.

## Running a test
After the platform and the SDK setup you can run the test:

~~~~~shell
make clean all [platform=rtl] run
~~~~~

If you want to run this test on RTL, remember to specify the platform which is gvsoc by default.
There are several flags useful to activate some functionalities:

- `cores=N_CORES` set the number of cores used for the execution to `N_CORES`, by default `cores=1`
- `fmt=FP_FMT` specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` and `FP16ALT` formats
- `vec=1` activates vectorial format  for FP16 and FP16ALT
- `check=1` activates results checking
- `verbose=1` activates wrong results printing
- `PRINT_RESULTS=1` print outputs of C code
- `stats=1` activates performance measurement


## Running the golden model
If you want to run the golden model and re-generate data, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --input_size=INPUT_SIZE --levels=LEVELS --mode=MODE --float_type=FP** --MAC_flag=MAC_FLAG --vec_flag=VEC_FLAG
~~~~~
- to define  --'mode' check https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html 
- specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` and `FP16ALT` formats. Also, you can run the mixed-precision golden model by using --float_type=FP**,FP**,FP** (input, filter, output).
- MAC_flag is used to emulate the multiply-and-add operator available on most DSP instruction sets for embedded devices. It can be true or false. To emulate FP16 and FP16ALT behavior on PULP, true this flag.
- vector flag to emulate SIMD vector instructions. It can be true or false. To emulate vectorized FP16 and FP16ALT behaviour on PULP, true this flag.

The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT):
