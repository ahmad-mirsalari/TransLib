# FFT test
This test performs a FFT (Fast Fourier Tranform).The C code only supports fixed-precision.


~~~~~shell
cd /fft_mixed_2_8/
make clean all [platform=rtl] run
~~~~~

If you want to run this test on RTL, remember to specify the platform which is gvsoc by default.
There are several flags useful to activate some functionalities:

- `CORES=N_CORES` set the number of cores used for the execution to `N_CORES`, by default `cores=1`
- `fmt=FP_FMT` specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` and `FP16ALT` format
- `vec=1` activates vectorial format **only for half-precision floating point (FP16 and FP16ALT)**
- `check=1` activates results checking
- `verbose=1` activates wrong results printing
- `PRINT_RESULTS=1` print outputs of C code
- `stats=1` activates performance measurement


## Running the golden model
If you want to run the golden model and re-generate data, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --input_size=INPUT_SIZE --float_type=FP** --MAC_flag=MAC_FLAG --vec_flag=VEC_FLAG
~~~~~
- specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` and `FP16ALT` formats. Also, you can run the mixed-precision golden model by using --float_type=FP**,FP**,FP** (input, twiddle, output).
- MAC_flag is used to emulate the multiply-and-add operator available on most DSP instruction sets for embedded devices. It can be true or false. To emulate FP16 and FP16ALT behavior  on PULP, true this flag.
- vector flag to emulate SIMD vector instructions. It can be true or false. To emulate vectorized FP16 and FP16ALT behaviour on PULP, true this flag.
It should be mentioned that for half-precision floating point (FP16 and FP16ALT) there is a small error based on some optimizations in the C code.
The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP16ALT):
