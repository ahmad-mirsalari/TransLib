# FIR filter test
This test performs signal processing through a FIR (Finite Impulse Response) filter.
In this folder you can find pre-generated golden models for FP32, FP16 and FP16ALT formats.

## Running a test
After the platform and the SDK setup you can run the test:

~~~~~shell
make clean all [platform=rtl] run
~~~~~

If you want to run this test on RTL, remeber to specify the platform which is gvsoc by default.
There are several flags useful to activate some functionalities:

- `cores=N_CORES` set the number of cores used for the execution to `N_CORES`, by default `cores=1`
- `fmt=FP_FMT` specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` `FP16ALT` format
- `vec=1` activates vectorial format **only for half-precision floating point (FP16 and FP16ALT)**
- `check=1` activates results checking
- `verbose=1` activates wrong results printing


## Generating the golden model
If you want to re-generate a golden model, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --LENGTH=length --ORDER=order --float_type=fmt
~~~~~

The script will generate three floating-point array of format `fmt` (FP32/FP16/FP16ALT) in fp'fmt'_ref.h:
- UnitImpulse[length] input array
- Filter0[order] filter array
- Buffer0[length-order] output array

And generate a floating-point array as reference output in out'fmt'_ref.h
- check[length-order] reference output

The generated header file will be written in the [references](./references) folder.

** WARNING: don't deactivate statistics!! ** Without statistics on, the correct functional behaviour is not guaranteed.
