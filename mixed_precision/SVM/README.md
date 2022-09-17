# SVM test
This test performs an SVM (Support Vector Machine) classification.
In this folder you can find pre-generated golden models for FP32 and FP16 formats.

## Running a test
After the platform and the SDK setup you can run the test:

~~~~~shell
make clean all [platform=rtl] run
~~~~~

If you want to run this test on RTL, remember to specify the platform which is gvsoc by default.
There are several flags useful to activate some functionalities:

- `cores=N_CORES` set the number of cores used for the execution to `N_CORES`, by default `cores=1`
- `fmt=FP_FMT` specifies the floating-point format for data, by deafult it is set to `FP32` but you can also choose `FP16` format
- `vec=1` activates vectorial format **only for half-precision floating point (FP16)**
- `check=1` activates results checking
- `verbose=1` activates wrong results printing

** Missing FP16ALT support **

** [TODO]: Golden model generator script **