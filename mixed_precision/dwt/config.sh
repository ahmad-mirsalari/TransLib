PYTHON_BIN=python3
PY_SCRIPT=./data_generator.py
MAKE_BIN=make
MAKE_TARGETS="clean all run"
LOG_DIR="./logs"
RUNS=(
# -------------------------------------- MAC FLAG = false --------------------------------------
# ------------------- Mixed precision vectorized
# fp16,fp16,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 vec=1 print_results=1 no_fmadd=1"
# fp16,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT vec=1 print_results=1 no_fmadd=1"
#fp16,fp16,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 vec=1 print_results=1 no_fmadd=1"
# fp16alt,fp16alt,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 vec=1 print_results=1 no_fmadd=1"
# fp16alt,fp16alt,fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 vec=1 print_results=1 no_fmadd=1"
# fp16alt, fp16alt, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 vec=1 print_results=1 no_fmadd=1"

# -----------------------Fixed precision
# fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP8 print_results=1 no_fmadd=1"
# fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 print_results=1 no_fmadd=1"
# fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT print_results=1 no_fmadd=1"
# fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP32 print_results=1 no_fmadd=1"
# fp16 + VECT
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 vec=1 print_results=1 no_fmadd=1"
# fp16alt + VECT
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT vec=1 print_results=1 no_fmadd=1"

# ---------------------- Mixed precision
# fp8 + fp8 + fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp8 + fp8 + fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#fp8  + fp8 + fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp8,fp16,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp8,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16,FP16ALT --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
# fp8,fp16,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
# fp8,fp16alt,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16ALT,FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp8,fp16alt,fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16ALT,FP16 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp8,fp16alt,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
# fp16, fp8, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16, fp8, fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
# fp16, fp8, fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP8_CUSTOM,FP8_CUSTOM --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP8 fmt_OUT=FP8 print_results=1 no_fmadd=1"
# fp16 + fp16 + fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
# fp16,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
# fp16,fp16,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16alt, fp16alt + fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt, fp16alt, fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp16alt,fp16alt,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1"

# ------------------ Mixed precision with hwmixed
# fp8, fp8, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp8, fp8, fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp8, fp8, fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16,fp16,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP32 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp16,fp116,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP16ALT --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt,fp16alt,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt,fp16alt,fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt, fp16alt, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"

# -------------------------------------- MAC FLAG = true --------------------------------------

# -----------------------Fixed precision
# fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP8 print_results=1"
# fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 print_results=1"
# fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT print_results=1"
# fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP32 print_results=1"
# fp16 + VECT
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 vec=1 print_results=1"
# fp16alt + VECT
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT vec=1 print_results=1"
# ----------------------- Mixed precision
# fp8 + fp8 + fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1"
# fp8 + fp8 + fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1"
#fp8  + fp8 + fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1"
# fp8,fp16,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1"
# fp8,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16,FP16ALT --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1"
# fp8,fp16,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1"
# fp8,fp16alt,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16ALT,FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1"
# fp8,fp16alt,fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16ALT,FP16 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1"
# fp8,fp16alt,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1"
# fp16, fp8, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1"
# fp16, fp8, fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1"
# fp16, fp8, fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP8_CUSTOM,FP8_CUSTOM --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP8 fmt_OUT=FP8 print_results=1"
# fp16 + fp16 + fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1"
# fp16,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1"
# fp16,fp16,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1"
# fp16alt, fp16alt + fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1"
#fp16alt, fp16alt, fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1"
# fp16alt,fp16alt,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1"

# ----------------------- Mixed precision with hwmixed
# fp8, fp8, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp8, fp8, fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp8, fp8, fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16,fp16,fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP32 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp16,fp116,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP16ALT --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16alt,fp16alt,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16alt,fp16alt,fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp16alt, fp16alt, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 hwmixed=1 print_results=1"

# ---------------- Mixed precision vectorized
# fp16,fp16,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 vec=1 print_results=1"
# fp16,fp16,fp16alt
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT vec=1 print_results=1"
# fp16alt,fp16alt,fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 vec=1 print_results=1"
# fp16alt,fp16alt,fp16
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 vec=1 print_results=1"
# fp16alt, fp16alt, fp32
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 vec=1 print_results=1"
# fp16alt, fp16alt, fp8
"PY: --input_size=128 --levels=4 --mode=sym4 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 vec=1 print_results=1"

)