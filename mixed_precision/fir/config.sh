PYTHON_BIN=python3
PY_SCRIPT=./data_generator.py
MAKE_BIN=make
MAKE_TARGETS="clean all run"

LOG_DIR="./logs"

RUNS=(
#---------------------------------------------- With mac flag ------------------------------------------
#-------------------------Fixed precision
#fp32 not reversed
"PY: --length=512 --order=32 --float_type=FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1"
# fp8 not reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1"
#fp8 vectorized not reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 vec=1"
#fp8 vectorized reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 vec=1"
# fp16 not reversed
"PY: --length=512 --order=32 --float_type=FP16 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1"
#fp16 vectorized not reversed
"PY: --length=512 --order=32 --float_type=FP16 --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 vec=1"
#fp16 vectorized reversed
"PY: --length=512 --order=32 --float_type=FP16 --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 vec=1"
#fp16alt not reversed
"PY: --length=512 --order=32 --float_type=FP16ALT --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1"
#fp16alt vectorized not reversed
"PY: --length=512 --order=32 --float_type=FP16ALT --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 vec=1"
#fp16alt vectorized reversed
"PY: --length=512 --order=32 --float_type=FP16ALT --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 vec=1"

#--------------------------Mixed precision cases
#fp8,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1"
#fp8,fp8,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1"
#fp8,fp8,fp16alt
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1"
#fp8,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1"
#fp8,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1"
#fp8,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1"
#fp16alt,fp8,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP8_CUSTOM,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP8 print_results=1"
#fp16alt,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1"
#fp16alt,fp16alt,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1"
#fp16,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1"
#fp8,fp16,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16 print_results=1"
#fp16,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1"
#fp16alt,fp16alt,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1"
#fp8,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP32 fmt_OUT=FP32 print_results=1"
#fp32,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1"
#fp16,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 print_results=1"
#fp16alt,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP32,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP32 fmt_OUT=FP32 print_results=1"
#fp16,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 print_results=1"
#fp32,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP32,FP16,FP32 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1"
#fp16,fp32,fp16
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP16 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP16 print_results=1"
#fp32,fp32,fp16
"PY: --length=512 --order=32 --float_type=FP32,FP32,FP16 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP32 fmt_OUT=FP16 print_results=1"
#fp16alt,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16 fmt_OUT=FP8 print_results=1"
#fp16,fp32,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP8 print_results=1"
#fp16alt,fp32,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP32,FP16 --mac_flag=true --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP32 fmt_OUT=FP16 print_results=1"

#---------------------Mixed precision cases with vectorization
#fp8,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 vec=1"
#fp8,fp8,fp32 reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 vec=1"
#fp8,fp8,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 vec=1"
#fp8,fp8,fp16 reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 vec=1"
#fp8,fp8,fp16alt
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 vec=1"
#fp8,fp8,fp16alt reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 vec=1"

#fp16alt,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 vec=1"
#fp16alt,fp16alt,fp8 reversed
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 vec=1"
#fp16alt,fp16alt,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 vec=1"
#fp16alt,fp16alt,fp16 reversed
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 vec=1"
#fp16alt,fp16alt,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 vec=1"
#fp16alt,fp16alt,fp32 reversed
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 vec=1"

#fp16,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 vec=1"
#fp16,fp16,fp32 reversed
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 vec=1"
#fp16,fp16,fp16alt
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 vec=1"
#fp16,fp16,fp16alt reversed
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 vec=1"
#fp16,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 vec=1"
#fp16,fp16,fp8 reversed
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 vec=1"

#--------------------Mixed precision with Hw mixed flag
#fp16alt,fp16alt,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 hwmixed=1"
#fp16alt,fp16alt,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 hwmixed=1"
#fp16alt,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 hwmixed=1"
#fp8,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 hwmixed=1"
#fp8,fp8,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 hwmixed=1"
#fp8,fp8,fp16alt
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 hwmixed=1"
#fp16,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 hwmixed=1"
#fp16,fp16,fp16alt
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 hwmixed=1"
#fp16,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 hwmixed=1"

#---------------------------------------------------- Without mac flag and with no_fmadd=1 in the C code

#-------------------------------------Fixed precision
#fp32 not reversed
"PY: --length=512 --order=32 --float_type=FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1 no_fmadd=1 no_fmadd=1"
# fp8 not reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1 no_fmadd=1"
#fp8 vectorized not reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1 no_fmadd=1 vec=1"
#fp8 vectorized reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1 vec=1"
# fp16 not reversed
"PY: --length=512 --order=32 --float_type=FP16 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1"
#fp16 vectorized not reversed
"PY: --length=512 --order=32 --float_type=FP16 --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1 vec=1"
#fp16 vectorized reversed
"PY: --length=512 --order=32 --float_type=FP16 --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1 vec=1"
#fp16alt not reversed
"PY: --length=512 --order=32 --float_type=FP16ALT --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1"
#fp16alt vectorized not reversed
"PY: --length=512 --order=32 --float_type=FP16ALT --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1 vec=1"
#fp16alt vectorized reversed
"PY: --length=512 --order=32 --float_type=FP16ALT --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1 vec=1"

#-------------------------------------Mixed precision cases
#fp8,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp8,fp8,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1"
#fp8,fp8,fp16alt
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#fp8,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp8,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp8,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt,fp8,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP8_CUSTOM,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt,fp16alt,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1"
#fp16,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp8,fp16,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16 print_results=1 no_fmadd=1"
#fp16,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16alt,fp16alt,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp8,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp32,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16alt,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP32,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16,fp32,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp32,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP32,FP16,FP32 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16,fp32,fp16
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP16 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP16 print_results=1 no_fmadd=1"
#fp32,fp32,fp16
"PY: --length=512 --order=32 --float_type=FP32,FP32,FP16 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP32 fmt_OUT=FP16 print_results=1 no_fmadd=1"
#fp16alt,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16,fp32,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP32,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt,fp32,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP32,FP16 --mac_flag=false --vec_flag=false --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP32 fmt_OUT=FP16 print_results=1 no_fmadd=1"

#-------------------------------------Mixed precision cases with vectorization
#fp8,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
#fp8,fp8,fp32 reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
#fp8,fp8,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1 vec=1"
#fp8,fp8,fp16 reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1 vec=1"
#fp8,fp8,fp16alt
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 vec=1"
#fp8,fp8,fp16alt reversed
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 vec=1"

#fp16alt,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 vec=1"
#fp16alt,fp16alt,fp8 reversed
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 vec=1"
#fp16alt,fp16alt,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 vec=1"
#fp16alt,fp16alt,fp16 reversed
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 vec=1"
#fp16alt,fp16alt,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
#fp16alt,fp16alt,fp32 reversed
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"

#fp16,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
#fp16,fp16,fp32 reversed
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
#fp16,fp16,fp16alt
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 vec=1"
#fp16,fp16,fp16alt reversed
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 vec=1"
#fp16,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=true --reversed=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 vec=1"
#fp16,fp16,fp8 reversed
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=true --reversed=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 vec=1"

#---------------------mixed precision with Hw mixed flag
#fp16alt,fp16alt,fp32
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 hwmixed=1"
#fp16alt,fp16alt,fp16
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 hwmixed=1"
#fp16alt,fp16alt,fp8
"PY: --length=512 --order=32 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 hwmixed=1"
#fp8,fp8,fp32
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 hwmixed=1"
#fp8,fp8,fp16
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1 hwmixed=1"
#fp8,fp8,fp16alt
"PY: --length=512 --order=32 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 hwmixed=1"
#fp16,fp16,fp32
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 hwmixed=1"
#fp16,fp16,fp16alt
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 hwmixed=1"
#fp16,fp16,fp8
"PY: --length=512 --order=32 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --reversed=false --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 hwmixed=1"
)