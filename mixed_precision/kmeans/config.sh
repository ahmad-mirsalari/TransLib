PYTHON_BIN=python3
PY_SCRIPT=./data_generator.py
MAKE_BIN=make
MAKE_TARGETS="clean all run"
LOG_DIR="./logs"
RUNS=(
# --------------------------------------- MAC FLAG = false --------------------------------------
# --------------------------------- Cores = 1 ---------------------------------
#-------------- Single precision
# fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1"
# fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1"
# fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1"
# fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32 --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1 no_fmadd=1"
# fp8 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=1 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 vec=1 print_results=1 no_fmadd=1"
# fp16 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=1 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 vec=1 print_results=1 no_fmadd=1"
# fp16alt + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=1 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT vec=1 print_results=1 no_fmadd=1"
# --------------Mixed precision
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=1 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#--------------Mixed precision with hwmixed
# fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=1 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# --------------------------------- Cores = 4 ---------------------------------
#-------------- Single precision
# fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP8 print_results=1 no_fmadd=1"
# fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16 print_results=1 no_fmadd=1"
# fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16ALT print_results=1 no_fmadd=1"
# fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32 --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP32 print_results=1 no_fmadd=1"
# fp8 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=4 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP8 vec=1 print_results=1 no_fmadd=1"
# fp16 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=4 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16 vec=1 print_results=1 no_fmadd=1"
# fp16alt + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=4 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16ALT vec=1 print_results=1 no_fmadd=1"
#--------------Mixed precision
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=4 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP32 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#--------------Mixed precision with hwmixed
# fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=4 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP32 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"

# --------------------------------- Cores = 8 ---------------------------------
# --------------Single precision
# fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP8 print_results=1 no_fmadd=1"
# fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 print_results=1 no_fmadd=1"
# fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT print_results=1 no_fmadd=1"
# fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32 --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP32 print_results=1 no_fmadd=1"
# fp8 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=8 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP8 vec=1 print_results=1 no_fmadd=1"
# fp16 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=8 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 vec=1 print_results=1 no_fmadd=1"
# fp16alt + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=8 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT vec=1 print_results=1 no_fmadd=1"
# --------------Mixed precision
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=8 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP32 fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
# --------------Mixed precision with hwmixed
# fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1 no_fmadd=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=8 --mac_flag=false --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP32 fmt_OUT=FP16ALT hwmixed=1 print_results=1 no_fmadd=1"

# -------------------------------------- MAC FLAG = true --------------------------------------
# --------------------------------- Cores = 1 ---------------------------------
#-------------- Single precision
# fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1"
# fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1"
# fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1"
# fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32 --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1"
# fp8 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=1 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 vec=1 print_results=1"
# fp16 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=1 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 vec=1 print_results=1"
# fp16alt + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=1 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT vec=1 print_results=1"
#--------------Mixed precision
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16 print_results=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16ALT print_results=1"
#fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP32 print_results=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP8 print_results=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP16ALT print_results=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP32 print_results=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP8 print_results=1"
#fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP16 print_results=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=1 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_OUT=FP16ALT print_results=1"
#--------------Mixed precision with hwmixed
# fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=1 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# --------------------------------- Cores = 4 ---------------------------------
#-------------- Single precision
# fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP8 print_results=1"
# fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16 print_results=1"
# fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16ALT print_results=1"
# fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32 --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP32 print_results=1"
# fp8 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=4 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP8 vec=1 print_results=1"
# fp16 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=4 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16 vec=1 print_results=1"
# fp16alt + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=4 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt=FP16ALT vec=1 print_results=1"
#--------------Mixed precision
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16 print_results=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16ALT print_results=1"
#fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP32 print_results=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP8 print_results=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP16ALT print_results=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP32 print_results=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP8 print_results=1"
#fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP16 print_results=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=4 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP32 fmt_OUT=FP16ALT print_results=1"
#--------------Mixed precision with hwmixed
# fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=4 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=4 fmt_INP=FP32 fmt_OUT=FP16ALT hwmixed=1 print_results=1"

# --------------------------------- Cores = 8 ---------------------------------
# --------------Single precision
# fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP8 print_results=1"
# fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 print_results=1"
# fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT print_results=1"
# fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32 --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP32 print_results=1"
# fp8 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM --cores=8 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP8 vec=1 print_results=1"
# fp16 + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16 --cores=8 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16 vec=1 print_results=1"
# fp16alt + VECT
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT --cores=8 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt=FP16ALT vec=1 print_results=1"
# --------------Mixed precision
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16 print_results=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16ALT print_results=1"
#fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP32 print_results=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP8 print_results=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP16ALT print_results=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP32 print_results=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP8 print_results=1"
#fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP16 print_results=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=8 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP32 fmt_OUT=FP16ALT print_results=1"
# --------------Mixed precision with hwmixed
# fp8 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP32 --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp8 + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16 --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp8 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP8_CUSTOM,FP16ALT --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP8 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16 + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP32 --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP32 hwmixed=1 print_results=1"
# fp16 + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP8_CUSTOM --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16 + fp16alt
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16,FP16ALT --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
# fp16alt + fp8
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP8_CUSTOM --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP8 hwmixed=1 print_results=1"
# fp16alt + fp16
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP16ALT,FP16 --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP16ALT fmt_OUT=FP16 hwmixed=1 print_results=1"
# fp16alt + fp32
"PY: --input_size=32 --features=33 --num_clusters=9 --float_type=FP32,FP16ALT --cores=8 --mac_flag=true --hwmixed=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=8 fmt_INP=FP32 fmt_OUT=FP16ALT hwmixed=1 print_results=1"
)