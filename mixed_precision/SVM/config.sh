PYTHON_BIN=python3
PY_SCRIPT=./data_generator.py
MAKE_BIN=make
MAKE_TARGETS="clean all run"

LOG_DIR="./logs"

RUNS=(
# ------------------------------------------------- LINEAR KERNEL -------------------------------------------------

#---------------- Fixed precision
#FP32  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 "
#FP8_CUSTOM  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 "
#FP8_CUSTOM vectorized
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8  vec=1"
#FP16  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 "
#FP16 vectorized  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16 --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16  vec=1"
#FP16ALT  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT "
#FP16ALT vectorized  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16ALT --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT  vec=1"

#------------------ Mixed precision
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP8_CUSTOM,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP8_CUSTOM,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 "
#FP8_CUSTOM,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP8_CUSTOM,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 "
#FP16,FP16ALT,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16ALT,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP16ALT "
#FP16ALT,FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP8_CUSTOM,FP16,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 "

#-------------------- Hw mixed precision cases with linear kernel
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 hwmixed=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 hwmixed=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT hwmixed=1 "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 hwmixed=1 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 hwmixed=1 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 hwmixed=1 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 hwmixed=1 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT hwmixed=1 "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 hwmixed=1 "

#----------------------- Mixed precision with vectorization
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 vec=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 vec=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT vec=1 "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 vec=1 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 vec=1 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 vec=1 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 vec=1 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT vec=1 "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 vec=1 "


# ------------------------------------------------- RBF KERNEL -------------------------------------------------

#--------------------Fixed precision
#FP32  
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 "
#FP8_CUSTOM  
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 "
#FP8_CUSTOM vectorized 
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8  vec=1"
#FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 "
#FP16 vectorized
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16 --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16  vec=1"
#FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT "
#FP16ALT vectorized
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT  vec=1"

#--------------------- Mixed precision
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP8_CUSTOM,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP8_CUSTOM,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 "
#FP8_CUSTOM,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP8_CUSTOM,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 "
#FP16,FP16ALT,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16ALT,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP16ALT "
#FP16ALT,FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP8_CUSTOM,FP16,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 "
#FP8_CUSTOM,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP32 fmt_OUT=FP32 "
#FP32,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP8 fmt_OUT=FP32 "
#FP16,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP32,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 "
#FP16ALT,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP32,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP32 fmt_OUT=FP32 "
#FP16,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP32,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 "
#FP32,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32,FP16,FP32 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP16,FP32,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP32,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP16 "
#FP32,FP32,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32,FP32,FP16 --mac_flag=true --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP32 fmt_FIL=FP32 fmt_OUT=FP16 "

#--------------------- Mixed precision with vectorization
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32  vec=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16  vec=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT  vec=1"
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8  vec=1"
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16  vec=1"
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32  vec=1"
# #FP16,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32  vec=1"
#FP16,FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT  vec=1"
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8  vec=1"


#----------------------- Mixed precision with Hw mixed flag
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false  --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32  hwmixed=1"
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false  --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16  hwmixed=1"
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false  --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8  hwmixed=1"
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32  hwmixed=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16  hwmixed=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT  hwmixed=1"
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32  hwmixed=1"
#FP16,FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT  hwmixed=1"
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8  hwmixed=1"
# ------------------------------------------------- LINEAR KERNEL NO MAC FLAG -------------------------------------------------

#---------------- Fixed precision
#FP32  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP32 "
#FP8_CUSTOM  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP8 "
#FP8_CUSTOM vectorized
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP8  vec=1"
#FP16  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16 "
#FP16 vectorized  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16 --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16  vec=1"
#FP16ALT  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16ALT "
#FP16ALT vectorized  
"PY: --input_size=180 --kernel=linear --dataset=cancer  --float_type=FP16ALT --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16ALT  vec=1"

#------------------ Mixed precision
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP8_CUSTOM,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP8_CUSTOM,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 "
#FP8_CUSTOM,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP8_CUSTOM,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 "
#FP16,FP16ALT,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16ALT,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP16ALT "
#FP16ALT,FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP8_CUSTOM,FP16,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 "

#-------------------- Hw mixed precision
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 hwmixed=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 hwmixed=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT hwmixed=1 "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 hwmixed=1 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 hwmixed=1 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 hwmixed=1 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 hwmixed=1 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT hwmixed=1 "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 hwmixed=1 "

#----------------------- Mixed precision with vectorization
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 vec=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 vec=1 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT vec=1 "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 vec=1 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 vec=1 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 vec=1 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 vec=1 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT vec=1 "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=linear --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 vec=1 "
# ------------------------------------------------- RBF KERNEL WITH NO MAC FLAG-------------------------------------------------

#--------------------Fixed precision
#FP32  
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP32 "
#FP8_CUSTOM  
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP8 "
#FP8_CUSTOM vectorized 
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP8  vec=1"
#FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16 "
#FP16 vectorized
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16 --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16  vec=1"
#FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16ALT "
#FP16ALT vectorized
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt=FP16ALT  vec=1"

#--------------------- Mixed precision
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32 "
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16 "
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP8_CUSTOM,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP8_CUSTOM,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP8 "
#FP8_CUSTOM,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP8_CUSTOM,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP8 "
#FP16ALT,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP8 fmt_OUT=FP16ALT "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 "
#FP16,FP16ALT,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16ALT,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP16ALT "
#FP16ALT,FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP8_CUSTOM,FP16,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP16 fmt_OUT=FP16 "
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP16, FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT "
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16 "
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8 "
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32 "
#FP8_CUSTOM,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP32 fmt_OUT=FP32 "
#FP32,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP32 fmt_FIL=FP8 fmt_OUT=FP32 "
#FP16,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP32,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 "
#FP16ALT,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP32,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP32 fmt_OUT=FP32 "
#FP16,FP32,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP32,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP32 "
#FP32,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32,FP16,FP32 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP32 fmt_FIL=FP16 fmt_OUT=FP32 "
#FP16,FP32,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP32,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP32 fmt_OUT=FP16 "
#FP32,FP32,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP32,FP32,FP16 --mac_flag=false --vec_flag=false  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP32 fmt_FIL=FP32 fmt_OUT=FP16 "

#--------------------- Mixed precision with vectorization
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32  vec=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16  vec=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT  vec=1"
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8  vec=1"
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16  vec=1"
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32  vec=1"
# #FP16,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32  vec=1"
#FP16,FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT  vec=1"
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer  --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=true  ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8  vec=1"


#----------------------- Mixed precision with Hw mixed flag
#FP16ALT,FP16ALT,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false  --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP32  hwmixed=1"
#FP16ALT,FP16ALT,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false  --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP16  hwmixed=1"
#FP16ALT,FP16ALT,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false  --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16ALT fmt_FIL=FP16ALT fmt_OUT=FP8  hwmixed=1"
#FP8_CUSTOM,FP8_CUSTOM,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP32  hwmixed=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16  hwmixed=1"
#FP8_CUSTOM,FP8_CUSTOM,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP8 fmt_FIL=FP8 fmt_OUT=FP16ALT  hwmixed=1"
#FP16,FP16,FP32
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP32  hwmixed=1"
#FP16,FP16,FP16ALT
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP16ALT  hwmixed=1"
#FP16,FP16,FP8_CUSTOM
"PY: --input_size=180 --kernel=rbf --dataset=cancer --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false  --hwmixed_flag=true; MAKE: check=1 verbose=1 stats=1 no_fmadd=1 cores=1 fmt_INP=FP16 fmt_FIL=FP16 fmt_OUT=FP8  hwmixed=1"
)