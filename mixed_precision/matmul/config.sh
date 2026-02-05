PYTHON_BIN=python3
PY_SCRIPT=./data_generator.py
MAKE_BIN=make
MAKE_TARGETS="clean all run"
LOG_DIR="./logs"
RUNS=(
# --------------------------------------------------------mac flag = true
# -------------- Single precision
# fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1"
#fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 transpose=1"
# fp16
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1"
# fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 transpose=1"
# fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1"
# fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 transpose=1"
# fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1"
# fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1 transpose=1"
# fp8 with vectorization
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 vec=1"
# fp8 with vectorization and transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 vec=1 transpose=1"
# fp16 with vectorization
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 vec=1"
# fp16 with vectorization and transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 vec=1 transpose=1"
# fp16alt with vectorization
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 vec=1"
# fp16alt with vectorization and transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 vec=1 transpose=1"


# --------------Mixed precision
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1"
#FP8,fp16,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP8 print_results=1"
#FP8,fp16,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP8 print_results=1 transpose=1"
#FP8,fp16alt,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16ALT fmt_OUT=FP8 print_results=1"
#FP8,fp16alt,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 transpose=1"
#FP8, fp16,fp16
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP16 print_results=1"
#FP8, fp16,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP16 print_results=1 transpose=1"
#FP8,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP32 print_results=1"
#FP8,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP32 print_results=1 transpose=1"
#fp16,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1"
#fp16,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 transpose=1"
#fp16alt,fp16alt,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1"
#fp16,fp16alt,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP16 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 transpose=1"
#fp16alt,fp16,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16 fmt_OUT=FP8 print_results=1"
#fp16alt,fp16,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16 fmt_OUT=FP8 print_results=1 transpose=1"
#fp16,fp16alt,fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP16ALT --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP16ALT print_results=1"
#fp16,fp16alt,fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP16ALT --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP16ALT print_results=1 transpose=1"
#fp16,fp16alt,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 transpose=1"
#fp8,fp32,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP32 print_results=1"
# fp8,fp32,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP32 print_results=1 transpose=1"
#fp32,fp8,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP8 fmt_OUT=FP32 print_results=1"
#fp32,fp8,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP8 fmt_OUT=FP32 print_results=1 transpose=1"
#fp16,fp32,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP32 print_results=1"
#fp16,fp32,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP32 print_results=1 transpose=1"
#fp32,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16 fmt_OUT=FP32 print_results=1"
#fp32,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16 fmt_OUT=FP32 print_results=1 transpose=1"
#fp16alt,fp32,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP32,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP32 fmt_OUT=FP32 print_results=1"
#fp16alt,fp32,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP32,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP32 fmt_OUT=FP32 print_results=1 transpose=1"
#fp32,fp16alt,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16ALT,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16ALT fmt_OUT=FP32 print_results=1"
#fp32,fp16alt,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16ALT,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 transpose=1"
#fp8,fp32,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP8 print_results=1"
#fp8,fp32,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP8 print_results=1 transpose=1"
#fp16,fp32,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP8 print_results=1"
#fp16,fp32,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP8 print_results=1 transpose=1"

# --------------Mixed precision with HW mixed
#FP8,FP8,FP32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 hwmixed=1"
#FP8,FP8,FP32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 transpose=1 hwmixed=1"
#FP8,FP8,FP16
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 hwmixed=1"
#FP8,FP8,FP16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 transpose=1 hwmixed=1"
#FP8,FP8,FP16ALT
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 hwmixed=1"
#FP8,FP8,FP16ALT with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 transpose=1 hwmixed=1"
#FP16,FP16,FP32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 hwmixed=1"
#FP16,FP16,FP32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 transpose=1 hwmixed=1"
#FP16,FP16,FP8
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 hwmixed=1"
#FP16,FP16,FP8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 transpose=1 hwmixed=1"
#FP16,FP16,FP16ALT
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 hwmixed=1"
#FP16,FP16,FP16ALT with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 transpose=1 hwmixed=1"
#FP16ALT,FP16ALT,FP32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 hwmixed=1"
#FP16ALT,FP16ALT,FP32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 transpose=1 hwmixed=1"
#FP16ALT,FP16ALT,FP8
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 hwmixed=1"
#FP16ALT,FP16ALT,FP8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 transpose=1 hwmixed=1"
#FP16ALT,FP16ALT,FP16
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 hwmixed=1"
#FP16ALT,FP16ALT,FP16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 transpose=1 hwmixed=1"
#--------------Vectorized mixed precision
#FP8,FP8,FP32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 vec=1"
#FP8,FP8,FP32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 transpose=1 vec=1"
#FP8,FP8,FP16
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 vec=1"
#FP8,FP8,FP16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 transpose=1 vec=1"
#FP8,FP8,FP16ALT
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 vec=1"
#FP8,FP8,FP16ALT with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 transpose=1 vec=1"
#FP16,FP16,FP32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 vec=1"
#FP16,FP16,FP32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 transpose=1 vec=1"
#FP16,FP16,FP8
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 vec=1"
#FP16,FP16,FP8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 transpose=1 vec=1"
#FP16ALT,FP16ALT,FP8
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 vec=1"
#FP16ALT,FP16ALT,FP8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 transpose=1 vec=1"
#FP16,FP16,FP16ALT
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 vec=1"
#FP16,FP16,FP16ALT with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 transpose=1 vec=1"
#FP16ALT,FP16ALT,FP32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 vec=1"
#FP16ALT,FP16ALT,FP32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 transpose=1 vec=1"
#FP16ALT,FP16ALT,FP16
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 vec=1"
#FP16ALT,FP16ALT,FP16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=true --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 transpose=1 vec=1"

# ---------------------------------------------- mac flag = false
#--------------fixed precision
# fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1"
# fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1 transpose=1"
# fp16
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1"
# fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1 transpose=1"
# fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1"
# fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1 transpose=1"
# fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1 no_fmadd=1"
# fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1 no_fmadd=1 transpose=1"
# fp8 with vectorization
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1 vec=1"
# fp8 with vectorization and transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP8 print_results=1 no_fmadd=1 vec=1 transpose=1"
# fp16 with vectorization
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1 vec=1"
# fp16 with vectorization and transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16 --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1 vec=1 transpose=1"
# fp16alt with vectorization
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1 vec=1"
# fp16alt with vectorization and transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1 vec=1 transpose=1"

# --------------Mixed precision without HW mixed
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#FP8,fp16,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#FP8,fp16,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1"
#FP8,fp16alt,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1"
#FP8,fp16alt,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1"
#FP8, fp16,fp16
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP16 print_results=1 no_fmadd=1"
#FP8, fp16,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP16 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP16 print_results=1 no_fmadd=1 transpose=1"
#FP8,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#FP8,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp16,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp16alt,fp16alt,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16,fp16alt,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP16 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 transpose=1"
#fp16alt,fp16,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16alt,fp16,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1"
#fp16,fp16alt,fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP16ALT --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP16ALT print_results=1 no_fmadd=1"
#fp16,fp16alt,fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP16ALT --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP16ALT print_results=1 no_fmadd=1 transpose=1"
#fp16,fp16alt,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16ALT,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp8,fp32,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1"
# fp8,fp32,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp32,fp8,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp32,fp8,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp16,fp32,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16,fp32,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp32,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp32,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp16alt,fp32,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP32,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp16alt,fp32,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP32,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP32 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp32,fp16alt,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16ALT,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1"
#fp32,fp16alt,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP32,FP16ALT,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP32 fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1"
#fp8,fp32,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp8,fp32,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP32,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP32 fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1"
#fp16,fp32,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP8 print_results=1 no_fmadd=1"
#fp16,fp32,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP32,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP32 fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1"

# --------------Mixed precision with HW mixed
#fp8,fp8,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 hwmixed=1"
#fp8,fp8,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp8,fp8,fp16
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1 hwmixed=1"
#fp8,fp8,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp8,fp8,fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 hwmixed=1"
#fp8,fp8,fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp16,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 hwmixed=1"
#fp16,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp16,fp16,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 hwmixed=1"
#fp16,fp16,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp16,fp16,fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 hwmixed=1"
#fp16,fp16,fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp16alt,fp16alt,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 hwmixed=1"
#fp16alt,fp16alt,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp16alt,fp16alt,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 hwmixed=1"
#fp16alt,fp16alt,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
#fp16alt,fp16alt,fp16
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false --transpose=false --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 hwmixed=1"
#fp16alt,fp16alt,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=false --transpose=true --hwmixed_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 transpose=1 hwmixed=1"
# -------------- Vectorized mixed precision
#FP8,FP8,FP32
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
# fp8,fp8,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP32 --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1 vec=1"
#FP8,FP8,FP16
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1 vec=1"
# fp8,fp8,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16 --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16 print_results=1 no_fmadd=1 transpose=1 vec=1"
# fp8,fp8,fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 vec=1"
# fp8,fp8,fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP8_CUSTOM,FP8_CUSTOM,FP16ALT --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP8 fmt_B=FP8 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 transpose=1 vec=1"
# fp16,fp16,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
# fp16,fp16,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP32 --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1 vec=1"
# fp16,fp16,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 vec=1"
# fp16,fp16,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP8_CUSTOM --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1 vec=1"
# fp16alt,fp16alt,fp8
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 vec=1"
# fp16alt,fp16alt,fp8 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP8_CUSTOM --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP8 print_results=1 no_fmadd=1 transpose=1 vec=1"
# fp16,fp16,fp16alt
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 vec=1"
# fp16,fp16,fp16alt with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16,FP16,FP16ALT --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16 fmt_B=FP16 fmt_OUT=FP16ALT print_results=1 no_fmadd=1 transpose=1 vec=1"
# fp16alt,fp16alt,fp32
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 vec=1"
# fp16alt,fp16alt,fp32 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP32 --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP32 print_results=1 no_fmadd=1 transpose=1 vec=1"
# fp16alt,fp16alt,fp16
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=true --transpose=false --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 vec=1"
# fp16alt,fp16alt,fp16 with transpose
"PY: --m=60 --n=57  --p=61 --float_type=FP16ALT,FP16ALT,FP16 --mac_flag=false --vec_flag=true --transpose=true --hwmixed_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt_A=FP16ALT fmt_B=FP16ALT fmt_OUT=FP16 print_results=1 no_fmadd=1 transpose=1 vec=1"

)