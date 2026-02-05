PYTHON_BIN=python3
PY_SCRIPT=./data_generator.py
MAKE_BIN=make
MAKE_TARGETS="clean all run"
LOG_DIR="./logs"
RUNS=(
#-------------------------------  fmadd = true
# fp16
"PY: --input_size=512 --float_type=FP16 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1"
# fp16alt
"PY: --input_size=512 --float_type=FP16ALT --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1"
# fp32
"PY: --input_size=512 --float_type=FP32 --mac_flag=true --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1"
# fp16 with vectorization
"PY: --input_size=512 --float_type=FP16 --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 vec=1"
# fp16alt with vectorization
"PY: --input_size=512 --float_type=FP16ALT --mac_flag=true --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 vec=1"

#---------------------------------- fmadd = false
# fp16
"PY: --input_size=512 --float_type=FP16 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1"
# fp16alt
"PY: --input_size=512 --float_type=FP16ALT --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1"
# fp32
"PY: --input_size=512 --float_type=FP32 --mac_flag=false --vec_flag=false ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP32 print_results=1 no_fmadd=1"
# fp16 with vectorization
"PY: --input_size=512 --float_type=FP16 --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16 print_results=1 no_fmadd=1 vec=1"
# fp16alt with vectorization
"PY: --input_size=512 --float_type=FP16ALT --mac_flag=false --vec_flag=true ; MAKE: check=1 verbose=1 stats=1 cores=1 fmt=FP16ALT print_results=1 no_fmadd=1 vec=1"
)