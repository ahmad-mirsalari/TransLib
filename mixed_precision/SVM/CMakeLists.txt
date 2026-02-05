cmake_minimum_required(VERSION 3.19)
project(svm C ASM)

# Include GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Source files
set(APP_FC_SRCS main.c)
set(APP_SRCS plp_SVM_predict.c plp_dot_prod_f32.c plp_exp.c)
list(APPEND APP_SRCS ${APP_FC_SRCS} ${APP_SRCS})

# Compiler and linker flags
set(APP_CFLAGS 
    -O3 
    -DLIBM_ALIAS 
    -mno-memcpy 
    -fno-tree-loop-distribute-patterns 
    -flto
)

set(APP_LDFLAGS -flto)

# Configurable build options
set(cores 1 CACHE STRING "Number of cores")
set(fmt "" CACHE STRING "Fixed-point mode flag")
set(fmt_INP "" CACHE STRING "Input data type")
set(fmt_FIL "" CACHE STRING "Filter data type")
set(fmt_OUT "" CACHE STRING "Output data type")

option(FABRIC "Use fabric mode" OFF)
option(check "Enable result check" OFF)
option(verbose "Enable verbose printing" OFF)
option(print_results "Enable result printing" OFF)
option(vec "Enable vectorial support" OFF)
option(stats "Enable statistics" OFF)

# NUM_CORES and parallel intrinsics
list(APPEND APP_CFLAGS -DNUM_CORES=${cores})

if(cores GREATER 1)
    list(APPEND APP_CFLAGS -DUSE_INTRINSICS -DPARALLEL)
endif()

# FABRIC flag
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

# Fixed-point or floating-point format
if(NOT fmt STREQUAL "")
    list(APPEND APP_CFLAGS -D${fmt} -DFIXED)
else()
    if(NOT fmt_INP STREQUAL "")
        list(APPEND APP_CFLAGS -DIN${fmt_INP})
    else()
        list(APPEND APP_CFLAGS -DINFP32)
    endif()

    if(NOT fmt_FIL STREQUAL "")
        list(APPEND APP_CFLAGS -DFIL${fmt_FIL})
    else()
        list(APPEND APP_CFLAGS -DFILFP32)
    endif()

    if(NOT fmt_OUT STREQUAL "")
        list(APPEND APP_CFLAGS -DOUT${fmt_OUT})
    else()
        list(APPEND APP_CFLAGS -DOUTFP32)
    endif()
endif()

# Optional flags
if(check)
    list(APPEND APP_CFLAGS -DCHECK)
endif()

if(verbose)
    list(APPEND APP_CFLAGS -DVERBOSE)
endif()

if(print_results)
    list(APPEND APP_CFLAGS -DPRINT_RESULTS)
endif()

if(vec)
    list(APPEND APP_CFLAGS -DVECT)
endif()

if(stats)
    list(APPEND APP_CFLAGS -DSTATS)
endif()
list(APPEND APP_CFLAGS -Wno-error)
# Create executable and apply flags
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Include optional rules directory
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# GAP SDK OS support
setupos(${PROJECT_NAME})
