cmake_minimum_required(VERSION 3.19)
project(fir C ASM)

# Load GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Source files
set(APP_FC_SRCS main.c)
set(APP_CL_SRCS fir_kernel.c)
list(APPEND APP_SRCS ${APP_FC_SRCS} ${APP_CL_SRCS})

# Compiler flags
set(APP_CFLAGS -O3 -g -w)  # -w disables all warnings

# Option and variable definitions
set(cores 1 CACHE STRING "Number of cores")
set(fmt "" CACHE STRING "Fixed-point mode (e.g., FIX16)")
set(fmt_INP "" CACHE STRING "Input data type (e.g., FP16)")
set(fmt_FIL "" CACHE STRING "Filter data type")
set(fmt_OUT "" CACHE STRING "Output data type")

option(FABRIC "Use fabric mode" OFF)
option(vec "Enable vectorial mode" OFF)
option(check "Enable result check" OFF)
option(verbose "Enable verbose printing" OFF)
option(print_results "Print output values" OFF)
option(stats "Enable statistics" OFF)

# NUM_CORES and USE_INTRINSICS
list(APPEND APP_CFLAGS -DNUM_CORES=${cores})

# FABRIC mode
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

# Vectorial
if(vec)
    list(APPEND APP_CFLAGS -DVECT)
endif()

# Fixed-point or floating-point handling
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


if(stats)
    list(APPEND APP_CFLAGS -DSTATS)
endif()

# Build the executable
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Optional rules
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# GAP SDK OS integration
setupos(${PROJECT_NAME})
