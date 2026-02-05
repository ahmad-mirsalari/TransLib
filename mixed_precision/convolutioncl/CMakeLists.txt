cmake_minimum_required(VERSION 3.19)
project(fp_conv C ASM)

# Load GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Sources
set(APP_APP_FC_SRCS main.c)
set(APP_APP_SRCS conv_kernels.c)
list(APPEND APP_SRCS ${APP_APP_FC_SRCS} ${APP_APP_SRCS})

# Compiler flags
set(APP_CFLAGS
    -O3 -g3
    -mno-memcpy
    -flto
    -fno-tree-vectorize
)

# Linker flags
set(APP_LDFLAGS -flto)

# Configurable options
set(cores 1 CACHE STRING "Number of cores")
set(fmt "" CACHE STRING "Fixed-point mode flag (e.g., FIX16)")
set(fmt_INP "" CACHE STRING "Input data type (e.g., FP16)")
set(fmt_FIL "" CACHE STRING "Filter data type")
set(fmt_OUT "" CACHE STRING "Output data type")

option(FABRIC "Use fabric mode" OFF)
option(vec "Enable vectorial support" OFF)
option(check "Enable result checking" OFF)
option(verbose "Enable verbose printing" OFF)
option(print_results "Print results" OFF)
option(stats "Enable statistics" OFF)

# Core count and intrinsics
list(APPEND APP_CFLAGS -DNUM_CORES=${cores})
if(cores GREATER 1)
    list(APPEND APP_CFLAGS -DUSE_INTRINSICS)
endif()

# FABRIC flag
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

# Data format configuration
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
if(vec)
    list(APPEND APP_CFLAGS -DVECTORIAL)
endif()

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
list(APPEND APP_CFLAGS -Wno-error)
list(APPEND APP_CFLAGS -Wno-error=maybe-uninitialized)

# Create the executable
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Include rules if needed
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# Setup OS (FreeRTOS / no_os)
setupos(${PROJECT_NAME})
