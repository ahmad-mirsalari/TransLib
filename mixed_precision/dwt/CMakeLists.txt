cmake_minimum_required(VERSION 3.19)
project(dwt C ASM)

# Load GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Configurable options
set(FABRIC OFF CACHE BOOL "Use fabric mode")
set(cores 1 CACHE STRING "Number of cores")
set(OPT 1 CACHE STRING "Optimization level (0 = rv32imfc)")
set(fmt "" CACHE STRING "Fixed-point mode (e.g., FIX16)")
set(fmt_INP "" CACHE STRING "Input data type (e.g., FP16)")
set(fmt_FIL "" CACHE STRING "Filter data type")
set(fmt_OUT "" CACHE STRING "Output data type")
set(BARRIER "" CACHE STRING "Barrier type: sw or tas")

option(BOARD "Target board" OFF)
option(stats "Enable statistics" OFF)
option(check "Enable result checking" OFF)
option(verbose "Enable verbose output" OFF)
option(print_results "Print output" OFF)
option(vec "Enable vectorial mode" OFF)

# Source files
if(FABRIC)
    set(APP_FC_SRCS main.c wavelet.c)
else()
    set(APP_FC_SRCS main.c)
    set(APP_SRCS wavelet.c barrier.c)
endif()
list(APPEND APP_SRCS ${APP_FC_SRCS} ${APP_SRCS})

# Base compiler flags
set(APP_CFLAGS -O3 -g -flto)

# Optional: parallel mode
if(cores GREATER 1)
    list(APPEND APP_CFLAGS -DPARALLEL)
endif()

# Add NUM_CORES definition
list(APPEND APP_CFLAGS -DNUM_CORES=${cores})

# FABRIC mode
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

# Optional flags
if(BOARD)
    list(APPEND APP_CFLAGS -DBOARD)
endif()

if(stats)
    list(APPEND APP_CFLAGS -DSTATS)
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

if(vec)
    list(APPEND APP_CFLAGS -DVECTORIAL)
endif()
list(APPEND APP_CFLAGS -Wno-error)
# Barrier type
if(BARRIER STREQUAL "sw")
    list(APPEND APP_CFLAGS -DSW_BARRIER)
elseif(BARRIER STREQUAL "tas")
    list(APPEND APP_CFLAGS -DTAS_BARRIER)
endif()

# RV32-specific flags (if OPT = 0)
if(OPT EQUAL 0)
    list(APPEND APP_CFLAGS -march=rv32imfc -DRV_ISA_RV32=1 -D__riscv__)
    set(APP_LDFLAGS -march=rv32imfc -DRV_ISA_RV32=1 -D__riscv__)
endif()

# Linker flags
list(APPEND APP_LDFLAGS -lgcc -lm -flto)

# Build target
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Optional include dir
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# GAP SDK support
setupos(${PROJECT_NAME})
