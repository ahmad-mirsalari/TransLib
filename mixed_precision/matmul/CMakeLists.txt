cmake_minimum_required(VERSION 3.19)
project(fp_matmul C ASM)

# Load GAP SDK setup if needed
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Source files
set(APP_FC_SRCS main.c)
set(APP_SRCS matmul.c)
list(APPEND APP_SRCS ${APP_FC_SRCS} ${APP_SRCS})

# Compiler flags
set(APP_CFLAGS
    -O3 -g3
    -mno-memcpy
    -fno-tree-vectorize
)

# Handle NUM_CORES
set(cores 1 CACHE STRING "Number of cores")
list(APPEND APP_CFLAGS -DNUM_CORES=${cores})

# Enable intrinsics if parallel
if(cores GREATER 1)
    list(APPEND APP_CFLAGS -DUSE_INTRINSICS)
endif()

# Handle FABRIC
option(FABRIC "Enable fabric" OFF)
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
    message(STATUS "Using fabric")
endif()

# Handle fixed-point or floating-point formats
set(fmt "" CACHE STRING "Fixed-point format")
set(fmt_A "" CACHE STRING "Input format")
set(fmt_B "" CACHE STRING "Filter format")
set(fmt_OUT "" CACHE STRING "Output format")

if(NOT fmt STREQUAL "")
    list(APPEND APP_CFLAGS -D${fmt} -DFIXED)
    message(STATUS "Using fixed-point format: ${fmt}")
else()
    if(NOT fmt_A STREQUAL "")
        list(APPEND APP_CFLAGS -DMA${fmt_A})
    else()
        list(APPEND APP_CFLAGS -DMAFP32)
    endif()

    if(NOT fmt_B STREQUAL "")
        list(APPEND APP_CFLAGS -DMB${fmt_B})
    else()
        list(APPEND APP_CFLAGS -DMBFP32)
    endif()

    if(NOT fmt_OUT STREQUAL "")
        list(APPEND APP_CFLAGS -DOUT${fmt_OUT})
    else()
        list(APPEND APP_CFLAGS -DOUTFP32)
    endif()
endif()

# VECTORIAL
option(vec "Enable vectorial" OFF)
if(vec)
    list(APPEND APP_CFLAGS -DVECTORIAL)
endif()

# Transpose
option(transpose "Enable transpose" OFF)
if(transpose)
    list(APPEND APP_CFLAGS -DTRANSPOSE)
endif()
# CHECK RESULTS
option(check "Enable check mode" OFF)
if(check)
    list(APPEND APP_CFLAGS -DCHECK)
endif()

# PRINT RESULTS
option(print_results "Enable result printing" OFF)
if(print_results)
    list(APPEND APP_CFLAGS -DPRINT_RESULTS)
endif()

# VERBOSE
option(verbose "Enable verbose mode" OFF)
if(verbose)
    list(APPEND APP_CFLAGS -DVERBOSE)
endif()

# STATISTICS
option(stats "Enable statistics collection" OFF)
if(stats)
    list(APPEND APP_CFLAGS -DSTATS)
endif()
list(APPEND APP_CFLAGS -Wno-error)

# Target and build setup
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})

# Link and include dirs if needed
target_link_libraries(${PROJECT_NAME} PRIVATE -lm)
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Setup OS if needed
setupos(${PROJECT_NAME})

# Optional: Include PMSIS rules dir
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()
