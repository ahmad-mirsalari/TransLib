cmake_minimum_required(VERSION 3.19)
project(ecg_processing C ASM)

# GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Configurable options
option(FABRIC "Enable fabric mode" OFF)
option(stats "Enable statistics collection" OFF)
option(debug "Enable debug mode" OFF)
option(vec "Enable vectorial mode" OFF)

set(fmt FP32 CACHE STRING "Floating-point format (e.g., FP32, FP16)")
set(CORES 1 CACHE STRING "Number of cores")

# Source files
set(APP_SRCS
    ecg_processing.c
    single_convolution.c
)

# Compiler flags
set(APP_CFLAGS
    -O3
    -fno-tree-loop-distribute-patterns
    -flto
)

# FABRIC
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

# Parallelism & NUM_CORES
if(CORES GREATER 1)
    list(APPEND APP_CFLAGS -DPARALLEL)
endif()

list(APPEND APP_CFLAGS -DNUM_CORES=${CORES})

# Feature flags
if(stats)
    list(APPEND APP_CFLAGS -DSTATS)
endif()

if(debug)
    list(APPEND APP_CFLAGS -DDEBUG)
endif()

if(vec)
    list(APPEND APP_CFLAGS -DVECTORIAL)
endif()

# Format type
if(fmt)
    list(APPEND APP_CFLAGS -D${fmt})
else()
    list(APPEND APP_CFLAGS -DFP32)
endif()

# Linker flags
set(APP_LDFLAGS -lm -flto -w)

# Build the executable
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Optional: include rules
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# Setup for PULP runtime
setupos(${PROJECT_NAME})
