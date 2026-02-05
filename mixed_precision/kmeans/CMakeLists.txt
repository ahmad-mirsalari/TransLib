cmake_minimum_required(VERSION 3.19)

# GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

project(kmeans C ASM)

# Configurable build options
set(cores 1 CACHE STRING "Number of cores")
set(fmt FP32 CACHE STRING "Data format (e.g., FP16, FP32, etc.)")

option(FABRIC "Enable fabric mode" OFF)
option(vec "Enable vector mode" OFF)
option(check "Enable result checking" OFF)
option(verbose "Enable verbose output" OFF)
option(stats "Enable statistics output" OFF)
option(print_results "Print final results" OFF)

# Source files
set(APP_FC_SRCS main.c)
set(APP_SRCS kmeans.c)
list(APPEND APP_SRCS ${APP_FC_SRCS} ${APP_SRCS})

# Compiler flags
set(APP_CFLAGS
    -O3
    -g3
    -w
    -mno-memcpy
    -fno-tree-loop-distribute-patterns
    -DNUM_CORES=${cores}
)

# Optional feature flags
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
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
    if(NOT fmt_OUT STREQUAL "")
        list(APPEND APP_CFLAGS -DOUT${fmt_OUT})
    else()
        list(APPEND APP_CFLAGS -DOUTFP32)
    endif()
endif()

if(vec)
    list(APPEND APP_CFLAGS -DVECT)
endif()

if(check)
    list(APPEND APP_CFLAGS -DCHECK)
endif()

if(verbose)
    list(APPEND APP_CFLAGS -DVERBOSE)
endif()

if(stats)
    list(APPEND APP_CFLAGS -DSTATS)
endif()

if(print_results)
    list(APPEND APP_CFLAGS -DPRINT_RESULTS)
endif()

# Linker flags (if needed)
# set(APP_LDFLAGS -flto)

# Build target
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Include optional rules
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# GAP SDK OS setup
setupos(${PROJECT_NAME})
