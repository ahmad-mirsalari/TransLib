cmake_minimum_required(VERSION 3.19)
project(test C ASM)

# Include GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Configurable options
option(FABRIC "Enable fabric mode" OFF)
option(BITREV_LUT "Enable bit reversal LUT" OFF)
option(print_results "Print FFT results" OFF)
option(FULL_TWIDDLES "Use full twiddle factors" OFF)
option(SORT_OUTPUT "Enable FFT output sorting" OFF)
option(stats "Enable statistics output" OFF)
option(vec "Enable vectorization" OFF)

set(cores 1 CACHE STRING "Number of cores")
set(fmt FP32 CACHE STRING "Floating-point format")

# Source files
if(FABRIC)
    set(APP_FC_SRCS
        main.c
        ../fft_radix2/fft_radix2.c
        ../fft_radix2/utils.c
        ../fft_radix8/utils.c
        fft_mixed_2_8.c
        utils.c
    )
else()
    set(APP_FC_SRCS main.c)
    set(APP_SRCS
        ../fft_radix2/fft_radix2.c
        ../fft_radix2/utils.c
        ../fft_radix8/utils.c
        fft_mixed_2_8.c
        utils.c
    )
endif()
list(APPEND APP_SRCS ${APP_FC_SRCS} ${APP_SRCS})

# Compiler flags
set(APP_CFLAGS
    -O3
    -g
    -DMIXED_RADIX
    -flto
    -DNUM_CORES=${cores}
)

if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

if(BITREV_LUT)
    list(APPEND APP_CFLAGS -DBITREV_LUT)
endif()

if(print_results)
    list(APPEND APP_CFLAGS -DPRINT_RESULTS)
endif()

if(FULL_TWIDDLES)
    list(APPEND APP_CFLAGS -DFULL_TWIDDLES)
endif()

if(SORT_OUTPUT)
    list(APPEND APP_CFLAGS -DSORT_OUTPUT)
endif()

if(stats)
    list(APPEND APP_CFLAGS -DSTATS)
endif()

if(vec)
    list(APPEND APP_CFLAGS -DVECTORIZATION)
endif()

if(fmt)
    list(APPEND APP_CFLAGS -D${fmt})
else()
    list(APPEND APP_CFLAGS -DFP32)
endif()
list(APPEND APP_CFLAGS -Wno-error)
# Linker flags
set(APP_LDFLAGS -flto -Wl,--gc-sections -lm)

# Create executable and apply flags
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Optional: Include rule directory
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# Setup SDK OS support
setupos(${PROJECT_NAME})
