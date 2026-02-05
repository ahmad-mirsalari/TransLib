cmake_minimum_required(VERSION 3.19)
project(test C ASM)

# Include GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Configurable build options
option(FABRIC "Enable fabric mode" OFF)
set(OPT 1 CACHE STRING "Optimization level (0 = RV32IMFC arch flags)")
set(CORES 1 CACHE STRING "Number of cores")

# Source files
if(FABRIC)
    set(APP_FC_SRCS
        main.c
        ../fft_radix2/fft_radix2.c
        ../fft_radix2/utils.c
        fft_mixed_2_4.c
        utils.c
        print_float.c
    )
else()
    set(APP_FC_SRCS main.c)
    set(APP_SRCS
        ../fft_radix2/fft_radix2.c
        ../fft_radix2/utils.c
        fft_mixed_2_4.c
        utils.c
        print_float.c
    )
endif()
list(APPEND APP_SRCS ${APP_FC_SRCS} ${APP_SRCS})

# Base compiler flags
set(APP_CFLAGS
    -O3
    -g
    -include ../fft_mixed_2_4/fft_mixed.h
    -DMIXED_RADIX
    -flto
)

# Conditional architecture flags (OPT = 0 → RV32IMFC)
if(OPT EQUAL 0)
    list(APPEND APP_CFLAGS -march=rv32imfc -D__riscv__)
    list(APPEND APP_CFLAGS -DNUM_CORES=1)
    set(APP_LDFLAGS -march=rv32imfc -D__riscv__)
elseif(OPT EQUAL 1 OR OPT EQUAL 2)
    list(APPEND APP_CFLAGS -DNUM_CORES=1)
endif()

# FABRIC support
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

# Parallel/multicore setup
if(CORES GREATER 1)
    list(APPEND APP_CFLAGS -DNUM_CORES=${CORES} -DPARALLEL)
else()
    list(APPEND APP_CFLAGS -DNUM_CORES=1)
endif()

# Linker flags
list(APPEND APP_LDFLAGS -flto -Wl,--gc-sections -lm)

# Build target
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Optional: add include directories manually if needed
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# GAP SDK setup
setupos(${PROJECT_NAME})
