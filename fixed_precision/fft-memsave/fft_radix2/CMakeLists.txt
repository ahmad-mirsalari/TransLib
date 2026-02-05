cmake_minimum_required(VERSION 3.19)
project(test C ASM)

# GAP SDK setup
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

# Configurable options
option(FABRIC "Enable fabric mode" OFF)
set(OPT 1 CACHE STRING "Optimization level (0 = RV32IMFC)")
set(CORES 1 CACHE STRING "Number of cores")

# Source files
if(FABRIC)
    set(APP_APP_FC_SRCS
        main.c
        fft_radix2.c
        print_float.c
        utils.c
    )
else()
    set(APP_APP_FC_SRCS main.c)
    set(APP_APP_SRCS
        fft_radix2.c
        print_float.c
        utils.c
    )
endif()
list(APPEND APP_SRCS ${APP_APP_FC_SRCS} ${APP_APP_SRCS})

# Compiler flags
set(APP_CFLAGS
    -O3
    -g
)

# Architecture-specific flags based on OPT
if(OPT EQUAL 0)
    list(APPEND APP_CFLAGS -march=rv32imfc -D__riscv__ -DNUM_CORES=1)
    set(APP_LDFLAGS -march=rv32imfc -D__riscv__)
elseif(OPT EQUAL 1 OR OPT EQUAL 2)
    list(APPEND APP_CFLAGS -DNUM_CORES=1)
endif()

# FABRIC mode
if(FABRIC)
    list(APPEND APP_CFLAGS -DFABRIC)
endif()

# Core configuration
if(CORES GREATER 1)
    list(APPEND APP_CFLAGS -DNUM_CORES=${CORES} -DPARALLEL)
else()
    list(APPEND APP_CFLAGS -DNUM_CORES=1)
endif()

# Linker flags
list(APPEND APP_LDFLAGS -lm)

# Executable and compile/link setup
add_executable(${PROJECT_NAME} ${APP_SRCS})
target_compile_options(${PROJECT_NAME} PRIVATE ${APP_CFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${APP_LDFLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TARGET_INCS})

# Optional include directory
if(DEFINED RULES_DIR)
    include_directories(${RULES_DIR})
endif()

# GAP SDK runtime integration
setupos(${PROJECT_NAME})
