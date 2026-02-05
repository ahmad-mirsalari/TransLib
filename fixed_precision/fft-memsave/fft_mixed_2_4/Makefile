PULP_APP = test

ifdef FABRIC
PULP_APP_FC_SRCS  = main.c ../fft_radix2/fft_radix2.c ../fft_radix2/utils.c fft_mixed_2_4.c utils.c print_float.c
else
PULP_APP_FC_SRCS  = main.c
PULP_APP_SRCS  = ../fft_radix2/fft_radix2.c ../fft_radix2/utils.c fft_mixed_2_4.c utils.c print_float.c
endif

PULP_CFLAGS += -O3 -g -include ../fft_mixed_2_4/fft_mixed.h -DMIXED_RADIX -flto

ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifeq (${OPT}, 0)
PULP_CL_ARCH_CFLAGS = -march=rv32imfc  -D__riscv__
PULP_ARCH_LDFLAGS = -march=rv32imfc -D__riscv__
PULP_CFLAGS += -DNUM_CORES=1
endif

ifeq (${OPT}, 1)
PULP_CFLAGS += -DNUM_CORES=1
endif

ifeq (${OPT}, 2)
PULP_CFLAGS += -DNUM_CORES=1
endif


ifdef CORES
PULP_CFLAGS += -DNUM_CORES=${CORES} -DPARALLEL
else
PULP_CFLAGS += -DNUM_CORES=1
endif

PULP_LDFLAGS += -flto -Wl,--gc-sections -lm

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
