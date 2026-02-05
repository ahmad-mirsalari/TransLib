PULP_APP = test

ifdef FABRIC
PULP_APP_FC_SRCS  = main.c fft_radix2.c print_float.c utils.c
else
PULP_APP_FC_SRCS  = main.c
PULP_APP_SRCS  = fft_radix2.c print_float.c utils.c
endif

PULP_CFLAGS += -O3 -g 

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

PULP_LDFLAGS += -lm

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
