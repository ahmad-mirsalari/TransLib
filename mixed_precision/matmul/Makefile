ifndef target
target = pulp
endif

# Use 'target=gap9' to compile for GAP9, otherwise defaults to PULP
ifeq ($(target),gap9)
# Debug output (optional)
$(info target is [$(target)])
APP = fp_matmul

APP_SRCS = main.c matmul.c

APP_CFLAGS += -O3 -g3 
APP_CFLAGS += -mno-memcpy


ifdef cores
APP_CFLAGS += -DNUM_CORES=${cores} #-flto -DFABRIC=1
else
APP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
APP_CFLAGS += -DFABRIC
endif

ifdef cores
APP_CFLAGS += -DUSE_INTRINSICS
endif

APP_CFLAGS += -fno-tree-vectorize

ifdef transpose
APP_CFLAGS += -DTRANSPOSE
endif

ifdef fmt
APP_CFLAGS += -D${fmt} -DFIXED

else
# FP FORMAT
#INPUT DATA TYPE
ifdef fmt_A
APP_CFLAGS += -DMA${fmt_A}
else
APP_CFLAGS += -DMAFP32
endif

#FILTER DATA TYPE
ifdef fmt_B
APP_CFLAGS += -DMB${fmt_B}
else
APP_CFLAGS += -DMBFP32
endif

# OUTPUT DATA TYPE

ifdef fmt_OUT
APP_CFLAGS += -DOUT${fmt_OUT}
else
APP_CFLAGS += -DOUTFP32
endif
endif

# VECTORIAL FORMAT for half-precision FP
ifdef vec
APP_CFLAGS += -DVECTORIAL
endif

# CHECK RESULTS
ifdef check
APP_CFLAGS += -DCHECK
endif

ifdef print_results
APP_CFLAGS += -DPRINT_RESULTS
endif

ifdef verbose
APP_CFLAGS += -DVERBOSE
endif

# STATISTICS
ifdef stats
APP_CFLAGS += -DSTATS
endif

include $(RULES_DIR)/pmsis_rules.mk

else
# Makefile for the PULP platform
# This Makefile is used to compile the matmul example for the PULP platform
# It includes the necessary flags and sources for the compilation
PULP_APP = fp_matmul

PULP_APP_FC_SRCS = main.c
PULP_APP_SRCS = matmul.c

PULP_CFLAGS += -O3 -g3 
PULP_CFLAGS += -mno-memcpy


ifdef cores
PULP_CFLAGS += -DNUM_CORES=${cores} #-flto -DFABRIC=1
else
PULP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifdef cores
PULP_CFLAGS += -DUSE_INTRINSICS
endif

PULP_CFLAGS += -fno-tree-vectorize

ifdef transpose
PULP_CFLAGS += -DTRANSPOSE
endif

ifdef fmt
PULP_CFLAGS += -D${fmt} -DFIXED

else
# FP FORMAT
#INPUT DATA TYPE
ifdef fmt_A
PULP_CFLAGS += -DMA${fmt_A}
else
PULP_CFLAGS += -DMAFP32
endif

#FILTER DATA TYPE
ifdef fmt_B
PULP_CFLAGS += -DMB${fmt_B}
else
PULP_CFLAGS += -DMBFP32
endif

# OUTPUT DATA TYPE

ifdef fmt_OUT
PULP_CFLAGS += -DOUT${fmt_OUT}
else
PULP_CFLAGS += -DOUTFP32
endif
endif

# VECTORIAL FORMAT for half-precision FP
ifdef vec
PULP_CFLAGS += -DVECTORIAL
endif

# CHECK RESULTS
ifdef check
PULP_CFLAGS += -DCHECK
endif

ifdef print_results
PULP_CFLAGS += -DPRINT_RESULTS
endif

ifdef verbose
PULP_CFLAGS += -DVERBOSE
endif

# STATISTICS
ifdef stats
PULP_CFLAGS += -DSTATS
endif

# Mixed precision HW support
ifdef hwmixed 
PULP_CFLAGS += -DHWMIXED
PULP_CFLAGS += -mfaux
endif

ifdef no_fmadd
PULP_CFLAGS += -ffp-contract=off
endif

include $(RULES_DIR)/pmsis_rules.mk

endif