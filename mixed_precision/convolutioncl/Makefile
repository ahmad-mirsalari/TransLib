ifndef target
target = pulp
endif

ifeq ($(target),gap9)
APP = fp_conv

APP_SRCS += main.c conv_kernels.c

APP_CFLAGS += -O3 -g3

APP_CFLAGS += -mno-memcpy -flto 

APP_LDFLAGS += -flto  

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


ifdef fmt
APP_CFLAGS += -D${fmt} -DFIXED

else
# FP FORMAT
#INPUT DATA TYPE
ifdef fmt_INP
APP_CFLAGS += -DIN${fmt_INP}
else
APP_CFLAGS += -DINFP32
endif

#FILTER DATA TYPE
ifdef fmt_FIL
APP_CFLAGS += -DFIL${fmt_FIL}
else
APP_CFLAGS += -DFILFP32
endif

# OUTPUT DATA TYPE

ifdef fmt_OUT
APP_CFLAGS += -DOUT${fmt_OUT}
else
APP_CFLAGS += -DOUTFP32
endif
endif

#ifdef TILING
#PULP_CFLAGS += -DTILING
#endif

# VECTORIAL FORMAT for half-precision FP
ifdef vec
APP_CFLAGS += -DVECTORIAL
endif

# CHECK RESULTS
ifdef check
APP_CFLAGS += -DCHECK
endif

ifdef verbose
APP_CFLAGS += -DVERBOSE
endif
ifdef print_results
APP_CFLAGS += -DPRINT_RESULTS
endif
# STATISTICS
ifdef stats
APP_CFLAGS += -DSTATS
endif

include $(RULES_DIR)/pmsis_rules.mk

else
# Make file for PULP

PULP_APP = fp_conv

PULP_APP_FC_SRCS += main.c
PULP_APP_SRCS += conv_kernels.c

PULP_CFLAGS += -O3 -g3

PULP_CFLAGS += -mno-memcpy -flto 

PULP_LDFLAGS += -flto  

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


ifdef fmt
PULP_CFLAGS += -D${fmt} -DFIXED

else
# FP FORMAT
#INPUT DATA TYPE
ifdef fmt_INP
PULP_CFLAGS += -DIN${fmt_INP}
else
PULP_CFLAGS += -DINFP32
endif

#FILTER DATA TYPE
ifdef fmt_FIL
PULP_CFLAGS += -DFIL${fmt_FIL}
else
PULP_CFLAGS += -DFILFP32
endif

# OUTPUT DATA TYPE

ifdef fmt_OUT
PULP_CFLAGS += -DOUT${fmt_OUT}
else
PULP_CFLAGS += -DOUTFP32
endif
endif

#ifdef TILING
#PULP_CFLAGS += -DTILING
#endif

# VECTORIAL FORMAT for half-precision FP
ifdef vec
PULP_CFLAGS += -DVECTORIAL
endif

# CHECK RESULTS
ifdef check
PULP_CFLAGS += -DCHECK
endif

ifdef verbose
PULP_CFLAGS += -DVERBOSE
endif
ifdef print_results
PULP_CFLAGS += -DPRINT_RESULTS
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