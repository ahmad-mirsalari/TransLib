ifndef target
target = pulp
endif

# Use 'target = gap9 ' to compile for GAP9
ifeq ($(target), gap9)
APP = svm

APP_SRCS  = main.c plp_SVM_predict.c plp_dot_prod_f32.c plp_exp.c

APP_CFLAGS += -O3  -DLIBM_ALIAS -mno-memcpy -fno-tree-loop-distribute-patterns -flto


ifdef cores
APP_CFLAGS += -DNUM_CORES=${cores} #-flto -DFABRIC=1
else
APP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
APP_CFLAGS += -DFABRIC
endif

ifdef cores
APP_CFLAGS +=  -DUSE_INTRINSICS -DPARALLEL #-flto
endif

# CHECK AND PRINT RESULTS
ifdef check
APP_CFLAGS += -DCHECK
endif

ifdef verbose
APP_CFLAGS += -DVERBOSE
endif


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

ifdef print_results
APP_CFLAGS += -DPRINT_RESULTS
endif

# VECTORIAL FORMAT for half-precision FP
ifdef vec
APP_CFLAGS += -DVECT
endif

APP_CFLAGS += -flto

# STATISTICS
ifdef stats
APP_CFLAGS += -DSTATS
endif

include $(RULES_DIR)/pmsis_rules.mk


else
# Use 'target = pulp' to compile for PULP
PULP_APP = svm

PULP_APP_FC_SRCS  = main.c
PULP_APP_SRCS  =    plp_SVM_predict.c plp_dot_prod_f32.c plp_exp.c

PULP_CFLAGS += -O3  -DLIBM_ALIAS -mno-memcpy -fno-tree-loop-distribute-patterns -flto


ifdef cores
PULP_CFLAGS += -DNUM_CORES=${cores} #-flto -DFABRIC=1
else
PULP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifdef cores
PULP_CFLAGS +=  -DUSE_INTRINSICS -DPARALLEL #-flto
endif

# CHECK AND PRINT RESULTS
ifdef check
PULP_CFLAGS += -DCHECK
endif

ifdef verbose
PULP_CFLAGS += -DVERBOSE
endif


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

ifdef print_results
PULP_CFLAGS += -DPRINT_RESULTS
endif

# VECTORIAL FORMAT for half-precision FP
ifdef vec
PULP_CFLAGS += -DVECT
endif

PULP_LDFLAGS += -flto

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