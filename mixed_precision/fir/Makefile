PULP_APP = fir

PULP_APP_FC_SRCS  = main.c
PULP_APP_CL_SRCS = fir_kernel.c


ifdef cores
PULP_CFLAGS += -DNUM_CORES=${cores} #-flto -DFABRIC=1
else
PULP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifdef vec
PULP_CFLAGS += -DVECT
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
ifdef check
PULP_CFLAGS += -DCHECK
endif

ifdef verbose
PULP_CFLAGS += -DVERBOSE
endif

ifdef PRINT_RESULTS
PULP_CFLAGS += -DPRINT_RESULTS
endif
#PULP_LDFLAGS += -flto

ifdef stats
PULP_CFLAGS += -DSTATS
endif

PULP_CFLAGS += -O3 -g -w


include $(RULES_DIR)/pmsis_rules.mk
