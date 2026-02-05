ifndef target
target = pulp
endif

# Use 'target=gap9' to compile for the GAP9
ifeq ($(target), gap9)
# Debug output (optional)
$(info target is [$(target)])
APP = kmeans

APP_SRCS += main.c kmeans.c

ifdef cores
APP_CFLAGS += -O3  -DNUM_CORES=${cores} -w -g
else
APP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
APP_CFLAGS += -DFABRIC
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
# OUTPUT DATA TYPE
ifdef fmt_OUT
APP_CFLAGS += -DOUT${fmt_OUT}
else
APP_CFLAGS += -DOUTFP32
endif
endif

ifdef vec
APP_CFLAGS += -DVECT
endif

ifdef check
APP_CFLAGS += -DCHECK
endif

ifdef verbose
APP_CFLAGS += -DVERBOSE
endif

# STATISTICS
ifdef stats
APP_CFLAGS += -DSTATS
endif
ifdef print_results
APP_CFLAGS += -DPRINT_RESULTS
endif

APP_CFLAGS += -O3 -g3

#PULP_CFLAGS += -flto 

# PULP_LDFLAGS += -lgcc 
PULP_CFLAGS += -mno-memcpy -fno-tree-loop-distribute-patterns

include $(RULES_DIR)/pmsis_rules.mk

else
# Use 'target=pulp' to compile for the PULP
PULP_APP = kmeans

PULP_APP_FC_SRCS += main.c
PULP_APP_SRCS += kmeans.c

ifdef cores
PULP_CFLAGS += -O3  -DNUM_CORES=${cores} -w -g
else
PULP_CFLAGS += -DNUM_CORES=1
endif


ifdef FABRIC
PULP_CFLAGS += -DFABRIC
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
# OUTPUT DATA TYPE
ifdef fmt_OUT
PULP_CFLAGS += -DOUT${fmt_OUT}
else
PULP_CFLAGS += -DOUTFP32
endif
endif

ifdef vec
PULP_CFLAGS += -DVECT
endif

ifdef check
PULP_CFLAGS += -DCHECK
endif

ifdef verbose
PULP_CFLAGS += -DVERBOSE
endif

# STATISTICS
ifdef stats
PULP_CFLAGS += -DSTATS
endif
ifdef print_results
PULP_CFLAGS += -DPRINT_RESULTS
endif

PULP_CFLAGS += -O3

# Mixed precision HW support
ifdef hwmixed 
PULP_CFLAGS += -DHWMIXED
PULP_CFLAGS += -mfaux
endif
#PULP_CFLAGS += -flto 
ifdef no_fmadd
PULP_CFLAGS += -ffp-contract=off
endif

# PULP_LDFLAGS += -lgcc 
PULP_CFLAGS += -mno-memcpy -fno-tree-loop-distribute-patterns  -fno-tree-loop-im 

include $(RULES_DIR)/pmsis_rules.mk
endif
