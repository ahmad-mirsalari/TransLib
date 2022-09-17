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
PULP_CFLAGS += -D${fmt}
else
PULP_CFLAGS += -DFP32
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
ifdef PRINT_RESULTS
PULP_CFLAGS += -DPRINT_RESULTS
endif

PULP_CFLAGS += -O3 -g3

#PULP_CFLAGS += -flto 

#PULP_LDFLAGS += -flto 
PULP_CFLAGS += -mno-memcpy -fno-tree-loop-distribute-patterns

include $(RULES_DIR)/pmsis_rules.mk
