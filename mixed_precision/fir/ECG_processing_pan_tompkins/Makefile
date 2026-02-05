PULP_APP = ecg_processing

APP_SRCS  = ecg_processing.c single_convolution.c

PULP_CFLAGS += -O3 -fno-tree-loop-distribute-patterns -flto 

#### to execute in the FABRIC: make clean all run FABRIC=1, and set in "stats.h": HOTTING=0 and REPEAT=1
ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifndef cores
CORES = 1
else
PULP_CFLAGS += -DPARALLEL
endif

ifdef stats
PULP_CFLAGS += -DSTATS
endif
ifdef CORES
PULP_CFLAGS += -DNUM_CORES=${CORES}
endif

# STATISTICS
#PULP_CFLAGS += -DSTATS

# CHECK RESULTS
ifdef debug
PULP_CFLAGS += -DDEBUG
endif

ifdef vec
PULP_CFLAGS += -DVECTORIAL
endif

PULP_LDFLAGS += -lm -flto

# FP FORMAT
ifdef fmt
PULP_CFLAGS += -D${fmt}
else
PULP_CFLAGS += -DFP32
endif
#PULP_LDFLAGS += -lgcc -lm -flto


include $(RULES_DIR)/pmsis_rules.mk
#### per cercare la variabile d'ambiente RULES_DIR usare: env | grep RULES #####
