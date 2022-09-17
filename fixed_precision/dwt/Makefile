PULP_APP = dwt

ifdef FABRIC
PULP_APP_FC_SRCS  = main.c wavelet.c
PULP_CFLAGS += -DFABRIC
else
PULP_APP_FC_SRCS  = main.c
PULP_APP_SRCS  = wavelet.c barrier.c
endif

ifndef cores
cores = 1
else
PULP_CFLAGS += -DPARALLEL
endif


ifdef BOARD
PULP_CFLAGS += -DBOARD
endif

PULP_CFLAGS += -O3 -g -flto 
#-mno-memcpy -ftree-loop-distribution -ftree-loop-distribute-patterns -fno-tree-loop-im -fno-tree-ch

ifeq (${OPT}, 0)
PULP_CL_ARCH_CFLAGS = -march=rv32imfc -DRV_ISA_RV32=1   -D__riscv__
PULP_ARCH_LDFLAGS = -march=rv32imfc -DRV_ISA_RV32=1  -D__riscv__
PULP_CFLAGS += -DNUM_CORES=1
endif

ifdef cores
PULP_CFLAGS += -DNUM_CORES=${cores}
endif

ifdef stats
PULP_CFLAGS += -DSTATS
endif

ifdef check
PULP_CFLAGS += -DCHECK
endif
# CHECK RESULTS
#PULP_CFLAGS += -DDEBUG
ifdef verbose
PULP_CFLAGS += -DVERBOSE
endif
ifdef PRINT_RESULTS
PULP_CFLAGS += -DPRINT_RESULTS
endif

# FP FORMAT
ifdef fmt
PULP_CFLAGS += -D${fmt}
else
PULP_CFLAGS += -DFP32
endif
ifdef vec
PULP_CFLAGS += -DVECTORIAL
endif
#PULP_CFLAGS  += -O3 -DNUM_CORES=${CORES}
PULP_LDFLAGS += -lm

ifeq (${BARRIER}, sw)
PULP_CFLAGS += -DSW_BARRIER
endif

ifeq (${BARRIER}, tas)
PULP_CFLAGS += -DTAS_BARRIER
endif

PULP_LDFLAGS += -lgcc -lm -flto


include $(RULES_DIR)/pmsis_rules.mk



