ifndef target
target = pulp
endif

# Use 'target = gap9' to compile for GAP9
ifeq (${target}, gap9)
APP = dwt

ifdef FABRIC
APP_SRCS  = main.c wavelet.c
APP_CFLAGS += -DFABRIC
else
APP_SRCS  = main.c wavelet.c barrier.c
endif

ifndef cores
cores = 1
else
APP_CFLAGS += -DPARALLEL
endif


ifdef BOARD
APP_CFLAGS += -DBOARD
endif

APP_CFLAGS += -O3 -g -flto 
#-mno-memcpy -ftree-loop-distribution -ftree-loop-distribute-patterns -fno-tree-loop-im -fno-tree-ch

ifeq (${OPT}, 0)
APP_CL_ARCH_CFLAGS = -march=rv32imfc -DRV_ISA_RV32=1   -D__riscv__
APP_ARCH_LDFLAGS = -march=rv32imfc -DRV_ISA_RV32=1  -D__riscv__
APP_CFLAGS += -DNUM_CORES=1
endif

ifdef cores
APP_CFLAGS += -DNUM_CORES=${cores}
endif

ifdef stats
APP_CFLAGS += -DSTATS
endif

ifdef check
APP_CFLAGS += -DCHECK
endif
# CHECK RESULTS
#PULP_CFLAGS += -DDEBUG
ifdef verbose
APP_CFLAGS += -DVERBOSE
endif
ifdef print_results
APP_CFLAGS += -DPRINT_RESULTS
endif

# FP FORMAT
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

ifdef vec
APP_CFLAGS += -DVECTORIAL
endif
#PULP_CFLAGS  += -O3 -DNUM_CORES=${CORES}
APP_LDFLAGS += -lm

ifeq (${BARRIER}, sw)
APP_CFLAGS += -DSW_BARRIER
endif

ifeq (${BARRIER}, tas)
APP_CFLAGS += -DTAS_BARRIER
endif

APP_LDFLAGS += -lgcc -lm -flto


include $(RULES_DIR)/pmsis_rules.mk


else
# Use 'target = pulp' to compile for PULP
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
ifdef print_results
PULP_CFLAGS += -DPRINT_RESULTS
endif

# FP FORMAT
ifdef fmt
PULP_CFLAGS += -D${fmt} -DFIXED

else
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

ifdef vec
PULP_CFLAGS += -DVECTORIAL
endif

# Mixed precision HW support
ifdef hwmixed 
PULP_CFLAGS += -DHWMIXED
PULP_CFLAGS += -mfaux
endif

ifdef no_fmadd
PULP_CFLAGS += -ffp-contract=off
endif

#PULP_CFLAGS  += -O3 -DNUM_CORES=${CORES}
# PULP_LDFLAGS += -lm

ifeq (${BARRIER}, sw)
PULP_CFLAGS += -DSW_BARRIER
endif


ifeq (${BARRIER}, tas)
PULP_CFLAGS += -DTAS_BARRIER
endif

PULP_LDFLAGS += -lgcc -lm -flto


include $(RULES_DIR)/pmsis_rules.mk
endif

