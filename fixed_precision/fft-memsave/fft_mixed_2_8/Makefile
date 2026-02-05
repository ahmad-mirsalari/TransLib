ifndef target
target = pulp
endif

# Use 'target = gap9' to compile for GAP9
ifeq ($(target), gap9)
APP = test

ifdef FABRIC
APP_SRCS  = main.c ../fft_radix2/fft_radix2.c ../fft_radix2/utils.c ../fft_radix8/utils.c fft_mixed_2_8.c utils.c
else
APP_SRCS  = main.c ../fft_radix2/fft_radix2.c ../fft_radix2/utils.c ../fft_radix8/utils.c fft_mixed_2_8.c utils.c 
endif

APP_CFLAGS += -O3 -g -DMIXED_RADIX -flto

ifdef FABRIC
APP_CFLAGS += -DFABRIC
endif

ifdef BITREV_LUT
APP_CFLAGS += -DBITREV_LUT
endif

ifdef print_results
APP_CFLAGS += -DPRINT_RESULTS
endif

ifdef FULL_TWIDDLES
APP_CFLAGS += -DFULL_TWIDDLES
endif


ifdef SORT_OUTPUT
APP_CFLAGS += -DSORT_OUTPUT
endif

ifdef stats
APP_CFLAGS += -DSTATS
endif

ifdef check
APP_CFLAGS += -DCHECK
endif

ifdef vec
APP_CFLAGS += -DVECTORIZATION
endif

ifdef cores
APP_CFLAGS += -DNUM_CORES=${cores}
else
APP_CFLAGS += -DNUM_CORES=1
endif

ifdef fmt
APP_CFLAGS += -D${fmt}
else
APP_CFLAGS += -DFP32
endif

APP_LDFLAGS += -flto -lm

include $(RULES_DIR)/pmsis_rules.mk

else # Use 'target = pulp' to compile for PULP
PULP_APP = test

ifdef FABRIC
PULP_APP_FC_SRCS  = main.c ../fft_radix2/fft_radix2.c ../fft_radix2/utils.c ../fft_radix8/utils.c fft_mixed_2_8.c utils.c
else
PULP_APP_FC_SRCS  = main.c
PULP_APP_SRCS  = ../fft_radix2/fft_radix2.c ../fft_radix2/utils.c ../fft_radix8/utils.c fft_mixed_2_8.c utils.c 
endif

PULP_CFLAGS += -O3 -g -DMIXED_RADIX -flto

ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifdef BITREV_LUT
PULP_CFLAGS += -DBITREV_LUT
endif

ifdef print_results
PULP_CFLAGS += -DPRINT_RESULTS
endif

ifdef FULL_TWIDDLES
PULP_CFLAGS += -DFULL_TWIDDLES
endif


ifdef SORT_OUTPUT
PULP_CFLAGS += -DSORT_OUTPUT
endif

ifdef stats
PULP_CFLAGS += -DSTATS
endif

ifdef check
PULP_CFLAGS += -DCHECK
endif

ifdef vec
PULP_CFLAGS += -DVECTORIZATION
endif

ifdef cores
PULP_CFLAGS += -DNUM_CORES=${cores}
else
PULP_CFLAGS += -DNUM_CORES=1
endif

ifdef fmt
PULP_CFLAGS += -D${fmt}
else
PULP_CFLAGS += -DFP32
endif

ifdef no_fmadd
PULP_CFLAGS += -ffp-contract=off
endif
PULP_LDFLAGS += -flto -Wl,--gc-sections -lm

include $(RULES_DIR)/pmsis_rules.mk

endif