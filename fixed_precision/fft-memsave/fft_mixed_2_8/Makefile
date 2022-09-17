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

ifdef PRINT_RESULTS
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


ifdef vec
PULP_CFLAGS += -DVECTORIZATION
endif

ifdef CORES
PULP_CFLAGS += -DNUM_CORES=${CORES}
else
PULP_CFLAGS += -DNUM_CORES=1
endif

ifdef fmt
PULP_CFLAGS += -D${fmt}
else
PULP_CFLAGS += -DFP32
endif

PULP_LDFLAGS += -flto -Wl,--gc-sections -lm

include $(RULES_DIR)/pmsis_rules.mk
