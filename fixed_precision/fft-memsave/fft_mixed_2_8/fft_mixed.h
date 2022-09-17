
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "config.h"

#ifndef FFT_H
#define FFT_H

#ifdef FABRIC
#define DATA_LOCATION
#else
#define DATA_LOCATION __attribute__((section(".data_l1")))
#endif




#if (LOG2_FFT_LEN % 3 == 2)
#define STAGE_2_2_8
#endif

#if (LOG2_FFT_LEN % 3 == 0)
#define RADIX_8
#endif

#define FFT_LEN              (1 << LOG2_FFT_LEN)
#define LOG2_FFT_LEN_RADIX2  LOG2_FFT_LEN
#define FFT_LEN_RADIX2       FFT_LEN


#define LOG2_FFT_LEN_RADIX8 (LOG2_FFT_LEN%3 == 0? LOG2_FFT_LEN: ((LOG2_FFT_LEN-1)%3 == 0? LOG2_FFT_LEN-1: LOG2_FFT_LEN-2))

#define FFT_LEN_RADIX8       (1 << LOG2_FFT_LEN_RADIX8)


#define FULL_TWIDDLES

//#define SORT_OUTPUT
#define BITREV_LUT


#ifdef FP32

#define DTYPE float
#define VDTYPE  v2f16

#elif defined(FP16)

#define DTYPE float16
#define VDTYPE  v2f16

#elif defined(FP16ALT)

#define DTYPE float16alt
#define VDTYPE  v2f16alt
#endif
//#define VECTORIZATION


typedef float16  v2f16  __attribute__ ((vector_size (4)));
typedef float16alt  v2f16alt  __attribute__ ((vector_size (4)));
typedef short    v2s    __attribute__ ((vector_size (4)));

union Complex_type_union {
  struct{
    DTYPE re;
    DTYPE im;
  };
#ifdef VECTORIZATION
  VDTYPE v;
#endif
};

typedef union Complex_type_union Complex_type;


void fft_mixed_2_8     (Complex_type * Inp_signal, Complex_type * Out_signal);
void par_fft_mixed_2_8 (Complex_type * Inp_signal, Complex_type * Out_signal);

#endif
