#ifndef FFT_H
#define FFT_H

#if defined (MIXED_RADIX)
#include "config.h"
#define LOG2_FFT_LEN_RADIX8 (LOG2_FFT_LEN%3 == 0? LOG2_FFT_LEN: ((LOG2_FFT_LEN-1)%3 == 0? LOG2_FFT_LEN-1: LOG2_FFT_LEN-2))

#else
#define LOG2_FFT_LEN_RADIX8 9
#endif


#if (LOG2_FFT_LEN_RADIX8 % 3 != 0)
#pragma GCC error "For this input size you cannot use the radix-8 algorithm"
#endif

#define FFT_LEN_RADIX8        (1 << LOG2_FFT_LEN_RADIX8)

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
#define VDTYPE  v2f16
#endif
//#define VDTYPE  v2f16
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


void fft_radix8     (Complex_type * Inp_signal, Complex_type * Out_signal);
void par_fft_radix8 (Complex_type * Inp_signal, Complex_type * Out_signal);

#endif
