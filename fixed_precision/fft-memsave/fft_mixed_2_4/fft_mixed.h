#ifndef FFT_H
#define FFT_H

#define LOG2_FFT_LEN     11

#if (LOG2_FFT_LEN % 2 == 0)
#pragma GCC error "For this FFT size you have to use radix-4"
#endif

#define FFT_LEN              (1 << LOG2_FFT_LEN)
#define LOG2_FFT_LEN_RADIX2  LOG2_FFT_LEN
#define FFT_LEN_RADIX2       FFT_LEN
#define LOG2_FFT_LEN_RADIX4  (LOG2_FFT_LEN - 1)
#define FFT_LEN_RADIX4       (FFT_LEN/2)

#define SORT_OUTPUT
#define BITREV_LUT

#define STATS
// #define BOARD

// #define PRINT_RESULTS



#define DTYPE   float16
#define VDTYPE  v2f16
#define VECTORIZATION


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

void fft_mixed_2_4     (Complex_type * Inp_signal, Complex_type * Out_signal);
void par_fft_mixed_2_4 (Complex_type * Inp_signal, Complex_type * Out_signal);

#endif
