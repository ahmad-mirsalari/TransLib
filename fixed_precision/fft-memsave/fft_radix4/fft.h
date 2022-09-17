#ifndef FFT_H
#define FFT_H

#define LOG2_FFT_LEN_RADIX4 10


#if (LOG2_FFT_LEN_RADIX4 % 2 == 1)
#pragma GCC error "For this input size you cannot use the radix-4 algorithm"
#endif

#define FFT_LEN_RADIX4        (1 << LOG2_FFT_LEN_RADIX4)

#define SORT_OUTPUT
#define BITREV_LUT

#define STATS
// #define BOARD

// #define PRINT_RESULTS


#define DTYPE float
#define VDTYPE v2f16
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

// struct Complex_type_str {
//   DTYPE re;
//   DTYPE im;
// };
typedef union Complex_type_union Complex_type;

void fft_radix4     (Complex_type * Inp_signal, Complex_type * Out_signal);
void par_fft_radix4 (Complex_type * Inp_signal, Complex_type * Out_signal);

#endif
