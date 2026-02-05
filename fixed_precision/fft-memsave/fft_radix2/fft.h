#ifndef FFT_H
#define FFT_H

#ifdef FABRIC
#define DATA_LOCATION
#else
#define DATA_LOCATION __attribute__((section(".data_l1")))
#endif

#define LOG2_FFT_LEN_RADIX2     11
#define FFT_LEN_RADIX2          (1 << LOG2_FFT_LEN_RADIX2)

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

void fft_radix2     (Complex_type * Inp_signal, Complex_type * Out_signal);
void par_fft_radix2 (Complex_type * Inp_signal, Complex_type * Out_signal);

#endif
