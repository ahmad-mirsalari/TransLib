#include "pmsis.h"
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "config.h"


#ifdef FABRIC
#define DATA_LOCATION PI_L2
#else
#define DATA_LOCATION PI_CL_L1
#endif
#define THR 0.000001f


//#define ELEMENT(a,stride,i) ((a)[(stride)*(i)])
/*#ifdef FP32
#define DATA_TYPE float

#elif defined(FP16)
#define DATA_TYPE float16
typedef float16      VDTYPE    __attribute__((vector_size (4)));

#elif defined(FP8)
#define DATA_TYPE float8
typedef float8      VDTYPE    __attribute__((vector_size (4)));

#elif defined(FP16ALT)
//#define DATA_TYPE_SIZE 16
#define DATA_TYPE float16alt
typedef float16alt      VDTYPE    __attribute__((vector_size (4)));
#endif*/

#ifdef FIXED

    #ifdef FP8
        typedef int8_t      v4s    __attribute__((vector_size (4)));
        typedef float8      INP_TYPE;
        typedef float8      FIL_TYPE;
        typedef float8      OUT_TYPE;
        typedef float8      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float8      FIL_VTYPE    __attribute__((vector_size (4)));
        typedef float8     OUT_VTYPE    __attribute__((vector_size (4)));
    #elif defined(FP16)
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      INP_TYPE;
        typedef float16      FIL_TYPE;
        typedef float16      OUT_TYPE;
        typedef float16      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16      FIL_VTYPE    __attribute__((vector_size (4)));
        typedef float16     OUT_VTYPE    __attribute__((vector_size (4)));
    #elif defined(FP16ALT)
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      INP_TYPE;
        typedef float16alt      FIL_TYPE;
        typedef float16alt      OUT_TYPE;
        typedef float16alt      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt      FIL_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt     OUT_VTYPE    __attribute__((vector_size (4)));
    #elif defined(FP32)
        typedef float  INP_TYPE;
        typedef float  FIL_TYPE;
        typedef float  OUT_TYPE;
        #define DATA_WIDTH 32
        #undef DOTP
    #endif

#else // MIXED
    #ifdef INFP32
        typedef float      INP_TYPE;
        #define DATA_WIDTH  32
    #elif INFP16
        #define FP16
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      INP_TYPE;
        typedef float16      INP_VTYPE    __attribute__((vector_size (4)));
    #elif INFP16ALT
        #define FP16ALT
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      INP_TYPE;
        typedef float16alt     INP_VTYPE    __attribute__((vector_size (4)));
    #elif INFP8
        #define FP8
        typedef int8_t      v4s    __attribute__((vector_size (4)));
        typedef float8      INP_TYPE;
        typedef float8     INP_VTYPE    __attribute__((vector_size (4)));
    #endif
    // Define Filter data types
    #ifdef FILFP32
        typedef float      FIL_TYPE;
        #define DATA_WIDTH 32
    #elif FILFP16
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      FIL_TYPE;
        typedef float16      FIL_VTYPE    __attribute__((vector_size (4)));
    #elif FILFP16ALT
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      FIL_TYPE;
        typedef float16alt     FIL_VTYPE    __attribute__((vector_size (4)));
    #elif FILFP8
        typedef int8_t      v4s    __attribute__((vector_size (4)));
        typedef float8      FIL_TYPE;
        typedef float8     FIL_VTYPE    __attribute__((vector_size (4)));
    #endif
    // Define output data types
    #ifdef OUTFP32
        typedef float      OUT_TYPE;
    #elif OUTFP16
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      OUT_TYPE;
        typedef float16      OUT_VTYPE    __attribute__((vector_size (4)));
    #elif OUTFP16ALT
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      OUT_TYPE;
        typedef float16alt     OUT_VTYPE    __attribute__((vector_size (4)));
    #elif OUTFP8
        typedef int8_t      v4s    __attribute__((vector_size (4)));
        typedef float8      OUT_TYPE;
        typedef float8     OUT_VTYPE    __attribute__((vector_size (4)));
    #endif

    // Check if the user is using vectorization in Mixed precision case
    #ifdef VECTORIAL
        #define MIXED_VECTOR
    #endif
#endif


#ifdef VECTORIAL
#define VECTORIZATION
#endif


#ifdef VECTORIAL
#if defined(FP8)
#error "Vectorization does not work for FP8 data type...!!!"
#endif
#endif

static void dwt_step (INP_TYPE *a, OUT_TYPE *output, size_t n, size_t output_dim);
int gsl_wavelet_transform (INP_TYPE *input, OUT_TYPE *output, size_t n);