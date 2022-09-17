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

//#define ELEMENT(a,stride,i) ((a)[(stride)*(i)])
#ifdef FP32
//#define DATA_TYPE_SIZE 32
#define DATA_TYPE float
#define THR 0.0001f
#elif defined(FP16)
#define DATA_TYPE float16
typedef float16      VDTYPE    __attribute__((vector_size (4)));
#define THR 0.0004f
#elif defined(FP16ALT)
//#define DATA_TYPE_SIZE 16
#define DATA_TYPE float16alt
typedef float16alt      VDTYPE    __attribute__((vector_size (4)));
#define THR 0.0005f
#endif

#ifdef VEC
#define VECTORIZATION
#endif


static void dwt_step (DATA_TYPE *a, DATA_TYPE *output, size_t n, size_t output_dim);
int gsl_wavelet_transform (DATA_TYPE *input, DATA_TYPE *output, size_t n);