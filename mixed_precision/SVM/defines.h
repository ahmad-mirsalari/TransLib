double __extendohfdf2(float16alt value);
double __extendhfdf2(float16 value);
#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <stdint.h>
#include "pmsis.h"


#ifdef FABRIC
#define DATA_LOCATION PI_L2
#else
#define DATA_LOCATION PI_L1
#endif

#ifdef VECT
#define VECTORIZATION
#endif

#ifdef FIXED
#ifdef FP16
typedef signed short v2s __attribute__((vector_size(4)));
typedef float16 INP_TYPE;
typedef float16 FIL_TYPE;
typedef float16 OUT_TYPE;
typedef float16 INP_VTYPE __attribute__((vector_size(4)));
typedef float16 FIL_VTYPE __attribute__((vector_size(4)));
typedef float16 OUT_VTYPE __attribute__((vector_size(4)));
typedef float16 v2f16 __attribute__((vector_size(4)));
#define BITS 16
#elif defined(FP8)
typedef int8_t v4s __attribute__((vector_size(4)));
typedef float8 INP_TYPE;
typedef float8 FIL_TYPE;
typedef float8 OUT_TYPE;
typedef float8 INP_VTYPE __attribute__((vector_size(4)));
typedef float8 FIL_VTYPE __attribute__((vector_size(4)));
typedef float8 OUT_VTYPE __attribute__((vector_size(4)));
typedef float8 v2f8 __attribute__((vector_size(4)));
#define BITS 8
#elif defined(FP16ALT)
typedef signed short v2s __attribute__((vector_size(4)));
typedef float16alt INP_TYPE;
typedef float16alt FIL_TYPE;
typedef float16alt OUT_TYPE;
typedef float16alt INP_VTYPE __attribute__((vector_size(4)));
typedef float16alt FIL_VTYPE __attribute__((vector_size(4)));
typedef float16alt OUT_VTYPE __attribute__((vector_size(4)));
typedef float16alt v2f16 __attribute__((vector_size(4)));
#define BITS 16
#elif defined(FP32)
typedef float INP_TYPE;
typedef float FIL_TYPE;
typedef float OUT_TYPE;
typedef float16 v2f16 __attribute__((vector_size(4)));
#define BITS 32
#undef DOTP
#endif

#else // MIXED
#ifdef INFP32
    typedef float INP_TYPE;
    #define BITS 32
    // typedef float16 v2f16 __attribute__((vector_size(4)));
#elif INFP16
    #define FP16
    #define BITS 16
    typedef signed short v2s __attribute__((vector_size(4)));
    typedef float16 INP_TYPE;
    typedef float16 INP_VTYPE __attribute__((vector_size(4)));
    typedef float16 v2f16 __attribute__((vector_size(4)));
#elif INFP16ALT
    #define FP16ALT
    #define BITS 16
    typedef signed short v2s __attribute__((vector_size(4)));
    typedef float16alt INP_TYPE;
    typedef float16alt INP_VTYPE __attribute__((vector_size(4)));
    typedef float16alt v2f16 __attribute__((vector_size(4)));
#elif INFP8
    #define FP8
    #define BITS 8
    typedef int8_t v4s __attribute__((vector_size(4)));
    typedef float8 INP_TYPE;
    typedef float8 INP_VTYPE __attribute__((vector_size(4)));
    typedef float8 v2f8 __attribute__((vector_size(4)));
#endif
// Define Filter data types
#ifdef FILFP32
    typedef float FIL_TYPE;
#elif FILFP16
    typedef signed short v2s __attribute__((vector_size(4)));
    typedef float16 FIL_TYPE;
    typedef float16 FIL_VTYPE __attribute__((vector_size(4)));
#elif FILFP16ALT
    typedef signed short v2s __attribute__((vector_size(4)));
    typedef float16alt FIL_TYPE;
    typedef float16alt FIL_VTYPE __attribute__((vector_size(4)));
#elif FILFP8
    typedef int8_t v4s __attribute__((vector_size(4)));
    typedef float8 FIL_TYPE;
    typedef float8 FIL_VTYPE __attribute__((vector_size(4)));
#endif
// Define output data types
#ifdef OUTFP32
    typedef float OUT_TYPE;
#elif OUTFP16
    typedef signed short v2s __attribute__((vector_size(4)));
    typedef float16 OUT_TYPE;
    typedef float16 OUT_VTYPE __attribute__((vector_size(4)));
#elif OUTFP16ALT
    typedef signed short v2s __attribute__((vector_size(4)));
    typedef float16alt OUT_TYPE;
    typedef float16alt OUT_VTYPE __attribute__((vector_size(4)));
#elif OUTFP8
    typedef int8_t v4s __attribute__((vector_size(4)));
    typedef float8 OUT_TYPE;
    typedef float8 OUT_VTYPE __attribute__((vector_size(4)));
#endif
    // Check if the user is using vectorization in Mixed precision case
    #ifdef VECT
        #define MIXED_VECTOR
    #endif
#endif //END of MIXED

#define THR 0.000001f

// Mixed Vectorization can be used only if the first two operands (INP and FIL) are of the same type
#ifdef VECT
    #if defined(INFP16) && defined (FILFP16ALT) || defined (INFP16ALT) && defined (FILFP16) 
        #error "Mixed Vectorization does not work for different data types...!!!" 
    #endif

    #if defined(INFP16) && defined (FILFP8) || defined (INFP8) && defined (FILFP16) 
        #error "Mixed Vectorization does not work for different data types...!!!" 
    #endif

    #if defined(INFP16ALT) && defined (FILFP8) || defined (INFP16ALT) && defined (FILFP8) 
        #error "Mixed Vectorization does not work for different data types...!!!" 
    #endif
    #if defined (INFP32) || defined (FILFP32)

        #error "Vectorization does not work for FP32 data type...!!!" 
    #endif
#endif



#define SCALE 1

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#endif
