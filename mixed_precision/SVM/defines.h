double __extendohfdf2(float16alt value);
double __extendhfdf2(float16 value);
#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <stdint.h>
#include "pmsis.h"

#ifdef FABRIC
#define DATA_LOCATION PI_L2
#else
#define DATA_LOCATION PI_CL_L1
#endif 
 


#ifdef VECT
#define VECTORIZATION
#endif



#ifdef FIXED
    #ifdef FP16
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      INP_TYPE;
        typedef float16      FIL_TYPE;
        typedef float16      OUT_TYPE;
        typedef float16      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16      FIL_VTYPE    __attribute__((vector_size (4)));
        typedef float16     OUT_VTYPE    __attribute__((vector_size (4)));
        typedef float16  v2f16  __attribute__ ((vector_size (4)));
        #define BITS 16
    #elif defined(FP16ALT)
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      INP_TYPE;
        typedef float16alt      FIL_TYPE;
        typedef float16alt      OUT_TYPE;
        typedef float16alt      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt      FIL_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt     OUT_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt  v2f16  __attribute__ ((vector_size (4)));
        #define BITS 16
    #elif defined(FP32)
        typedef float  INP_TYPE;
        typedef float  FIL_TYPE;
        typedef float  OUT_TYPE;
        typedef float16  v2f16  __attribute__ ((vector_size (4)));
        #define BITS 32
        #undef DOTP
    #endif

#else // MIXED
    #ifdef INFP32
        typedef float      INP_TYPE;
        #define BITS 32
        typedef float16  v2f16  __attribute__ ((vector_size (4)));
    #elif INFP16
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      INP_TYPE;
        typedef float16      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16  v2f16  __attribute__ ((vector_size (4)));
        #define BITS 16
    #elif INFP16ALT
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      INP_TYPE;
        typedef float16alt     INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt  v2f16  __attribute__ ((vector_size (4)));
        #define BITS 16
    #endif
    // Define Filter data types
    #ifdef FILFP32
        typedef float      FIL_TYPE;
    #elif FILFP16
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      FIL_TYPE;
        typedef float16      FIL_VTYPE    __attribute__((vector_size (4)));
    #elif FILFP16ALT
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      FIL_TYPE;
        typedef float16alt     FIL_VTYPE    __attribute__((vector_size (4)));
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
    #endif
#endif

#ifdef VECT
    #if defined(INFP16) && defined (FILFP16ALT) || defined (INFP16ALT) && defined (FILFP16) 
        #error "Vecotrization does not work for different data types...!!!" 
    #endif

    #if defined (INFP32) || defined (FILFP32) || defined (OUTFP32)

        #error "Vecotrization does not work for FP32 data type...!!!" 
    #endif
#endif


#define THR 0.0001f



#define SCALE 1


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#endif
