#ifndef _CONFIG_MATMUL_
#define _CONFIG_MATMUL_

#ifdef FABRIC
#define DATA_LOCATION
#else
#define DATA_LOCATION __attribute__((section(".data_l1")))
#endif

//Define INPUT data types

#ifdef FIXED
    #ifdef FP8
        typedef int8_t      v4s    __attribute__((vector_size (4)));
        typedef float8      MA_TYPE;
        typedef float8      MB_TYPE;
        typedef float8      OUT_TYPE;
        typedef float8      MA_VTYPE    __attribute__((vector_size (4)));
        typedef float8      MB_VTYPE    __attribute__((vector_size (4)));
        typedef float8     OUT_VTYPE    __attribute__((vector_size (4)));
        #undef USE_INTRINSICS
    #elif FP16
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      MA_TYPE;
        typedef float16      MB_TYPE;
        typedef float16      OUT_TYPE;
        typedef float16      MA_VTYPE    __attribute__((vector_size (4)));
        typedef float16      MB_VTYPE    __attribute__((vector_size (4)));
        typedef float16     OUT_VTYPE    __attribute__((vector_size (4)));
        #undef USE_INTRINSICS
    #elif defined(FP16ALT)
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      MA_TYPE;
        typedef float16alt      MB_TYPE;
        typedef float16alt      OUT_TYPE;
        typedef float16alt      MA_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt      MB_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt     OUT_VTYPE    __attribute__((vector_size (4)));
        #undef USE_INTRINSICS
    #elif defined(FP32)
        typedef float  MA_TYPE;
        typedef float  MB_TYPE;
        typedef float  OUT_TYPE;
    #endif

#else // MIXED
    // Define First Matrix (A) Data Types
    #ifdef MAFP32
        typedef float      MA_TYPE;
    #elif MAFP16
            #define FP16
            typedef signed short      v2s    __attribute__((vector_size (4)));
            typedef float16      MA_TYPE;
            typedef float16      MA_VTYPE    __attribute__((vector_size (4)));
            #undef USE_INTRINSICS
    #elif MAFP16ALT
            #define FP16ALT
            typedef signed short      v2s    __attribute__((vector_size (4)));
            typedef float16alt      MA_TYPE;
            typedef float16alt     MA_VTYPE    __attribute__((vector_size (4)));
            #undef USE_INTRINSICS
    #elif MAFP8
            #define FP8
            typedef int8_t      v4s    __attribute__((vector_size (4)));
            typedef float8      MA_TYPE;
            typedef float8     MA_VTYPE    __attribute__((vector_size (4)));
            #undef USE_INTRINSICS
    #endif

    // Define Second Matrix (B) Data Types
    #ifdef MBFP32
            typedef float   MB_TYPE;
    #elif MBFP16
            typedef signed short      v2s    __attribute__((vector_size (4)));
            typedef float16      MB_TYPE;
            typedef float16      MB_VTYPE    __attribute__((vector_size (4)));
            #undef USE_INTRINSICS
    #elif MBFP16ALT
            typedef signed short      v2s    __attribute__((vector_size (4)));
            typedef float16alt      MB_TYPE;
            typedef float16alt     MB_VTYPE    __attribute__((vector_size (4)));
            #undef USE_INTRINSICS
    #elif MBFP8
            typedef int8_t      v4s    __attribute__((vector_size (4)));
            typedef float8      MB_TYPE;
            typedef float8     MB_VTYPE    __attribute__((vector_size (4)));
            #undef USE_INTRINSICS
    #endif
        // Define Output Matrix (C) Data Types
    #ifdef OUTFP32
            typedef float OUT_TYPE;
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
#endif // MIXED



#define THR 0.000001f

// Mixed Vectorization can be used only if the first two operands (Matrix A and Matrix B) are of the same type
#ifdef VECTORIAL
    #if defined(MAFP16) && defined (MBFP16ALT) || defined (MAFP16ALT) && defined (MBFP16) 
        #error "Mixed Vectorization does not work for different data types...!!!" 
    #endif

    #if defined(MAFP16) && defined (MBFP8) || defined (MAFP8) && defined (MBFP16) 
        #error "Mixed Vectorization does not work for different data types...!!!" 
    #endif

    #if defined(MBFP16ALT) && defined (MBFP8) || defined (MAFP8) && defined (MBFP16ALT) 
        #error "Mixed Vectorization does not work for different data types...!!!" 
    #endif

    // Mixed precision cannot be used if one of the first two operands is FP32
    #if defined (MAFP32) || defined (MBFP32)
        #error "Mixed Vectorization does not work for FP32 data type...!!!" 
    #endif
#endif

void matMul(MA_TYPE * __restrict__ A, MB_TYPE * __restrict__ B, OUT_TYPE * __restrict__ C, int M, int N, int P);

#endif
