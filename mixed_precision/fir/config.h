#ifndef _CONFIG_FIR_
#define _CONFIG_FIR_


#ifdef VECT
#define VECTORIZATION
#endif

#define IS_FLOAT   1

#ifdef FIXED
    #ifdef FP16
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
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      INP_TYPE;
        typedef float16      INP_VTYPE    __attribute__((vector_size (4)));
    #elif INFP16ALT
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      INP_TYPE;
        typedef float16alt     INP_VTYPE    __attribute__((vector_size (4)));
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


#ifdef FABRIC
#define DATA_LOCATION
#else
#define DATA_LOCATION __attribute__((section(".data_l1")))
#endif

void convolve(
        INP_TYPE Signal[],
        FIL_TYPE *Filter, int FilterLength,
        OUT_TYPE *Output, int OutputLength);

#endif
