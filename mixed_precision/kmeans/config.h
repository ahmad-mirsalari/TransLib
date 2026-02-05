#ifdef FIXED
    #ifdef FP8
        typedef int8_t      v4s    __attribute__((vector_size (4)));
        typedef float8      INP_TYPE;
        typedef float8      OUT_TYPE;
        typedef float8      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float8     OUT_VTYPE    __attribute__((vector_size (4)));
    #elif defined(FP16)
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16      INP_TYPE;
        typedef float16      OUT_TYPE;
        typedef float16      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16     OUT_VTYPE    __attribute__((vector_size (4)));
    #elif defined(FP16ALT)
        typedef signed short      v2s    __attribute__((vector_size (4)));
        typedef float16alt      INP_TYPE;
        typedef float16alt      OUT_TYPE;
        typedef float16alt      INP_VTYPE    __attribute__((vector_size (4)));
        typedef float16alt     OUT_VTYPE    __attribute__((vector_size (4)));
    #elif defined(FP32)
        typedef float  INP_TYPE;
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
    #ifdef VECT
        #error "Vectorization does not work for Mixed precision...!!!"
    #endif
#endif

#ifdef FABRIC
#define DATA_LOCATION PI_L2
#else
#define DATA_LOCATION PI_CL_L1
#endif

#define THR 0.000001f

#ifdef VECT
#define VECTOR_MODE
#endif

