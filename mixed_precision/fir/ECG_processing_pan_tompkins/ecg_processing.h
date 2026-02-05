#ifndef ECG_PROCESSING_H
#define ECG_PROCESSING_H

#ifdef FABRIC
#define DATA_LOCATION PI_L2
#else
#define DATA_LOCATION PI_L1
#endif

#define STACK_SIZE 2048
#define MOUNT 1
#define UNMOUNT 0
#define CID 0

#ifdef VECTORIAL

#ifdef FP16
#define FP16_VECTORIATION
typedef float16 dataType; // int or float
typedef float16 qrsdataType;
typedef float16 hrdataType;
typedef signed short v2s __attribute__((vector_size(4)));
typedef float16 INP_TYPE;
typedef float16 FIL_TYPE;
typedef float16 OUT_TYPE;
typedef float16 INP_VTYPE __attribute__((vector_size(4)));
typedef float16 FIL_VTYPE __attribute__((vector_size(4)));
typedef float16 OUT_VTYPE __attribute__((vector_size(4)));
typedef enum
{
    false,
    true
} bool_enum;
#elif FP16ALT
#define FP16_VECTORIATION
typedef float16alt dataType; // int or float
typedef float16alt qrsdataType;
typedef float16alt hrdataType;
typedef signed short v2s __attribute__((vector_size(4)));
typedef float16alt INP_TYPE;
typedef float16alt FIL_TYPE;
typedef float16alt OUT_TYPE;
typedef float16alt INP_VTYPE __attribute__((vector_size(4)));
typedef float16alt FIL_VTYPE __attribute__((vector_size(4)));
typedef float16alt OUT_VTYPE __attribute__((vector_size(4)));
typedef enum
{
    false,
    true
} bool_enum;
#elif FP8
#define FP8_VECTORIATION
typedef float8 dataType; //  ***************** ATTENTION: IN FP8, WE USE FP16 FOR THE SIGNAL DATA ****************
typedef float16 qrsdataType;
typedef float16 hrdataType;
typedef float8 v2f16 __attribute__((vector_size(4))); // ***************** ATTENTION: IN FP8, WE USE FP16 FOR THE SIGNAL DATA ****************
typedef signed short v2s __attribute__((vector_size(4)));
typedef float8 INP_TYPE;
typedef float8 FIL_TYPE;
typedef float8 OUT_TYPE;
typedef float8 INP_VTYPE __attribute__((vector_size(4)));
typedef float8 FIL_VTYPE __attribute__((vector_size(4)));
typedef float8 OUT_VTYPE __attribute__((vector_size(4)));
typedef enum
{
    false,
    true
} bool_enum;
#endif
#else // NO_VECTORIAL
#ifdef FP32
typedef float dataType; // int or float
typedef float qrsdataType;
typedef float hrdataType;
typedef signed short v2s __attribute__((vector_size(4)));
typedef float INP_TYPE;
typedef float FIL_TYPE;
typedef float OUT_TYPE;
typedef enum
{
    false,
    true
} bool_enum;

#elif FP16

typedef float16 dataType; // int or float
typedef float16 qrsdataType;
typedef float16 hrdataType;
typedef signed short v2s __attribute__((vector_size(4)));
typedef float16 INP_TYPE;
typedef float16 FIL_TYPE;
typedef float16 OUT_TYPE;
typedef float16 INP_VTYPE __attribute__((vector_size(4)));
typedef float16 FIL_VTYPE __attribute__((vector_size(4)));
typedef float16 OUT_VTYPE __attribute__((vector_size(4)));
typedef enum
{
    false,
    true
} bool_enum;
#elif FP16ALT
typedef float16alt dataType; // int or float
typedef float16alt qrsdataType;
typedef float16alt hrdataType;
typedef signed short v2s __attribute__((vector_size(4)));
typedef float16alt INP_TYPE;
typedef float16alt FIL_TYPE;
typedef float16alt OUT_TYPE;
typedef float16alt INP_VTYPE __attribute__((vector_size(4)));
typedef float16alt FIL_VTYPE __attribute__((vector_size(4)));
typedef float16alt OUT_VTYPE __attribute__((vector_size(4)));
typedef enum
{
    false,
    true
} bool_enum;
#else // FP8
typedef float8 dataType; // int or float
typedef float8 qrsdataType;
typedef float16 hrdataType;
typedef signed short v2s __attribute__((vector_size(4)));
typedef float8 INP_TYPE;
typedef float8 FIL_TYPE;
typedef float8 OUT_TYPE;
typedef float8 INP_VTYPE __attribute__((vector_size(4)));
typedef float8 FIL_VTYPE __attribute__((vector_size(4)));
typedef float8 OUT_VTYPE __attribute__((vector_size(4)));
typedef enum
{
    false,
    true
} bool_enum;

#endif
#endif

#define N 7443 // 4096//5528    // ECG input length

#ifdef VECTORIAL
#ifdef FP16_VECTORIATION
#define NC_Lo 14  // Number of low-pass filter components
#define NC_Hi 34  // Number of high-pass filter components
#define NC_Der 6  // Number of components in the derivative transfer function
#define NC_Int 32 // Number of components in the window integration

#elif defined(FP8_VECTORIATION)
#define NC_Lo 16  // Number of low-pass filter components
#define NC_Hi 36  // Number of high-pass filter components
#define NC_Der 8  // Number of components in the derivative transfer function
#define NC_Int 32 // Number of components in the window integration

#endif
#else
#define NC_Lo 13  // Number of low-pass filter components
#define NC_Hi 33  // Number of high-pass filter components
#define NC_Der 5  // Number of components in the derivative transfer function
#define NC_Int 31 // Number of components in the window integration

#endif


#define DELAY 52       // 41 //52

// #define FS 128         // 360			// Sampling frequency
// #define MAX_VALUE 4331 // 2282 //1311  // Max input value
// #define N_AVG 8
// #define MAX_SAMPLES 65536
// #define BUFFER_SIZE 205 //(FS*1.6)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))



#endif
