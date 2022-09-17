
#ifdef FP32
#define FLOAT float
#define THR 0.0001f
#elif defined(FP16)
#define FLOAT float16
#define THR 0.04f
#elif defined(FP16ALT)
#define FLOAT float16alt
#ifdef VECT
#define THR 0.0625f
#else
#define THR 0.0172f
#endif
#endif

#ifdef VECT
#define VECTOR_MODE
#endif
 
#ifdef FABRIC
#define DATA_LOCATION PI_L2
#else
#define DATA_LOCATION PI_CL_L1
#endif 

#ifdef VECT
    #if defined (FP32)

        #error "Vecotrization does not work for FP32 data type...!!!" 
    #endif
#endif