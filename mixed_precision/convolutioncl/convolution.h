// Copyright 2017 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef _CONF_HEADER_
#define _CONF_HEADER_

#include "config.h"

/*#if DATA_TYPE_SIZE == 16
    typedef signed short      v2s    __attribute__((vector_size (4)));
    typedef float16alt      Filtc;
    typedef float16alt      Pixel;
    typedef float16alt      FiltcV    __attribute__((vector_size (4)));
    typedef float16alt      PixelV    __attribute__((vector_size (4)));
#else
    typedef float  FLOAT;
    typedef float  Filtc;
    typedef float  Pixel;
    #undef DOTP
    #endif*/

//Define INPUT data types


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
        #undef DOTP
    #endif

#else // MIXED
    #ifdef INFP32
        typedef float      INP_TYPE;
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

#ifdef VECTORIAL
    #if defined(INFP16) && defined (FILFP16ALT) || defined (INFP16ALT) && defined (FILFP16) 
        #error "Vecotrization does not work for different data types...!!!" 
    #endif

    #if defined (INFP32) || defined (FILFP32) || defined (OUTFP32)

        #error "Vecotrization does not work for FP32 data type...!!!" 
    #endif
#endif


#define THR 0.0001f


void Conv_Scalar     (INP_TYPE * In, OUT_TYPE * Out,FIL_TYPE  *  Kernel, int R, int C, int S,int IC, int FW);
void Conv5x5_Vector     (INP_TYPE * In, OUT_TYPE * Out, int R, int C, int IC, FIL_TYPE  * Kernel);
void Conv3x3_Vector     (INP_TYPE * In, OUT_TYPE * Out, int R, int C,int IC, FIL_TYPE  * Kernel);
int check_result       (OUT_TYPE * result, int SIZE);


#ifdef VEC
#define VECTORIZATION
#endif

#ifdef FABRIC
#define DATA_LOCATION
#else
#define DATA_LOCATION __attribute__((section(".data_l1")))
#endif
#endif
