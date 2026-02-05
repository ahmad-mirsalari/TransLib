/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_SVM_predict.c
 * Description:  Support Vector Machine (SVM) prediction.
 *
 * $Date:        20. December 2019
 * $Revision:    V0
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2019 ETH Zurich and University of Bologna. All rights reserved.
 *
 * Author: Fabio Montagna, University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "plp_dot_prod_f32.h"
#include "plp_SVM_predict.h"
#include "configs.h"
#include "defines.h"
#include "plp_exp.h"
#define FSIGN(x) (*((int *)(&x)) >> 31)
#define V2FSIGN(x) __builtin_pulp_sra2(*((v4s *)(&x)), (v2s){15, 15})

#define GIST_A 12102203.17133801f
#define GIST_B 1064986823.010288f
#define GIST_C 8388608
#define GIST_D 2139095040

#define GIST16_A 1477.0f  // 1024 / ln(2)
#define GIST16_B 15360.0f // 15 << 10
#define GIST16_C 0        // Lower clamp (zero exponent)
#define GIST16_D 31743    // 0x7BFF, max finite float16 (non-Inf)

#define GIST8_E5M2_A 288.539f
#define GIST8_E5M2_B 60.0f
#define GIST8_E5M2_C 0
#define GIST8_E5M2_D 0x7F

float fastexp_gist(float x)
{
    x = GIST_A * x + GIST_B;

    if (x < GIST_C || x > GIST_D)
        x = (x < GIST_C) ? 0.0f : GIST_D;

    uint32_t n = (uint32_t)(x);
    return *(float *)&n;
}

float16 fastexp_gist_f16(float16 x)
{
    float16 fx = GIST16_A * x + GIST16_B;

    if (fx < GIST16_C || fx > GIST16_D)
        fx = (fx < GIST16_C) ? 0.0f : GIST16_D;

    uint16_t n = (uint16_t)(fx);
    return *(float16 *)&n;
}

#if BITS == 8
float8 fastexp_gist_f8(float8 x)
{
    float8 v = GIST8_E5M2_A * x + GIST8_E5M2_B;

    if (v < GIST8_E5M2_C || v > GIST8_E5M2_D)
        v = (v < GIST8_E5M2_C) ? 0.0f : GIST8_E5M2_D;

    uint8_t n = (uint8_t)(v);
    return *(float8 *)&n;
}
#endif

static OUT_TYPE approx_exp(OUT_TYPE x)
{
    return 1 + x + (x * x) * 0.5f;
}
// static OUT_TYPE approx_exp(OUT_TYPE x) {
//     OUT_TYPE x2 = x * x;      // x^2
//     OUT_TYPE x3 = x2 * x;    // x^3
//     OUT_TYPE x4 = x2 * x2;   // x^4
//     return 1 + x + x2 * 0.5f + x3 * (1.0f / 6.0f) + x4 * (1.0f / 24.0f);
// }
#if defined(FP16) || defined(FP16ALT)
static inline v2f16 pack_f16(float16 a, float16 b)
{
    v2f16 result;
    __asm__ __volatile__("pv.pack.h %0, %1, %2" : "=f"(result) : "f"(a), "f"(b) :);
    return result;
}
#endif


#ifdef LINEAR_KERNEL

#ifdef VECTORIZATION

#ifdef FP8 // Vectorized FP8 Linear 

void plp_SVM_linear(svm_model model, INP_TYPE *data_model, int *Pred, FIL_TYPE *sv_coef, FIL_TYPE *bias)
{

    #ifdef MIXED_VECTOR
    float temp = 0.0f;
    #else
    OUT_VTYPE temp;
    #endif
    INP_VTYPE Av;
    INP_VTYPE Av1;
    FIL_VTYPE Bv0;
    FIL_VTYPE Bv1;
    int j;
    int i = 0;

    int l = model.SVS;
    int f_dim = model.F_DIM;
    int nr_class = model.N_CLASS;
    #ifndef FABRIC
        int core_id = pi_core_id();
    #else
        int core_id = 0;
    #endif

    int blockSize = (l + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * blockSize;
    int end = start + blockSize < l ? start + blockSize : l;
    int full_chunks = f_dim / 4; // number of full chunks
    int processed_elements = 4 * full_chunks;
    for (int i = start; i < end; i++)
    {
        // for (int j = 0; j < nr_class-1; j++) {
        int j = 0;

        #ifdef MIXED_VECTOR
            temp = 0.0f;
        #else
            temp = (OUT_VTYPE){0, 0};
        #endif

        #ifdef FDIM_GT_32
            // Manual unrolling
            for (int k = 0; k < processed_elements; k += 8)
            {
                Av = *((INP_VTYPE *)&data_model[i * f_dim + k]);
                Av1 = *((INP_VTYPE *)&data_model[i * f_dim + k + 4]);
                Bv0 = *((FIL_VTYPE *)&sv_coef[k * (nr_class - 1) + j]);
                Bv1 = *((FIL_VTYPE *)&sv_coef[k * (nr_class - 1) + j + 4]);

                #ifdef MIXED_VECTOR
                __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp) : "f"(Av), "f"(Bv0) :);
                __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp) : "f"(Av1), "f"(Bv1) :);
                #else
                    temp += Av * Bv0;
                    temp += Av1 * Bv1;
                #endif
            }

            for (int k = processed_elements; k < f_dim; k++)
            {
                #ifdef MIXED_VECTOR
                temp += (float)data_model[i * f_dim + k] * (float)sv_coef[k * (nr_class - 1) + j];
                #else
                temp[0] += data_model[i * f_dim + k] * sv_coef[k * (nr_class - 1) + j];
                #endif
            }
        #else
            // Manual unrolling
            for (int k = 0; k < processed_elements; k += 4)
            {
                Av = *((INP_VTYPE *)&data_model[i * f_dim + k]);
                Bv0 = *((FIL_VTYPE *)&sv_coef[k * (nr_class - 1) + j]);
                #ifdef MIXED_VECTOR
                __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp) : "f"(Av), "f"(Bv0) :);
                #else
                temp += Av * Bv0;
                #endif
            }

            for (int k = processed_elements; k < f_dim; k++)
            {
                #ifdef MIXED_VECTOR
                temp += (float)data_model[i * f_dim + k] * (float)sv_coef[k * (nr_class - 1) + j];
                #else
                temp[0] += data_model[i * f_dim + k] * sv_coef[k * (nr_class - 1) + j];
                #endif
            }

        #endif

        #ifdef MIXED_VECTOR
        if ((OUT_TYPE)temp + (OUT_TYPE)bias[0] >= 0)
        #else
        if (temp[0] + temp[1] + temp[2] + temp[3] + bias[0] >= 0)
        #endif
            Pred[i * (nr_class - 1) + j] = 1;
        else
            Pred[i * (nr_class - 1) + j] = 0;

        // }
    }
#if NUM_CORES > 1
    pi_cl_team_barrier();
#endif
}

#else // vectorized FP16 or FP16ALT Linear
void plp_SVM_linear(svm_model model, INP_TYPE *data_model, int *Pred, FIL_TYPE *sv_coef, FIL_TYPE *bias)
{

    #ifdef MIXED_VECTOR
    float temp = 0.0f;
    #else
    OUT_VTYPE temp;
    #endif
    INP_VTYPE Av;
    INP_VTYPE Av1;
    FIL_VTYPE Bv0;
    FIL_VTYPE Bv1;
    int j;
    int i = 0;
#ifndef FABRIC
    int core_id = pi_core_id();
#else
    int core_id = 0;
#endif

    int l = model.SVS;
    int f_dim = model.F_DIM;
    int nr_class = model.N_CLASS;

    int blockSize = (l + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * blockSize;
    int end = start + blockSize < l ? start + blockSize : l;

    int vec_limit = f_dim & ~0x3;
    int remainder_inloop = f_dim & 0x3;

    for (int i = start; i < end; i++)
    {
        // for (int j = 0; j < nr_class-1; j++) {
        int j = 0;
        #ifdef MIXED_VECTOR
        temp = 0.0f;
        #else
        temp = (OUT_VTYPE){0, 0};
        #endif
        // Manual unrolling

#ifdef FDIM_GT_32
        for (int k = 0; k < vec_limit; k += 4)
        {
            Av = *((INP_VTYPE *)&data_model[i * f_dim + k]);
            Av1 = *((INP_VTYPE *)&data_model[i * f_dim + k + 2]);

            Bv0 = *((FIL_VTYPE *)&sv_coef[k * (nr_class - 1) + j]);
            Bv1 = *((FIL_VTYPE *)&sv_coef[k * (nr_class - 1) + j + 2]);
            #ifdef MIXED_VECTOR
            #ifdef FP16
            __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp) : "f"(Av), "f"(Bv0) :);
            __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp) : "f"(Av1), "f"(Bv1) :);
            #else
            __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp) : "f"(Av), "f"(Bv0) :);
            __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp) : "f"(Av1), "f"(Bv1) :);
            #endif
            #else
            temp += Av * Bv0;
            temp += Av1 * Bv1;
            #endif
        }

        for (int k = f_dim - remainder_inloop; k < f_dim; k++)
        {
            #ifdef MIXED_VECTOR
            temp += (float)data_model[i * f_dim + k] * (float)sv_coef[k * (nr_class - 1) + j];
            #else
            temp[0] += data_model[i * f_dim + k] * sv_coef[k * (nr_class - 1) + j];
            #endif
        }
#else
        // Manual unrolling
        for (int k = 0; k < (f_dim & 0xfffffffe); k += 2)
        {

            Av = *((INP_VTYPE *)&data_model[i * f_dim + k]);
            Bv0 = *((FIL_VTYPE *)&sv_coef[k * (nr_class - 1) + j]);
            #ifdef MIXED_VECTOR
            #ifdef FP16
            __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp) : "f"(Av), "f"(Bv0) :);
            #else
            __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp) : "f"(Av), "f"(Bv0) :);
            #endif
            #else
            temp += Av * Bv0;
            #endif
        }
        if (f_dim & 0x00000001)
        {
            #ifdef MIXED_VECTOR
            temp += (float)data_model[i * f_dim + f_dim - 1] * (float)sv_coef[(f_dim - 1) * (nr_class - 1) + j];
            #else
            temp[0] += data_model[i * f_dim + f_dim - 1] * sv_coef[(f_dim - 1) * (nr_class - 1) + j];
            #endif
        }
#endif

        #ifdef MIXED_VECTOR
        if ((OUT_TYPE)temp + (OUT_TYPE)bias[0] >= 0)
        #else
        if (temp[0] + temp[1] + bias[0] >= 0)
        #endif
            Pred[i * (nr_class - 1) + j] = 1;
        else
            Pred[i * (nr_class - 1) + j] = 0;
    }
#if NUM_CORES > 1
    pi_cl_team_barrier();
#endif
}
#endif // End of Vectorization cases
#else // No vectorization
void __attribute__((noinline)) plp_SVM_linear(svm_model model, INP_TYPE *data_model, int *Pred, FIL_TYPE *sv_coef, FIL_TYPE *bias)
{
    #ifdef HWMIXED
    float sum = 0.0f;
    float temp = 0.0f;
    #else
    OUT_TYPE sum = 0;
    OUT_TYPE temp = 0;
    #endif

    int j;
    int i = 0;

    int l = model.SVS;
    int f_dim = model.F_DIM;
    int nr_class = model.N_CLASS;
#ifndef FABRIC
    int core_id = pi_core_id();
#else
    int core_id = 0;
#endif

    int blockSize = (l + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * blockSize;
    int end = start + blockSize < l ? start + blockSize : l;

    int vec_limit = f_dim & ~0x3;
    int remainder_inloop = f_dim & 0x3;

    for (int i = start; i < end; i++)
    {
        // for (int j = 0; j < nr_class-1; j++) {
        int j = 0;
        temp = 0;

#ifdef FDIM_GT_32
        // Manual unrolling
        for (int k = 0; k < vec_limit; k += 4)
        {

            #ifdef HWMIXED
            temp += (float)data_model[i * f_dim + k] * (float)sv_coef[k * (nr_class - 1) + j];
            temp += (float)data_model[i * f_dim + k + 1] * (float)sv_coef[k * (nr_class - 1) + j + (nr_class - 1)];
            temp += (float)data_model[i * f_dim + k + 2] * (float)sv_coef[k * (nr_class - 1) + j + (nr_class - 1) * 2];
            temp += (float)data_model[i * f_dim + k + 3] * (float)sv_coef[k * (nr_class - 1) + j + (nr_class - 1) * 3];
            #else
            temp += (OUT_TYPE)data_model[i * f_dim + k] * (OUT_TYPE)sv_coef[k * (nr_class - 1) + j];
            temp += (OUT_TYPE)data_model[i * f_dim + k + 1] * (OUT_TYPE)sv_coef[k * (nr_class - 1) + j + (nr_class - 1)];
            temp += (OUT_TYPE)data_model[i * f_dim + k + 2] * (OUT_TYPE)sv_coef[k * (nr_class - 1) + j + (nr_class - 1) * 2];
            temp += (OUT_TYPE)data_model[i * f_dim + k + 3] * (OUT_TYPE)sv_coef[k * (nr_class - 1) + j + (nr_class - 1) * 3];
            #endif
        }
        for (int k = f_dim - remainder_inloop; k < f_dim; k++)
        {
            #ifdef HWMIXED
            temp += (float)data_model[i * f_dim + k] * (float)sv_coef[k * (nr_class - 1) + j];
            #else   
            temp += (OUT_TYPE)data_model[i * f_dim + k] * (OUT_TYPE)sv_coef[k * (nr_class - 1) + j];
            #endif
        }
#else
        // Manual unrolling
        for (int k = 0; k < (f_dim & 0xfffffffe); k += 2)
        {
            #ifdef HWMIXED
            temp += (float)data_model[i * f_dim + k] * (float)sv_coef[k * (nr_class - 1) + j];
            temp += (float)data_model[i * f_dim + k + 1] * (float)sv_coef[k * (nr_class - 1) + j + (nr_class - 1)];
            #else
            temp += (OUT_TYPE)data_model[i * f_dim + k] * (OUT_TYPE)sv_coef[k * (nr_class - 1) + j];
            temp += (OUT_TYPE)data_model[i * f_dim + k + 1] * (OUT_TYPE)sv_coef[k * (nr_class - 1) + j + (nr_class - 1)];
            #endif
        }
        if (f_dim & 0x00000001)
        {
            #ifdef HWMIXED
            temp += (float)data_model[i * f_dim + f_dim - 1] * (float)sv_coef[(f_dim - 1) * (nr_class - 1) + j];
            #else
            temp += (OUT_TYPE)data_model[i * f_dim + f_dim - 1] * (OUT_TYPE)sv_coef[(f_dim - 1) * (nr_class - 1) + j];
            #endif
        }
#endif

        if ((OUT_TYPE)temp + (OUT_TYPE)bias[0] >= 0)
            Pred[i * (nr_class - 1) + j] = 1;
        else
            Pred[i * (nr_class - 1) + j] = 0;

        //}
    }
#if NUM_CORES > 1
    pi_cl_team_barrier();
#endif
}

#endif // end of non-vectorized cases

#else  // RBF kernel

// remove inline later
static __attribute__((noinline)) OUT_TYPE plp_rbf(const INP_TYPE *x, const INP_TYPE *y, FIL_TYPE gamma, int f_dim)
{
    #ifdef HWMIXED
    float sum = 0.0f;
    #else
    OUT_TYPE sum = 0.0f;
    #endif

    #ifdef VECTORIZATION
    #if BITS == 16 // FP16 or FP16alt VECTORIZED
        INP_VTYPE *v0 = (INP_VTYPE *)x;
        INP_VTYPE *v1 = (INP_VTYPE *)y;
        #ifdef MIXED_VECTOR
        INP_VTYPE r_1; // I assume that the first two operands are of the same type
        float r1 = 0.0f;
        float r2 = 0.0f;
        #else
        OUT_VTYPE r_1;
        OUT_VTYPE r_2;
        OUT_VTYPE r1 = {0};
        OUT_VTYPE r2 = {0};
        #endif
        int i = 0;


        // fp16 vectorization without unrolling -> process 2 elements

        int vec_limit = f_dim & 0xfffffffe;
        for (; i < vec_limit / 2; i += 1)
        {
            r_1 = ((*v0++) - (*v1++));

            #ifdef MIXED_VECTOR
            #ifdef FP16
            __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(r1) : "f"(r_1), "f"(r_1) :);
            #else
            __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(r1) : "f"(r_1), "f"(r_1) :);
            #endif
            #else
                r1 += (r_1) * (r_1);
            #endif
        }

        if (f_dim & 0x00000001)
        {
            #ifdef MIXED_VECTOR
            r1 += (float)(x[f_dim - 1] - y[f_dim - 1]) * (float)(x[f_dim - 1] - y[f_dim - 1]);
            #else
            r1[0] += (OUT_TYPE)(x[f_dim - 1] - y[f_dim - 1]) * (OUT_TYPE)(x[f_dim - 1] - y[f_dim - 1]);
            #endif
        }
        #ifdef MIXED_VECTOR
        sum = (OUT_TYPE)(r1);
        #else
        sum = (OUT_TYPE)(r1[0] += r1[1]);
        #endif

        // Uncomment it if you want to unroll fp16 vectorized -> process 4 elements
        
        /*int vec_limit = f_dim & ~0x1;   // number of full chunks
        int remainder_inloop = f_dim & 0x3; // number of remaining elements
        for (; i < vec_limit / 4; i += 1)
        {
            r_1 = ((*v0++) - (*v1++));
            r_2 = ((*v0++) - (*v1++));
            r1 += (r_1) * (r_1);
            r2 += (r_2) * (r_2);
        }

        for (; i < f_dim; i++)
        {
            r1[0] += (OUT_TYPE)(x[i] - y[i]) * (OUT_TYPE)(x[i] - y[i]);
        }

        r1[0] += r1[1];
        r2[0] += r2[1];
        sum = (OUT_TYPE)(r1[0] + r2[0]);
        */


    #else // BITS == 8 VECTORIZED

        INP_VTYPE *v0 = (INP_VTYPE *)x;
        INP_VTYPE *v1 = (INP_VTYPE *)y;
        #ifdef MIXED_VECTOR
        INP_VTYPE r_1; // I assume that the first two operands are of the same type
        float r1 = 0.0f;
        float r2 = 0.0f;
        #else
        OUT_VTYPE r_1;
        OUT_VTYPE r_2;
        OUT_VTYPE r1 = {0};
        OUT_VTYPE r2 = {0};
        #endif
        int i = 0;

        
        // FP8 vectorization without unrolling -> process 4 elements
        int vec_limit = f_dim & ~0x3;       // number of full chunks
        int remainder_inloop = f_dim & 0x3; // number of remaining elements

        for (; i < vec_limit / 4; i += 1)
        {
            r_1 = ((*v0++) - (*v1++));
            #ifdef MIXED_VECTOR
            __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(r1) : "f"(r_1), "f"(r_1) :);
            #else
            r1 += (r_1) * (r_1);
            #endif
        }
        for (i = f_dim - remainder_inloop; i < f_dim; i++)
        {
            
            #ifdef MIXED_VECTOR
            r1 += (float)(x[i] - y[i]) * (float)(x[i] - y[i]);
            #else
            r1[0] += (OUT_TYPE)(x[i] - y[i]) * (OUT_TYPE)(x[i] - y[i]);
            #endif
        }

        #ifdef MIXED_VECTOR
        sum = (OUT_TYPE)(r1);
        #else
        r1[0] = r1[0] + r1[1] + r1[2] + r1[3];
        // r1[0] += r1[1] + r1[2] + r1[3];
        sum = (OUT_TYPE)(r1[0]);
        #endif
        // Uncomment it if you want to unroll fp8 vectorized -> process 8 elements
        /*
        int vec_limit = f_dim & ~0x7;       // number of full chunks of 8 elements
        int remainder_inloop = f_dim & 0x7; // number of remaining elements

        for (; i < vec_limit; i += 8)
        {
            r_1 = ((*v0++) - (*v1++));
            r_2 = ((*v0++) - (*v1++));
            r1 += (r_1) * (r_1);
            r2 += (r_2) * (r_2);
        }
        for (i = f_dim - remainder_inloop; i < f_dim; i++)
        {
            r1[0] += (OUT_TYPE)(x[i] - y[i]) * (OUT_TYPE)(x[i] - y[i]);
        }
        r1[0] += r1[1] + r1[2] + r1[3];
        r2[0] += r2[1] + r2[2] + r2[3];
        sum = (OUT_TYPE)(r1[0] + r2[0]);
        */
        

    #endif // END OF BITS

    #else // Non-vectorized

        INP_TYPE a, b, c, e;
        #ifdef HWMIXED
        OUT_TYPE d, d1;
        #else
        OUT_TYPE d, d1;
        #endif

        for (int i = 0; i < (f_dim  & 0xfffffffe); i = i + 2)
        {

            a = x[i];
            b = y[i];
            c = x[i + 1];
            e = y[i + 1];
            asm volatile("" ::: "memory");
            d = (OUT_TYPE)a - (OUT_TYPE)b;
            d1 = (OUT_TYPE)c - (OUT_TYPE)e;
            #ifdef HWMIXED
            sum += (float)d * (float)d;
            sum += (float)d1 * (float)d1;
            #else
            sum += (OUT_TYPE)d * (OUT_TYPE)d;
            sum += (OUT_TYPE)d1 * (OUT_TYPE)d1;
            #endif
        }
        if (f_dim & 0x00000001)
        {
            a = x[f_dim - 1];
            b = y[f_dim - 1];
            d = (OUT_TYPE)a - (OUT_TYPE)b;
            #ifdef HWMIXED
            sum += (float)d * (float)d;
            #else
            sum += (OUT_TYPE)d * (OUT_TYPE)d;
            #endif
        }

#endif

#if BITS == 32
    OUT_TYPE gs = (OUT_TYPE)-gamma * (OUT_TYPE)sum;
    OUT_TYPE result = fastexp_gist(gs);
    return result;
#elif BITS == 16
    OUT_TYPE gs = (OUT_TYPE)-gamma * (OUT_TYPE)sum;
    OUT_TYPE result = fastexp_gist(gs);
    return result;
#elif BITS == 8
    OUT_TYPE gs = (OUT_TYPE)-gamma * (OUT_TYPE)sum;
    OUT_TYPE result = fastexp_gist(gs);
    return result;

    // return fastexp_gist_f8(-gamma * sum);
    // return fastexp_gist_f8((float8)(-gamma * sum));
    // return fast_exp(-gamma * sum);
    // return exp16(-gamma * sum);

#endif
}

/**
  @addtogroup plp_SVM_predict
  @{
 */

/**
  @brief Code SVM prediction 32-bit floating-point.
  @param[in]  model  structur that contains all the SVM parameters
  @param[in]  x      points to the input matrix
  @param[in]  data_model      points to SVs of the model
  @param[in]  sv_coef  points to the coefficients of the model
  @return     the predicted class
*/

void __attribute__((noinline)) plp_SVM_RBF(svm_model model, INP_TYPE *data_model, FIL_TYPE *sv_coef, FIL_TYPE *bias, INP_TYPE *x_ref, int *Pred)
{
#ifndef FABRIC
    int core_id = pi_core_id();
#else
    int core_id = 0;
#endif

    int j;
    int i = 0;
    FIL_TYPE gamma1 = model.GAMMA1;
    int l = model.SVS;
    int coef_dim = model.COEF_DIM;
    int f_dim = model.F_DIM;
    FIL_TYPE SV[f_dim + 1];
    INP_TYPE x1[f_dim + 1];
    INP_TYPE *ptrx, *ptrs;
#if NUM_CORES > 1

    int blockSize = (l + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * blockSize;
    int end = start + blockSize;
    if (end > l)
        end = l;

    for (int i = start; i < end; i++)
    {
#else

    for (int i = 0; i < l; i++)
    {
#endif
        #ifdef HWMIXED
        OUT_TYPE temp;
        OUT_TYPE inter;
        #else
        OUT_TYPE temp, inter;
        #endif
        inter = 0;
        ptrx = &data_model[i * f_dim];
        for (int k = 0; k < coef_dim; k++)
        {
            ptrs = &x_ref[k * f_dim];
            temp = plp_rbf(ptrx, ptrs, gamma1, f_dim);
            #ifdef HWMIXED
            temp = (OUT_TYPE)temp * (OUT_TYPE)sv_coef[k];
            inter = inter + temp;
            #else
            temp = (OUT_TYPE)temp * (OUT_TYPE)sv_coef[k];
            inter = inter + temp;
            #endif
        }
        if ((OUT_TYPE)inter + (OUT_TYPE)bias[0] >= 0)
            Pred[i] = 1;
        else
            Pred[i] = 0;
    }

#if NUM_CORES > 1
    pi_cl_team_barrier();
#endif
}
#endif // End of RBF kernel

/**
  @} end of plp_SVM_predict group
 */