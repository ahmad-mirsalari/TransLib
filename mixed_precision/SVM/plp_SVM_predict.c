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
#include "defines.h"
#include "plp_exp.h"
#define FSIGN(x)    (*((int *)(&x)) >> 31)
#define V2FSIGN(x)  __builtin_pulp_sra2(*((v4s *)(&x)), (v2s) {15, 15})



static inline v2f16 pack_f16(float16 a, float16 b) {
  v2f16 result;
  __asm__ __volatile__ ("pv.pack.h %0, %1, %2" : "=f" (result): "f" (a), "f" (b) : );
  return result;
}

#ifdef VECTORIZATION
void plp_SVM_linear(svm_model model, INP_TYPE *data_model, int *Pred, FIL_TYPE *sv_coef, FIL_TYPE *bias)
{
        
        OUT_VTYPE temp;
        INP_VTYPE Av;
        INP_VTYPE Av1;
        FIL_VTYPE Bv0;
        FIL_VTYPE Bv1;
        int j;
        int i = 0;

        int l = model.SVS;
        int f_dim = model.F_DIM;
        int nr_class = model.N_CLASS;

        int blockSize = (l+NUM_CORES-1)/NUM_CORES;
        int start = pi_core_id()*blockSize;
        int end = start + blockSize < l? start + blockSize : l;

        for (int i = start; i < end; i++) {
            for (int j = 0; j < nr_class-1; j++) {
            temp = (OUT_VTYPE) {0, 0};

              //Manual unrolling
              for (int k = 0; k<(f_dim & 0xfffffffe); k+=2) {
                Av  = *((INP_VTYPE *) &data_model[i*f_dim+k]);

                //Av1  = *((INP_VTYPE *) &data_model[i*f_dim+k+2]);
                Bv0 = *((FIL_VTYPE *) &sv_coef[k*(nr_class-1)+j]);
                //Bv1 = *((FIL_VTYPE *) &sv_coef[k*(nr_class-1)+j+2]);
                temp +=  Av * Bv0;
                //printf("k is %d Av0 %f, Av1 %f , Bv0 %f Bv1 %f temp0 %f temp %f \n", k,Av[0], Av[1], Bv0[0], Bv0[1], temp[0], temp[1]);
                //temp += Av1 * Bv1;
                //temp += data_model[i*f_dim+k]   * sv_coef[k*(nr_class-1)+j];
                
                }

            //temp = temp + (v2f16) {bias[0], bias[0]};
            if (f_dim & 0x00000001) 
             {
              temp[0] += data_model[i*f_dim+f_dim-1] * sv_coef[(f_dim-1)*(nr_class-1)+j];
             }

            if (temp[0] + temp[1] + bias[0] >= 0)
                Pred[i*(nr_class-1)+j] = 1 ;
            else
                Pred[i*(nr_class-1)+j] = 0 ;
            
        }
    }
    #if NUM_CORES > 1
        pi_cl_team_barrier();
    #endif 
}

#else
void plp_SVM_linear(svm_model model, INP_TYPE *data_model, int *Pred, FIL_TYPE *sv_coef, FIL_TYPE *bias)
{
        OUT_TYPE sum = 0;
        int j;
        int i = 0;

        int l = model.SVS;
        int f_dim = model.F_DIM;
        int nr_class = model.N_CLASS;

        int blockSize = (l+NUM_CORES-1)/NUM_CORES;
        int start = pi_core_id()*blockSize;
        int end = start + blockSize < l? start + blockSize : l;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < nr_class-1; j++) {
              OUT_TYPE temp = 0;

              //Manual unrolling
              for (int k = 0; k< (f_dim & 0xfffffffe); k+=2) {
                temp += data_model[i*f_dim+k]   * sv_coef[k*(nr_class-1)+j];
                temp += data_model[i*f_dim+k+1]   * sv_coef[k*(nr_class-1)+j+(nr_class-1)];
                //printf("  x is %f , w is  %f, ab is %f ,temp %f\n", data_model[i*f_dim+k] ,sv_coef[k*(nr_class-1)+j],data_model[i*f_dim+k]   * sv_coef[k*(nr_class-1)+j], temp);
                }
                if (f_dim & 0x00000001) 
                     {
                      temp += data_model[i*f_dim+f_dim-1] * sv_coef[(f_dim-1)*(nr_class-1)+j];
                     }
            if (temp + bias[0] >= 0)
                Pred[i*(nr_class-1)+j] = 1 ;
            else
                Pred[i*(nr_class-1)+j] = 0 ;
            
        }
    }
        #if NUM_CORES > 1
        pi_cl_team_barrier();
        #endif 
}

#endif
static OUT_TYPE plp_rbf(const INP_TYPE *x, const INP_TYPE *y, FIL_TYPE gamma, int f_dim){

        OUT_TYPE sum = 0.0f;

#ifdef VECTORIZATION
        
        INP_VTYPE *v0 = (INP_VTYPE *)x;  
        INP_VTYPE *v1 = (INP_VTYPE *)y;
        OUT_VTYPE r_1;
        OUT_VTYPE r_2;
        OUT_VTYPE r1 = {0};
        OUT_VTYPE r2 = {0}; 

        for(int i = 0; i < f_dim/2; i+=2){ 
          r_1 = ((*v0++) - (*v1++));
          r_2 = ((*v0++) - (*v1++));   
          r1 += (r_1) * (r_1);  
          r2 += (r_2) * (r_2);  
        }  
        r1[0] += r1[1];
        r2[0] += r2[1];
        sum = (OUT_TYPE)(r1[0] + r2[0]);        
#else  
        OUT_TYPE d, d1;
        INP_TYPE a, b, c, e; 

        for(int i = 0; i < f_dim; i=i+2){
                
                a = x[i]; 
                b = y[i];
                c = x[i+1]; 
                e = y[i+1];
                asm volatile ("":::"memory");
                d = a - b;
                d1 = c - e;
                sum += d * d;
                sum += d1 * d1;
        }
    
#endif 
#if BITS == 32
        return exp32(-gamma*sum);
#else
        return exp16(-gamma*sum);
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

 
void  plp_SVM_RBF(svm_model model, INP_TYPE *data_model,FIL_TYPE *sv_coef, FIL_TYPE *bias, INP_TYPE *x_ref,int *Pred)
{
 
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

                int blockSize = (l+NUM_CORES-1) / NUM_CORES;
                int start = pi_core_id()*blockSize;
                int end = start+blockSize;
                if(end > l) end = l;

                for (int i=start; i < end; i++) {
#else
                
                for(int i=0; i < l; i++){
#endif
                    OUT_TYPE temp,inter;
                    inter =0;
                    ptrx = &data_model[i*f_dim];
                     for (int k=0; k < coef_dim; k++)
                     {
                        ptrs = &x_ref[k*f_dim];
                        temp = plp_rbf(ptrx, ptrs, gamma1, f_dim);
                        temp = temp * sv_coef[k];
                        inter = inter + temp ;
                     }
                     if (inter + bias[0] >= 0)
                            Pred[i] = 1 ;
                        else
                            Pred[i] = 0 ;        
        }
        
#if NUM_CORES > 1
        pi_cl_team_barrier();
#endif
}


/**
  @} end of plp_SVM_predict group
 */
