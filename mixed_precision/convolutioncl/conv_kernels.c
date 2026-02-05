// Copyright 2017 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pmsis.h"
#include "convolution.h"
#include "config.h"

#if (FILT_WIN == 3) // 3x3 filter non-vectorized
void __attribute__((noinline)) Conv_Scalar(INP_TYPE *In_Img, OUT_TYPE *Out_Img, FIL_TYPE *Kernel, int ROW, int COL, int Stride, int Input_cl, int FILT_W)
{

  #ifndef FABRIC
    int core_id = pi_core_id();
  #else
    int core_id = 0;
  #endif

  #ifdef HWMIXED
  float acc=0;
  #else
  OUT_TYPE acc;
  #endif

  int k, w, t;
  
  int blockSize = ((ROW) + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;

  int end = start + blockSize < ROW ? start + blockSize : ROW;
  for (int row = start; row < end; row++)
  {
    for (int col = 0; col < COL; col++)
    {

      acc = 0;
      t = row * COL + col;

      for (int i = 0; i < FILT_W; i++)
      {
        FIL_TYPE coeff0, coeff1, coeff2;
        INP_TYPE data0, data1, data2;
        k = (row * Stride + i) * Input_cl + (col * Stride); // coeff for one dimension matrix
        data0 = In_Img[k++];
        data1 = In_Img[k++];
        data2 = In_Img[k];
        w = (i)*FILT_W;
        coeff0 = Kernel[w++];
        coeff1 = Kernel[w++];
        coeff2 = Kernel[w];

        #ifdef HWMIXED
        asm volatile("" : : : "memory");
        acc += (float)coeff0 * (float)data0;
        acc += (float)coeff1 * (float)data1;
        acc += (float)coeff2 * (float)data2;
        #else
        asm volatile("" : : : "memory");
        acc += (OUT_TYPE)coeff0 * (OUT_TYPE)data0;
        acc += (OUT_TYPE)coeff1 * (OUT_TYPE)data1;
        acc += (OUT_TYPE)coeff2 * (OUT_TYPE)data2;
        #endif
      }
      Out_Img[t] = (OUT_TYPE)acc;
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}

#elif (FILT_WIN == 5) // 5x5 filter non-vectorized
void __attribute__((noinline)) Conv_Scalar(INP_TYPE *In_Img, OUT_TYPE *Out_Img, FIL_TYPE *Kernel, int ROW, int COL, int Stride, int Input_cl, int FILT_W)
{
  #ifndef FABRIC
    int core_id = pi_core_id();
  #else
    int core_id = 0;
  #endif
  
  #ifdef HWMIXED
  float acc=0;
  #else
  OUT_TYPE acc;
  #endif

  int k, w, t;
  int blockSize = ((ROW) + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < ROW ? start + blockSize : ROW;
  for (int row = start; row < end; row++)
  {
    for (int col = 0; col < COL; col++)
    {

      acc = 0;
      t = row * COL + col;
      for (int i = 0; i < FILT_W; i++)
      {
        FIL_TYPE coeff0, coeff1, coeff2, coeff3, coeff4;
        INP_TYPE data0, data1, data2, data3, data4;
        k = (row * Stride + i) * Input_cl + (col * Stride); // coeff for one dimension matrix
        data0 = In_Img[k++];
        data1 = In_Img[k++];
        data2 = In_Img[k++];
        data3 = In_Img[k++];
        data4 = In_Img[k];
        w = (i)*FILT_W;
        coeff0 = Kernel[w++];
        coeff1 = Kernel[w++];
        coeff2 = Kernel[w++];
        coeff3 = Kernel[w++];
        coeff4 = Kernel[w];
        #ifdef HWMIXED
        asm volatile("" : : : "memory");
        acc += (float)coeff0 * (float)data0;
        acc += (float)coeff1 * (float)data1;
        acc += (float)coeff2 * (float)data2;
        acc += (float)coeff3 * (float)data3;
        acc += (float)coeff4 * (float)data4;
        #else
        asm volatile("" : : : "memory");
        acc += (OUT_TYPE)coeff0 * (OUT_TYPE)data0;
        acc += (OUT_TYPE)coeff1 * (OUT_TYPE)data1;
        acc += (OUT_TYPE)coeff2 * (OUT_TYPE)data2;
        acc += (OUT_TYPE)coeff3 * (OUT_TYPE)data3;
        acc += (OUT_TYPE)coeff4 * (OUT_TYPE)data4;
        #endif
      }
      Out_Img[t] = (OUT_TYPE)acc;
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}

#else // Generic filter non-vectorized
void __attribute__((noinline)) Conv_Scalar(INP_TYPE *In_Img, OUT_TYPE *Out_Img, FIL_TYPE *Kernel, int ROW, int COL, int Stride, int Input_cl, int FILT_W)
{
  #ifndef FABRIC
    int core_id = pi_core_id();
  #else
    int core_id = 0;
  #endif
  #ifdef HWMIXED
  float acc=0;
  #else
  OUT_TYPE acc;
  #endif

  int k, w, t;
  int blockSize = ((ROW) + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;

  int end = start + blockSize < ROW ? start + blockSize : ROW;
  for (int row = start; row < end; row++)
  {
    for (int col = 0; col < COL; col++)
    {

      acc = 0;
      t = row * COL + col;

      for (int i = 0; i < FILT_W; i++)
      {
        for (int j = 0; j < FILT_W; j++)
        {
          register FIL_TYPE coeff;                                // asm("t1");
          register INP_TYPE data;                                 // asm("t2");
          k = (row * Stride + i) * Input_cl + (col * Stride + j); // coeff for one dimension matrix
          data = In_Img[k];
          w = (i)*FILT_W + (j);
          coeff = Kernel[w];

          #ifdef HWMIXED
          acc += (float)coeff * (float)data;
          #else
          acc += (OUT_TYPE)coeff * (OUT_TYPE)data;
          #endif
        }
      }
      Out_Img[t] = (OUT_TYPE)acc;
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}
#endif // Non vectorized endif

#ifndef FP8 

#if (FILT_WIN == 5) // 5x5 FP16 or FP16alt vectorization __attribute__((always_inline))
void __attribute__((always_inline)) Conv5x5_Vector(INP_TYPE *In_Img, OUT_TYPE *Out_Img, int ROW, int COL, int Input_cl, FIL_TYPE *Kernel)
{
#ifndef FABRIC
  int core_id = pi_core_id();
#else
  int core_id = 0;
#endif

#ifdef VECTORIAL
  FIL_VTYPE coeff_0, coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6, coeff_7, coeff_8, coeff_9, coeff_10, coeff_11, coeff_12;
  INP_VTYPE Img_0, Img_1, Img_2, Img_3, Img_4, Img_5, Img_6, Img_7, Img_8, Img_9, Img_10, Img_11, Img_12;
  INP_VTYPE new_data1, new_data2, new_data3;
  int row, col, t;
  v2s mask0;

  #ifdef MIXED_VECTOR
    float acc = 0;
  #else
    OUT_TYPE acc;
    OUT_VTYPE temp;
  #endif

  coeff_0 = *((FIL_VTYPE *)(&Kernel[0]));
  coeff_1 = *((FIL_VTYPE *)(&Kernel[2]));
  coeff_2 = *((FIL_VTYPE *)(&Kernel[5]));
  coeff_3 = *((FIL_VTYPE *)(&Kernel[7]));
  coeff_4 = *((FIL_VTYPE *)(&Kernel[10]));
  coeff_5 = *((FIL_VTYPE *)(&Kernel[12]));
  coeff_6 = *((FIL_VTYPE *)(&Kernel[15]));
  coeff_7 = *((FIL_VTYPE *)(&Kernel[17]));
  coeff_8 = *((FIL_VTYPE *)(&Kernel[20]));
  coeff_9 = *((FIL_VTYPE *)(&Kernel[22]));
  coeff_10[0] = Kernel[4];
  coeff_10[1] = Kernel[9];
  coeff_11[0] = Kernel[14];
  coeff_11[1] = Kernel[19];
  coeff_12[0] = Kernel[24];
  coeff_12[1] = 0.0f;

  mask0 = (v2s){1, 2};

  int offset = 0;

  // image board is black
#ifdef TILING2D
  int xBlockSize = (ROW) / 2;
  int yBlockSize = ((COL) + (NUM_CORES / 2) - 1) / (NUM_CORES / 2);
  int x = (core_id / (NUM_CORES / 2)) * xBlockSize;
  int y = (core_id % (NUM_CORES / 2)) * yBlockSize;
  offset = x * Input_cl;

  for (col = y; (y < COL) && (col < y + yBlockSize); col++)
  {

#else

  int blockSize = ((COL) + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < COL ? start + blockSize : COL;
  for (col = start; col < end; col++)
  {
#endif

    Img_0 = *((INP_VTYPE *)(&In_Img[col + offset]));
    Img_1 = *((INP_VTYPE *)(&In_Img[col + 2 + offset]));
    Img_2 = *((INP_VTYPE *)(&In_Img[col + Input_cl + offset]));
    Img_3 = *((INP_VTYPE *)(&In_Img[col + 2 + Input_cl + offset]));
    Img_4 = *((INP_VTYPE *)(&In_Img[col + 2 * Input_cl + offset]));
    Img_5 = *((INP_VTYPE *)(&In_Img[col + 2 + 2 * Input_cl + offset]));
    Img_6 = *((INP_VTYPE *)(&In_Img[col + 3 * Input_cl + offset]));
    Img_7 = *((INP_VTYPE *)(&In_Img[col + 2 + 3 * Input_cl + offset]));
    Img_8 = *((INP_VTYPE *)(&In_Img[col + 4 * Input_cl + offset]));
    Img_9 = *((INP_VTYPE *)(&In_Img[col + 2 + 4 * Input_cl + offset]));
    Img_10[0] = In_Img[col + 4 + offset];
    Img_10[1] = In_Img[col + 4 + Input_cl + offset];
    Img_11[0] = In_Img[col + 4 + 2 * Input_cl + offset];
    Img_11[1] = In_Img[col + 4 + 3 * Input_cl + offset];
    Img_12[0] = In_Img[col + 4 + 4 * Input_cl + offset];
    Img_12[1] = 0;

#ifdef TILING2D
    for (row = x; (row < ROW) && (row < x + xBlockSize); row++)
    {
#else
    for (row = 0; row < ROW; row++)
    {
#endif
      t = (row)*ROW + col;

      #ifdef MIXED_VECTOR
      acc = 0;
      #else
      temp = (OUT_VTYPE){0, 0};
      #endif

      #ifdef MIXED_VECTOR

        #ifdef FP16
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_0), "f"(coeff_0) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_1), "f"(coeff_1) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_2), "f"(coeff_2) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_3), "f"(coeff_3) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_4), "f"(coeff_4) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_5), "f"(coeff_5) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_6), "f"(coeff_6) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_7), "f"(coeff_7) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_8), "f"(coeff_8) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_9), "f"(coeff_9) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_10), "f"(coeff_10) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_11), "f"(coeff_11) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_12), "f"(coeff_12) :);

      #else //FP16alt
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_0), "f"(coeff_0) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_1), "f"(coeff_1) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_2), "f"(coeff_2) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_3), "f"(coeff_3) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_4), "f"(coeff_4) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_5), "f"(coeff_5) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_6), "f"(coeff_6) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_7), "f"(coeff_7) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_8), "f"(coeff_8) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_9), "f"(coeff_9) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_10), "f"(coeff_10) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_11), "f"(coeff_11) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_12), "f"(coeff_12) :);
      #endif

      #else
      // asm volatile("":::"memory");
        temp += Img_0 * coeff_0;
        temp += Img_1 * coeff_1;
        temp += Img_2 * coeff_2;
        temp += Img_3 * coeff_3;
        temp += Img_4 * coeff_4;
        temp += Img_5 * coeff_5;
        temp += Img_6 * coeff_6;
        temp += Img_7 * coeff_7;
        temp += Img_8 * coeff_8;
        temp += Img_9 * coeff_9;
        temp += Img_10 * coeff_10;
        temp += Img_11 * coeff_11;
        temp += Img_12 * coeff_12;
        acc = temp[0] + temp[1];
      #endif

      Out_Img[t] = (OUT_TYPE)acc;

      new_data1 = *((INP_VTYPE *)(&In_Img[col + (row + 5) * Input_cl]));
      new_data2 = *((INP_VTYPE *)(&In_Img[col + 2 + (row + 5) * Input_cl]));
      new_data3[0] = In_Img[col + 4 + (row + 5) * Input_cl];
      new_data3[1] = 0;

      // Move the window
      /*
        thirteen vectors:

        Img_0  = {A0, A1}
        Img_1  = {B0, B1}
        Img_2  = {C0, C1}
        Img_3  = {D0, D1}
        Img_4  = {E0, E1}
        Img_5  = {F0, F1}
        Img_6  = {G0, G1}
        Img_7  = {H0, H1}
        Img_8  = {I0, I1}
        Img_9  = {J0, J1}
        Img_10 = {K0, K1}
        Img_11 = {L0, L1}
        Img_12 = {M0,  0}

        Current Windonw:
        XX XX XX XX XX
        A0 A1 B0 B1 K0
        C0 C1 D0 D1 K1
        E0 E1 F0 F1 L0
        G0 G1 H0 H1 L1
        I0 I1 J0 J1 M0
        N0 N1 P0 P1 M1
        XX XX XX XX XX

        We want to load next line (N0, N1, P0, P1, M1)
        in vectors new_data1 and new_data2
        new_data1 = {N0, N1}
        new_data2 = {P0, P1}
        new_data3 = {M1,  0}

        Move each vector one line down and shuffle the vertical vector

        Img_0  = Img_2
        Img_1  = Img_3
        Img_2  = Img_4
        Img_3  = Img_5
        Img_4  = Img_6
        Img_5  = Img_7
        Img_6  = Img_8
        Img_7  = Img_9
        Img_8  = new_data1
        Img_9  = new_data2
        Img_10 = {K1, L0}
        Img_11 = {L1, M0}
        Img_12 = new_data3
      */

      Img_0 = Img_2;
      Img_1 = Img_3;
      Img_2 = Img_4;
      Img_3 = Img_5;
      Img_4 = Img_6;
      Img_5 = Img_7;
      Img_6 = Img_8;
      Img_7 = Img_9;
      Img_8 = new_data1;
      Img_9 = new_data2;
      Img_10 = (INP_VTYPE)__builtin_shuffle(Img_10, Img_11, mask0);
      Img_11 = (INP_VTYPE)__builtin_shuffle(Img_11, Img_12, mask0);
      Img_12 = new_data3;
    }
#ifdef TILING2D
    // last iteration
    t = (row + 1) * ROW + col + 2;

    OUT_VTYPE temp;
    temp = Img_0 * coeff_0;
    temp += Img_1 * coeff_1;
    temp += Img_2 * coeff_2;
    temp += Img_3 * coeff_3;
    temp += Img_4 * coeff_4;
    temp += Img_5 * coeff_5;
    temp += Img_6 * coeff_6;
    temp += Img_7 * coeff_7;
    temp += Img_8 * coeff_8;
    temp += Img_9 * coeff_9;
    temp += Img_10 * coeff_10;
    temp += Img_11 * coeff_11;
    temp += Img_12 * coeff_12;
    acc = temp[0] + temp[1];

    Out_Img[t] = (OUT_TYPE)acc;
#endif
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif

#endif
}

#elif (FILT_WIN == 3)
// FP16 or FP16alt 3x3 Vectorization
void __attribute__((always_inline)) Conv3x3_Vector(INP_TYPE *In_Img, OUT_TYPE *Out_Img, int ROW, int COL, int Input_cl, FIL_TYPE *Kernel)
{
  #ifndef FABRIC
    int core_id = pi_core_id();
  #else
    int core_id = 0;
  #endif
  #ifdef VECTORIAL
  FIL_VTYPE coeff_0, coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6;
  INP_VTYPE Img_0, Img_1, Img_2, Img_3, Img_4, Img_5;
  INP_VTYPE new_data1, new_data2;
  int row, col, t;
  v2s mask0;

  #ifdef MIXED_VECTOR
  float acc = 0.0f;
  #else
  OUT_TYPE acc;
  OUT_VTYPE temp;
  #endif

  coeff_0 = *((FIL_VTYPE *)(&Kernel[0]));
  coeff_6 = *((FIL_VTYPE *)(&Kernel[8]));
  coeff_1 = *((FIL_VTYPE *)(&Kernel[3]));
  coeff_2 = *((FIL_VTYPE *)(&Kernel[6]));
  coeff_3[0] = Kernel[2];
  coeff_3[1] = Kernel[5];
  coeff_4[0] = Kernel[8];
  coeff_4[1] = 0.0f;

  mask0 = (v2s){1, 2};

  int offset = 0;

  // image board is black
#ifdef TILING2D
  int xBlockSize = (ROW) / 2;
  int yBlockSize = ((COL) + (NUM_CORES / 2) - 1) / (NUM_CORES / 2);
  int x = (core_id / (NUM_CORES / 2)) * xBlockSize;
  int y = (core_id % (NUM_CORES / 2)) * yBlockSize;
  offset = x * Input_cl;

  for (col = y; (y < COL) && (col < y + yBlockSize); col++)
  {

#else

  int blockSize = ((COL) + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < COL ? start + blockSize : COL;
  for (col = start; col < end; col++)
  {
#endif

    Img_0 = *((INP_VTYPE *)(&In_Img[col + offset]));
    Img_1 = *((INP_VTYPE *)(&In_Img[col + Input_cl + offset]));
    Img_2 = *((INP_VTYPE *)(&In_Img[col + 2 * Input_cl + offset]));
    Img_3[0] = In_Img[col + 2 + offset];
    Img_3[1] = In_Img[col + 2 + Input_cl + offset];
    Img_4[0] = In_Img[col + 2 + 2 * Input_cl + offset];
    Img_4[1] = 0;

#ifdef TILING2D
    for (row = x + 1; (row < ROW) && (row < x + xBlockSize); row++)
    {
#else
    for (row = 0; row < ROW; row++)
    {
#endif
      t = (row)*ROW + col;

      #ifdef MIXED_VECTOR
      acc = 0;
      #else
      temp = (OUT_VTYPE){0, 0};
      #endif


      #ifdef MIXED_VECTOR
        #ifdef FP16
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_0), "f"(coeff_0) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_1), "f"(coeff_1) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_2), "f"(coeff_2) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_3), "f"(coeff_3) :);
      __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(acc) : "f"(Img_4), "f"(coeff_4) :);

      #else //FP16alt
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_0), "f"(coeff_0) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_1), "f"(coeff_1) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_2), "f"(coeff_2) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_3), "f"(coeff_3) :);
      __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(acc) : "f"(Img_4), "f"(coeff_4) :);
      #endif
      #else
      temp += Img_0 * coeff_0;
      temp += Img_1 * coeff_1;
      temp += Img_2 * coeff_2;
      temp += Img_3 * coeff_3;
      temp += Img_4 * coeff_4;
      acc = temp[0] + temp[1];
      #endif

      Out_Img[t] = (OUT_TYPE)acc;

      new_data1 = *((INP_VTYPE *)(&In_Img[col + (row + 3) * Input_cl]));
      new_data2[0] = In_Img[col + 2 + (row + 3) * Input_cl];
      new_data2[1] = 0;
      Img_0 = Img_1;
      Img_1 = Img_2;
      Img_2 = new_data1;
      Img_3 = (INP_VTYPE)__builtin_shuffle(Img_3, Img_4, mask0);
      Img_4 = new_data2;
    }
#ifdef TILING2D
    // last iteration
    t = (row + 1) * ROW + col + 2;

    OUT_VTYPE temp;
    temp = Img_0 * coeff_0;
    temp += Img_1 * coeff_1;
    temp += Img_2 * coeff_2;
    temp += Img_3 * coeff_3;
    temp += Img_4 * coeff_4;
    acc = temp[0] + temp[1];

    Out_Img[t] = (OUT_TYPE)acc;
#endif
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif

#endif
}
#endif // FP16 and FP16ALT
#else // FP8

#if (FILT_WIN == 3)
// FP8 3x3 vectorization
void __attribute__((always_inline)) Conv3x3_Vector(INP_TYPE *In_Img, OUT_TYPE *Out_Img, int ROW, int COL, int Input_cl, FIL_TYPE *Kernel)
{
#ifndef FABRIC
  int core_id = pi_core_id();
#else
  int core_id = 0;
#endif
#ifdef VECTORIAL
  FIL_VTYPE coeff_0, coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6;
  INP_VTYPE Img_0, Img_1, Img_2, Img_3, Img_4, Img_5;
  INP_VTYPE new_data1, new_data2;
  int row, col, t;
  v2s mask0;

  #ifdef MIXED_VECTOR
  float acc = 0;
  #else
  OUT_TYPE acc;
  OUT_VTYPE temp;
  #endif

  coeff_0 = *((FIL_VTYPE *)(&Kernel[0]));
  coeff_0[3] = 0;
  coeff_1 = *((FIL_VTYPE *)(&Kernel[3]));
  coeff_1[3] = 0;
  coeff_2 = *((FIL_VTYPE *)(&Kernel[6]));
  coeff_2[3] = 0;

  mask0 = (v2s){1, 2};

  int offset = 0;

  // image board is black
#ifdef TILING2D
  int xBlockSize = (ROW) / 2;
  int yBlockSize = ((COL) + (NUM_CORES / 2) - 1) / (NUM_CORES / 2);
  int x = (core_id / (NUM_CORES / 2)) * xBlockSize;
  int y = (core_id % (NUM_CORES / 2)) * yBlockSize;
  offset = x * Input_cl;

  for (col = y; (y < COL) && (col < y + yBlockSize); col++)
  {

#else

  int blockSize = ((COL) + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < COL ? start + blockSize : COL;
  for (col = start; col < end; col++)
  {
#endif

    Img_0 = *((INP_VTYPE *)(&In_Img[col + offset]));
    Img_1 = *((INP_VTYPE *)(&In_Img[col + Input_cl + offset]));
    Img_2 = *((INP_VTYPE *)(&In_Img[col + 2 * Input_cl + offset]));

#ifdef TILING2D
    for (row = x + 1; (row < ROW) && (row < x + xBlockSize); row++)
    {
#else
    for (row = 0; row < ROW; row++)
    {
#endif
      t = (row)*ROW + col;

      #ifdef MIXED_VECTOR
      acc = 0;
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_0), "f"(coeff_0) :);
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_1), "f"(coeff_1) :);
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_2), "f"(coeff_2) :);
      #else
      temp = (OUT_VTYPE){0, 0, 0, 0};
      temp += Img_0 * coeff_0;

      temp += Img_1 * coeff_1;

      temp += Img_2 * coeff_2;
      acc = temp[0] + temp[1] + temp[2];
      #endif
      Out_Img[t] = (OUT_TYPE)acc;

      new_data1 = *((INP_VTYPE *)(&In_Img[col + (row + 3) * Input_cl]));
      Img_0 = Img_1;
      Img_1 = Img_2;
      Img_2 = new_data1;
    }
#ifdef TILING2D
    // last iteration
    t = (row + 1) * ROW + col + 2;

    OUT_VTYPE temp;
    temp = Img_0 * coeff_0;
    temp += Img_1 * coeff_1;
    temp += Img_2 * coeff_2;
    // temp += Img_3 * coeff_3;
    // temp += Img_4 * coeff_4;
    acc = temp[0] + temp[1];

    Out_Img[t] = (OUT_TYPE)acc;
#endif
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif

#endif
}
#elif (FILT_WIN == 5)
// FP8 5x5 vectorization
void __attribute__((always_inline)) Conv5x5_Vector(INP_TYPE *In_Img, OUT_TYPE *Out_Img, int ROW, int COL, int Input_cl, FIL_TYPE *Kernel)
{
#ifndef FABRIC
  int core_id = pi_core_id();
#else
  int core_id = 0;
#endif
#ifdef VECTORIAL
  FIL_VTYPE coeff_0, coeff_1, coeff_2, coeff_3, coeff_4, coeff_5; //, coeff_6, coeff_7, coeff_8, coeff_9, coeff_10, coeff_11, coeff_12;
  FIL_TYPE coeff_6;
  INP_VTYPE Img_0, Img_1, Img_2, Img_3, Img_4, Img_5; //, Img_6, Img_7, Img_8, Img_9, Img_10, Img_11, Img_12;
  INP_TYPE Img_6;
  INP_VTYPE new_data1; //, new_data2, new_data3;
  INP_TYPE new_data2, sh_img;
  int row, col, t;
  v4s mask0;
  #ifdef MIXED_VECTOR
  float acc = 0;
  #else
  OUT_VTYPE temp;
  OUT_TYPE acc;
  #endif

  coeff_0 = *((FIL_VTYPE *)(&Kernel[0]));
  coeff_1 = *((FIL_VTYPE *)(&Kernel[5]));
  coeff_2 = *((FIL_VTYPE *)(&Kernel[10]));
  coeff_3 = *((FIL_VTYPE *)(&Kernel[15]));
  coeff_4 = *((FIL_VTYPE *)(&Kernel[20]));
  coeff_5[0] = Kernel[4];
  coeff_5[1] = Kernel[9];
  coeff_5[2] = Kernel[14];
  coeff_5[3] = Kernel[19];
  coeff_6 = Kernel[24];

  mask0 = (v4s){1, 2, 3, 4};

  int offset = 0;

  // image board is black
#ifdef TILING2D
  int xBlockSize = (ROW) / 2;
  int yBlockSize = ((COL) + (NUM_CORES / 2) - 1) / (NUM_CORES / 2);
  int x = (core_id / (NUM_CORES / 2)) * xBlockSize;
  int y = (core_id % (NUM_CORES / 2)) * yBlockSize;
  offset = x * Input_cl;

  for (col = y; (y < COL) && (col < y + yBlockSize); col++)
  {

#else

  int blockSize = ((COL) + NUM_CORES - 1) / NUM_CORES;
  int start = pi_core_id() * blockSize;
  int end = start + blockSize < COL ? start + blockSize : COL;
  for (col = start; col < end; col++)
  {
#endif

    Img_0 = *((INP_VTYPE *)(&In_Img[col + offset]));
    Img_1 = *((INP_VTYPE *)(&In_Img[col + Input_cl + offset]));
    Img_2 = *((INP_VTYPE *)(&In_Img[col + 2 * Input_cl + offset]));
    Img_3 = *((INP_VTYPE *)(&In_Img[col + 3 * Input_cl + offset]));
    Img_4 = *((INP_VTYPE *)(&In_Img[col + 4 * Input_cl + offset]));
    Img_5[0] = In_Img[col + 4 + offset];
    Img_5[1] = In_Img[col + 4 + Input_cl + offset];
    Img_5[2] = In_Img[col + 4 + 2 * Input_cl + offset];
    Img_5[3] = In_Img[col + 4 + 3 * Input_cl + offset];
    Img_6 = In_Img[col + 4 + 4 * Input_cl + offset];
    // Img_12[1]  = 0;

#ifdef TILING2D
    for (row = x; (row < ROW) && (row < x + xBlockSize); row++)
    {
#else
    for (row = 0; row < ROW; row++)
    {
#endif
      t = (row)*ROW + col;
      #ifdef MIXED_VECTOR
      acc = 0;
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_0), "f"(coeff_0) :);
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_1), "f"(coeff_1) :);
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_2), "f"(coeff_2) :);
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_3), "f"(coeff_3) :);
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_4), "f"(coeff_4) :);
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(acc) : "f"(Img_5), "f"(coeff_5) :);
      acc += (float)Img_6 * (float)coeff_6;
      #else
      temp = (OUT_VTYPE){0, 0, 0, 0};
      // asm volatile("":::"memory");
      temp += Img_0 * coeff_0;

      temp += Img_1 * coeff_1;
      temp += Img_2 * coeff_2;
      temp += Img_3 * coeff_3;
      temp += Img_4 * coeff_4;
      temp += Img_5 * coeff_5;
      temp[0] += Img_6 * coeff_6;
      /*temp += Img_7 * coeff_7;
      temp += Img_8 * coeff_8;
      temp += Img_9 * coeff_9;
      temp += Img_10 *coeff_10;
      temp += Img_11 *coeff_11;
      temp += Img_12 *coeff_12;*/
      acc = temp[0] + temp[1] + temp[2] + temp[3];
      #endif

      Out_Img[t] = (OUT_TYPE)acc;

      new_data1 = *((INP_VTYPE *)(&In_Img[col + (row + 5) * Input_cl]));
      // new_data2 = *((INP_VTYPE *) (&In_Img[col+2+(row+5)*Input_cl]));
      new_data2 = In_Img[col + 4 + (row + 5) * Input_cl];
      // new_data3[1] = 0;

      // Move the window
      /*
        thirteen vectors:

        Img_0  = {A0, A1}
        Img_1  = {B0, B1}
        Img_2  = {C0, C1}
        Img_3  = {D0, D1}
        Img_4  = {E0, E1}
        Img_5  = {F0, F1}
        Img_6  = {G0, G1}
        Img_7  = {H0, H1}
        Img_8  = {I0, I1}
        Img_9  = {J0, J1}
        Img_10 = {K0, K1}
        Img_11 = {L0, L1}
        Img_12 = {M0,  0}

        Current Windonw:
        XX XX XX XX XX
        A0 A1 B0 B1 K0
        C0 C1 D0 D1 K1
        E0 E1 F0 F1 L0
        G0 G1 H0 H1 L1
        I0 I1 J0 J1 M0
        N0 N1 P0 P1 M1
        XX XX XX XX XX

        We want to load next line (N0, N1, P0, P1, M1)
        in vectors new_data1 and new_data2
        new_data1 = {N0, N1}
        new_data2 = {P0, P1}
        new_data3 = {M1,  0}

        Move each vector one line down and shuffle the vertical vector

        Img_0  = Img_2
        Img_1  = Img_3
        Img_2  = Img_4
        Img_3  = Img_5
        Img_4  = Img_6
        Img_5  = Img_7
        Img_6  = Img_8
        Img_7  = Img_9
        Img_8  = new_data1
        Img_9  = new_data2
        Img_10 = {K1, L0}
        Img_11 = {L1, M0}
        Img_12 = new_data3
      */

      Img_0 = Img_1;
      Img_1 = Img_2;
      Img_2 = Img_3;
      Img_3 = Img_4;
      Img_4 = new_data1;

      Img_5 = (INP_VTYPE)__builtin_shuffle(Img_5, mask0);

      Img_5[3] = Img_6;

      Img_6 = new_data2;
    }
#ifdef TILING2D
    // last iteration
    t = (row + 1) * ROW + col + 2;

    OUT_VTYPE temp;
    temp = Img_0 * coeff_0;
    temp += Img_1 * coeff_1;
    temp += Img_2 * coeff_2;
    temp += Img_3 * coeff_3;
    temp += Img_4 * coeff_4;
    temp += Img_5 * coeff_5;
    temp += Img_6 * coeff_6;
    temp += Img_7 * coeff_7;
    temp += Img_8 * coeff_8;
    temp += Img_9 * coeff_9;
    temp += Img_10 * coeff_10;
    temp += Img_11 * coeff_11;
    temp += Img_12 * coeff_12;
    acc = temp[0] + temp[1];

    Out_Img[t] = (OUT_TYPE)acc;
#endif
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif

#endif
}
#endif // FP8 5x5
#endif // FP8