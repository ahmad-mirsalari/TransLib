#include "pmsis.h"



#include <stdio.h>
#ifndef MIXED_RADIX
#include "fft.h"
#else
#include "../fft_mixed_2_8/fft_mixed.h" 
#endif
#include "twiddle_factor.h"


#ifdef BITREV_LUT
#include "bit_reverse.h"
#endif

double __attribute__ ((used)) __extendohfdf2(float16alt value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.ah %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}

double  __attribute__ ((used)) __extendhfdf2(float16 value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.h %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}


#ifdef VECTORIZATION
DATA_LOCATION static VDTYPE MONE_ONE = (VDTYPE) {-1, 1};
#endif

#ifdef VECTORIZATION
Complex_type __attribute__ ((always_inline)) complex_mul (Complex_type A, Complex_type B)
{
  Complex_type c_tmp;

  // c_tmp.re = A.re * B.re - A.im * B.im;
  // c_tmp.im = A.re * B.im + A.im * B.re;

  VDTYPE P0, P1, P2, P3;

  P0 = A.v * B.v;
  B.v = __builtin_pulp_shuffleh(B.v, (v2s){1,0});
  P1 = A.v * B.v;
  P2 = __builtin_pulp_shuffle2h(P0, P1, (v2s) {0,2});
  P3 = __builtin_pulp_shuffle2h(P0, P1, (v2s) {1,3});
  P3 = P3 * MONE_ONE;
  c_tmp.v = P2 + P3;
  return c_tmp;
}
#else
Complex_type __attribute__ ((always_inline)) complex_mul (Complex_type A, Complex_type B)
{
  Complex_type c_tmp;

  c_tmp.re = A.re * B.re - A.im * B.im;
  c_tmp.im = A.re * B.im + A.im * B.re;

  return c_tmp;
}
#endif

/*
Complex_type __attribute__ ((inline)) complex_mul_N_4 (Complex_type B) //caso indice twiddle N/4
{
  Complex_type c_tmp;

  c_tmp.re = B.im;
  c_tmp.im = - B.re;
  return c_tmp;
}

#define R2_2        0.707107f  //radice di 2/2 (f perché è DTYPE)

Complex_type __attribute__ ((inline)) complex_mul_N_8 (Complex_type B) //caso N/8
{
  Complex_type c_tmp;

  c_tmp.re =  R2_2 * B.im + R2_2 * B.re;
  c_tmp.im =  R2_2 * B.im - R2_2 * B.re;

  return c_tmp;
}

Complex_type __attribute__ ((inline)) complex_mul_3N_8 (Complex_type B) //caso 3N/8
{
  Complex_type c_tmp;

  c_tmp.re = -R2_2 * B.re + R2_2 * B.im;
  c_tmp.im = -R2_2 * B.re - R2_2 * B.im;

  return c_tmp;
}


Complex_type __attribute__ ((inline)) complex_mul_0 (Complex_type B)
{
  return B;
}
typedef Complex_type (*complex_mul_x) (Complex_type B);

RT_L1_DATA complex_mul_x mul_functions[] = {complex_mul_0, complex_mul_N_8, complex_mul_N_4, complex_mul_3N_8};
*/

#ifdef VECTORIZATION
Complex_type __attribute__ ((always_inline)) complex_mul_real (DTYPE A, Complex_type B)
{
  Complex_type c_tmp;
  // c_tmp.re = A * B.re;
  // c_tmp.im = A * B.im;
  c_tmp.v = (VDTYPE) {A, A} * B.v;
  return c_tmp;
}
#else
Complex_type __attribute__ ((always_inline)) complex_mul_real (DTYPE A, Complex_type B)
{
  Complex_type c_tmp;
  c_tmp.re = A * B.re;
  c_tmp.im = A * B.im;
  return c_tmp;
}
#endif

//FFT_LEN_RADIX2=2^11=2048 --address=11 --digit=1
int __attribute__ ((always_inline)) bit_rev_radix2(int index_torevert)  //bit reverse total number
{
  unsigned int revNum = 0;
  unsigned i;

  for (i=0; i<LOG2_FFT_LEN_RADIX2; i++)
  {
    unsigned int temp = (index_torevert & (1 << i));
    if (temp != 0) //if (temp)
      revNum |= (1 << ((LOG2_FFT_LEN_RADIX2 -1) - i));
  }

  return revNum;
 }


void __attribute__ ((always_inline)) process_butterfly_real_radix2 (Complex_type* input, int twiddle_index, int distance, Complex_type* twiddle_ptr) //1^stage
{
  int index = 0;
  DTYPE d0         = input[index].re;
  DTYPE d1         = input[index+distance].re;

  //printf("index %d d0 %f d1 %f \n",twiddle_index, d0,d1);  
  Complex_type r0, r1;

  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im, since c1 is real = c1.re*c2.re
  r0.re = d0 + d1;
  r1.re = d0 - d1;

  Complex_type tw0 = twiddle_ptr[twiddle_index];
  // printf("core if %d twiddle %f imag %f \n", pi_core_id(),tw0.re,tw0.im); 

  // input[index]            = complex_mul_real(r0.re,tw0);
  input[index]            = r0;
  // printf("index %d input %f twiddle %f , %f \n\n", twiddle_index,r1.re, tw0.re,tw0.im);

  input[index+distance]   = complex_mul_real(r1.re,tw0);
   //printf("index + distance %d \n", index+distance);
  //printf("index %d output %f s %f \n\n", twiddle_index,input[index+distance].re, input[index+distance].im);
  //printf("%d \t %f \t %f \n", twiddle_index,input[index+distance].re, input[index+distance].im);

}

void __attribute__ ((always_inline)) process_butterfly_radix2 (Complex_type* input, int twiddle_index, int index, int distance, Complex_type* twiddle_ptr)
{
  Complex_type r0, r1;

#ifdef VECTORIZATION
  VDTYPE v0         = input[index].v;
  VDTYPE v1         = input[index+distance].v;

  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im
  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

  r0.v = v0 + v1;
  r1.v = v0 - v1;

#else
  DTYPE d0         = input[index].re;
  DTYPE d1         = input[index+distance].re;
  DTYPE e0         = input[index].im;
  DTYPE e1         = input[index+distance].im;

  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im

  r0.re = d0 + d1;
  r1.re = d0 - d1;

  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

  r0.im = e0 + e1;
  r1.im = e0 - e1;
#endif

  Complex_type tw0 = twiddle_ptr[twiddle_index];

  input[index]           = r0;
  input[index+distance]  = complex_mul(tw0, r1);
}

void __attribute__ ((always_inline)) process_butterfly_last_radix2 (Complex_type* input, Complex_type* output, int outindex ) //uscita
{
  int index = 0;
  Complex_type r0, r1;

#ifdef VECTORIZATION
  VDTYPE v0  = input[index].v;
  VDTYPE v1  = input[index+1].v;


  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im
  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

  r0.v = v0 + v1;
  r1.v = v0 - v1;


#else
  DTYPE d0  = input[index].re;
  DTYPE d1  = input[index+1].re;
  DTYPE e0  = input[index].im;
  DTYPE e1  = input[index+1].im;


  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im

  r0.re = d0 + d1;
  r1.re = d0 - d1;

  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

  r0.im = e0 + e1;
  r1.im = e0 - e1;
#endif

  /* In the Last step, twiddle factors are all 1 */
#ifndef SORT_OUTPUT
#ifdef BITREV_LUT
  unsigned int index12 = *((unsigned int *)(&bit_rev_radix2_LUT[outindex]));
  unsigned int index1  = index12 & 0x0000FFFF;
  unsigned int index2  = index12 >> 16;
  output[index1] = r0;
  output[index2] = r1;
#else // !BITREV_LUT
  output[bit_rev_radix2(outindex  )] = r0;
  output[bit_rev_radix2(outindex+1)] = r1;
#endif // BITREV_LUT
#else // !SORT_OUTPUT
  output[outindex  ] = r0;
  output[outindex+1] = r1;
#endif // SORT_OUTPUT
}

void __attribute__ ((noinline)) fft_radix2 (Complex_type * Inp_signal, Complex_type * Out_signal)
{
  int k,j,stage, step, d, index;
  Complex_type * _in;
  Complex_type * _out;
  Complex_type  temp;
  int dist = FFT_LEN_RADIX2 >> 1;
  int nbutterfly = FFT_LEN_RADIX2 >> 1;
  int butt = 2; //number of butterfly in the same group
  Complex_type * _in_ptr;
  Complex_type * _out_ptr;
  Complex_type * _tw_ptr;
  _in  = &(Inp_signal[0]);
  _out = &(Out_signal[0]);

  // FIRST STAGE, input is real, stage=1
  stage = 1;

  _in_ptr = _in;
  _tw_ptr = twiddle_factors;

  for(j=0;j<nbutterfly;j++)
  {
    process_butterfly_real_radix2(_in_ptr, j, dist, _tw_ptr);
    _in_ptr++;
  } //j

  stage = stage + 1;
  dist  = dist >> 1;

  // STAGES 2 -> n-1
  while(dist > 1)
  {
    step = dist << 1;
    for(j=0;j<butt;j++)
    {
      _in_ptr = _in;
      for(d = 0; d < dist; d++)
      {
         process_butterfly_radix2(_in_ptr, d*butt, j*step, dist, _tw_ptr);
         _in_ptr++;
       } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 1;
    butt = butt  << 1;
  }

  // LAST STAGE
  _in_ptr = _in;
  index=0;
  for(j=0;j<FFT_LEN_RADIX2>>1;j++)
  {
    process_butterfly_last_radix2(_in_ptr, _out, index);
    _in_ptr +=2;
    index   +=2;
   } //j

   // ORDER VALUES
#ifdef SORT_OUTPUT
  for(j=0; j<FFT_LEN_RADIX2; j+=4)
  {
#ifdef BITREV_LUT
    unsigned int index12 = *((unsigned int *)(&bit_rev_radix2_LUT[j]));
    unsigned int index34 = *((unsigned int *)(&bit_rev_radix2_LUT[j+2]));
    unsigned int index1  = index12 & 0x0000FFFF;
    unsigned int index2  = index12 >> 16;
    unsigned int index3  = index34 & 0x0000FFFF;
    unsigned int index4  = index34 >> 16;
#else
    int index1 = bit_rev_radix2(j);
    int index2 = bit_rev_radix2(j+1);
    int index3 = bit_rev_radix2(j+2);
    int index4 = bit_rev_radix2(j+3);
#endif
    if(index1 > j)
    {
      temp         = _out[j];
      _out[j]      = _out[index1];
      _out[index1] = temp;
    }
    if(index2 > j+1)
    {
      temp         = _out[j+1];
      _out[j+1]    = _out[index2];
      _out[index2] = temp;
    }
    if(index3 > j+2)
    {
      temp         = _out[j+2];
      _out[j+2]    = _out[index3];
      _out[index3] = temp;
    }
    if(index4 > j+3)
    {
      temp         = _out[j+3];
      _out[j+3]    = _out[index4];
      _out[index4] = temp;
    }
  }
#endif // SORT_OUTPUT
}

// PARALLEL VERSION
void __attribute__ ((noinline)) par_fft_radix2 (Complex_type * Inp_signal, Complex_type * Out_signal)
{
  printf("in parallel version \n");
  int k, j, stage, step, d, index;
  Complex_type * _in;
  Complex_type * _out;
  Complex_type  temp;
  int dist = FFT_LEN_RADIX2 >> 1;
  int nbutterfly = FFT_LEN_RADIX2 >> 1;
  int butt = 2;
  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif
  Complex_type * _in_ptr;
  Complex_type * _out_ptr;
  Complex_type * _tw_ptr;

  _in  = &(Inp_signal[0]);
  _out = &(Out_signal[0]);

  // FIRST STAGE
  _in_ptr = &_in[core_id];
  _tw_ptr = twiddle_factors;
  stage = 1;

  for(j = 0; j < nbutterfly/NUM_CORES; j++)
  {
    process_butterfly_real_radix2(_in_ptr, j*NUM_CORES+core_id, dist, _tw_ptr);
    _in_ptr+=NUM_CORES;
  } //j

  stage = stage + 1;
  dist  = dist >> 1;

  // STAGES 2 -> n-1
  while(dist > NUM_CORES/2)
  {
    pi_cl_team_barrier();
    step = dist << 1;
    for(j=0; j < butt; j++)
    {
      _in_ptr = &_in[core_id];
      for(d = 0; d < dist/NUM_CORES; d++)
      {
        process_butterfly_radix2(_in_ptr, (d*NUM_CORES+core_id)*butt, j*step,dist, _tw_ptr);
        _in_ptr+=NUM_CORES;
       } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 1;
    butt = butt << 1;
  }

  while(dist > 1)
  {
    pi_cl_team_barrier();
    step = dist << 1;
    for(j = 0; j < butt/NUM_CORES; j++)
    {
      _in_ptr = _in;
      for(d = 0; d < dist; d++)
      {
        process_butterfly_radix2(_in_ptr, butt*d, (j*NUM_CORES+core_id)*step, dist, _tw_ptr);
        _in_ptr++;
      } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 1;
    butt = butt  << 1;
  }

  pi_cl_team_barrier();

  // LAST STAGE
  _in_ptr  = &_in[2*core_id];
  index    = 2*core_id;
  //N must be at least 64 for multicore
  for(j = 0; j < FFT_LEN_RADIX2/(2*NUM_CORES); j++)
  {
    process_butterfly_last_radix2(_in_ptr,_out,index);
    _in_ptr +=2*NUM_CORES;
    index   +=2*NUM_CORES;
  } //j

#ifdef SORT_OUTPUT
  pi_cl_team_barrier();

  for(j = 4*core_id; j < FFT_LEN_RADIX2; j+=NUM_CORES*4)
  {
#ifdef BITREV_LUT
    unsigned int index12 = *((unsigned int *)(&bit_rev_radix2_LUT[j]));
    unsigned int index34 = *((unsigned int *)(&bit_rev_radix2_LUT[j+2]));
    unsigned int index1      = index12 & 0x0000FFFF;
    unsigned int index2      = index12 >> 16;
    unsigned int index3      = index34 & 0x0000FFFF;
    unsigned int index4      = index34 >> 16;
#else
    int index1 = bit_rev_radix2(j);
    int index2 = bit_rev_radix2(j+1);
    int index3 = bit_rev_radix2(j+2);
    int index4 = bit_rev_radix2(j+3);
#endif
    if(index1 > j)
    {
      temp         = _out[j];
      _out[j]      = _out[index1];
      _out[index1] = temp;
    }
    if(index2 > j+1)
    {
      temp         = _out[j+1];
      _out[j+1]    = _out[index2];
      _out[index2] = temp;
    }
    if(index3 > j+2)
    {
      temp         = _out[j+2];
      _out[j+2]    = _out[index3];
      _out[index3] = temp;
    }
    if(index4 > j+3)
    {
      temp         = _out[j+3];
      _out[j+3]    = _out[index4];
      _out[index4] = temp;
    }
  }
  pi_cl_team_barrier();
#endif // SORT_OUTPUT
}
