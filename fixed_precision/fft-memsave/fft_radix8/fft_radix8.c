#include "pmsis.h"

#include <stdio.h>


#ifndef MIXED_RADIX
#include "fft.h"
#endif

#if defined(MIXED_RADIX) && !defined(FULL_TWIDDLES)
extern Complex_type twiddle_factors[];
#else
#include "twiddle_factor.h"
#endif

#ifdef BITREV_LUT
#include "bit_reverse.h"
#endif



DATA_LOCATION DTYPE  ROT_CONST   = 0.707106781f;
#ifdef VECTORIZATION
DATA_LOCATION VDTYPE ROT_CONST_V = (VDTYPE) {0.707106781f, 0.707106781f};
DATA_LOCATION VDTYPE ONE_MONE = (VDTYPE) {1.0f, -1.0f};
DATA_LOCATION VDTYPE MONE_ONE = (VDTYPE) {-1.0f, 1.0f};
#endif


#ifdef VECTORIZATION
static Complex_type __attribute__ ((always_inline)) complex_mul (Complex_type A, Complex_type B)
{
    Complex_type c_tmp;
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
static Complex_type __attribute__ ((always_inline)) complex_mul (Complex_type A, Complex_type B)
{
  Complex_type c_tmp;

  c_tmp.re = A.re * B.re - A.im * B.im;
  c_tmp.im = A.re * B.im + A.im * B.re;
  return c_tmp;
}
#endif

#ifdef VECTORIZATION
static Complex_type __attribute__ ((always_inline)) complex_mul_real (DTYPE A, Complex_type B)
{
    Complex_type c_tmp;
    // c_tmp.re = A * B.re;
    // c_tmp.im = A * B.im;
    c_tmp.v = (VDTYPE) {A, A} * B.v;
    return c_tmp;
}
#else
static Complex_type __attribute__ ((always_inline)) complex_mul_real (DTYPE A, Complex_type B)
{
  Complex_type c_tmp;
  c_tmp.re = A * B.re;
  c_tmp.im = A * B.im;
  return c_tmp;
}
#endif


int __attribute__ ((always_inline)) bit_rev_2_8(int value)
{
  int i;
  unsigned int new_value = 0;

  for (i = 0; i < LOG2_FFT_LEN_RADIX8/3; i++)
  {
    new_value <<= 3;
    new_value |= (value & 0x7);
    value >>= 3;
  }
#ifdef STAGE_2_2_8
  new_value = (new_value << 2) | ((value & 0x2) >> 1) | ((value & 0x1) << 1);
#else
  new_value = (new_value << 1) | value;
#endif
  return new_value;
}

int __attribute__ ((always_inline)) bit_rev_radix8(int value) //digit reverse
{
  int i;
  unsigned int new_value = 0;
  for (i = 0; i < LOG2_FFT_LEN_RADIX8/3; i++)
  {
    new_value <<= 3;
    new_value |= (value & 0x7);
    value >>= 3;
  }
  return new_value;
}


void __attribute__ ((always_inline)) process_butterfly_real_radix8 (Complex_type* input, int twiddle_index, int distance, Complex_type* twiddle_ptr)
{
  DTYPE d0, d1, d2, d3, d4, d5, d6, d7;
  DTYPE e0, e1, e2, e3, e4, e5, e6, e7;

  Complex_type tw1, tw2, tw3, tw4, tw5, tw6, tw7;

  Complex_type r0, r1, r2, r3, r4, r5, r6, r7;

  int index = 0;

#if defined(MIXED_RADIX) && !defined(FULL_TWIDDLES)
  twiddle_index = twiddle_index * 4;
  #define MAX_TWIDDLES (FFT_LEN_RADIX2)
#elif defined(FULL_TWIDDLES)
  #define MAX_TWIDDLES (FFT_LEN_RADIX8)
#else
  #define MAX_TWIDDLES (FFT_LEN_RADIX8/2)
#endif


  #ifdef VECTORIZATION
  VDTYPE v0         = input[index].v;
  VDTYPE v1         = input[index+distance].v;
  VDTYPE v2         = input[index+2*distance].v;
  VDTYPE v3         = input[index+3*distance].v;
  VDTYPE v4         = input[index+4*distance].v;
  VDTYPE v5         = input[index+5*distance].v;
  VDTYPE v6         = input[index+6*distance].v;
  VDTYPE v7         = input[index+7*distance].v;

  VDTYPE t0, t1, t2, t3, t4, t5, t6, t7;
  VDTYPE q0, q1, q2, q3, q4, q5, q6, q7;
  VDTYPE s0, s1, s2, s3, s4, s5, s6, s7;
  //
  t0 = v0 + v4;
  t1 = v1 + v5;
  t2 = v2 + v6;
  t3 = v3 + v7;
  t4 = v0 - v4;
  t5 = v1 - v5;
  t6 = v2 - v6;
  t7 = v3 - v7;

  q0 = t0 + t2;
  q1 = t1 + t3;
  q2 = t0 - t2;
  q3 = t1 - t3;
  q4 = t4;
  q5 = t5 + t7;
  q6 = t6;
  q7 = t5 - t7;


  s0 = q0 + q1;
  s1 = q0 - q1;
  s2 = q2 + __builtin_shuffle(q3, (v2s){1,0})*ONE_MONE;
  s3 = q2 + __builtin_shuffle(q3, (v2s){1,0})*MONE_ONE;

  s4 = q4 + ROT_CONST_V * (__builtin_shuffle(q5, (v2s){1,0}))*ONE_MONE;
  s5 = q4 + ROT_CONST_V * (__builtin_shuffle(q5, (v2s){1,0}))*MONE_ONE;
  s6 = ROT_CONST_V * q7 + (__builtin_shuffle(q6, (v2s){1,0}))*ONE_MONE;
  s7 = ROT_CONST_V * q7 + (__builtin_shuffle(q6, (v2s){1,0}))*MONE_ONE;


  r0.v = s0;
  r1.v = s4 + s6;
  r2.v = s2;
  r3.v = s4 - s6;
  r4.v = s1;
  r5.v = s5 - s7;
  r6.v = s3;
  r7.v = s5 + s7;


  #else

  d0         = input[index].re;
  d1         = input[index+distance].re;
  d2         = input[index+2*distance].re;
  d3         = input[index+3*distance].re;
  d4         = input[index+4*distance].re;
  d5         = input[index+5*distance].re;
  d6         = input[index+6*distance].re;
  d7         = input[index+7*distance].re;

  // Basic buttefly rotation
  /*

r0   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i
r1   1.0000 + 0.0000i   0.7071 - 0.7071i   0.0000 - 1.0000i  -0.7071 - 0.7071i  -1.0000 - 0.0000i  -0.7071 + 0.7071i   0.0000 + 1.0000i   0.7071 + 0.7071i
r2   1.0000 + 0.0000i   0.0000 - 1.0000i  -1.0000 - 0.0000i   0.0000 + 1.0000i   1.0000 + 0.0000i   0.0000 - 1.0000i   -1.0000 - 0.0000i  0.0000 + 1.0000i
r3   1.0000 + 0.0000i  -0.7071 - 0.7071i   0.0000 + 1.0000i   0.7071 - 0.7071i  -1.0000 - 0.0000i   0.7071 + 0.7071i   0.0000 - 1.0000i  -0.7071 + 0.7071i

r4   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i
r5   1.0000 + 0.0000i  -0.7071 + 0.7071i   0.0000 - 1.0000i   0.7071 + 0.7071i  -1.0000 - 0.0000i   0.7071 - 0.7071i   0.0000 + 1.0000i  -0.7071 - 0.7071i
r6   1.0000 + 0.0000i   0.0000 + 1.0000i  -1.0000 - 0.0000i   0.0000 - 1.0000i   1.0000 + 0.0000i   0.0000 + 1.0000i  -1.0000 - 0.0000i   0.0000 - 1.0000i
r7   1.0000 + 0.0000i   0.7071 + 0.7071i  -0.0000 + 1.0000i  -0.7071 + 0.7071i  -1.0000 - 0.0000i  -0.7071 - 0.7071i   0.0000 - 1.0000i   0.7071 - 0.7071i

  */

  DTYPE d13   = d1 + d3;
  DTYPE d1_3   = d1 - d3;
  DTYPE d26   = d2 + d6;
  DTYPE d57   = d5 + d7;
  DTYPE d_57   = d7 - d5;
  DTYPE temp1 = ROT_CONST*(d1_3+d_57);
  DTYPE temp2 = ROT_CONST*(d57-d13);
  DTYPE d04   = d0 + d4;
  DTYPE d0_4  = d0 - d4;
  DTYPE d0246 = d04 + d26;
  DTYPE d_26  = d6 - d2;

  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im, since c1 is real = c1.re*c2.re
  r0.re = d0246 + d13 + d57;
  //r1.re = d0 + ROT_CONST*d1 - ROT_CONST*d3 - d4 - ROT_CONST*d5 + ROT_CONST*d7;
  r1.re = d0_4 + temp1;
  r2.re = d04 - d26;
  //r3.re = d0 - ROT_CONST*d1 + ROT_CONST*d3 - d4 + ROT_CONST*d5 - ROT_CONST*d7;
  r3.re = d0_4 - temp1;
  r4.re = d0246 - d13 - d57;
  r5.re = r3.re;
  r6.re = r2.re;
  r7.re = r1.re;

  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re, since c1 is real = c1.re*c2.im
  r0.im = 0.0f;
  //r1.im = -ROT_CONST*d1 - d2 - ROT_CONST*d3 + ROT_CONST*d5 + d6 + ROT_CONST*d7;
  r1.im = d_26 + temp2;
  r2.im = d_57 - d1_3;
  //r3.im = -ROT_CONST*d1 + d2 - ROT_CONST*d3 + ROT_CONST*d5 - d6 + ROT_CONST*d7;
  r3.im = temp2 - d_26 ;
  r4.im = 0.0f;
  r5.im = -r3.im;
  r6.im = -r2.im;
  r7.im = -r1.im;

  #endif

  tw1 = twiddle_ptr[twiddle_index*1];
  tw2 = twiddle_ptr[twiddle_index*2];
  tw3 = twiddle_ptr[twiddle_index*3];
  tw4 = twiddle_ptr[twiddle_index*4];
#ifdef FULL_TWIDDLES
  tw5 = twiddle_ptr[twiddle_index*5];
  tw6 = twiddle_ptr[twiddle_index*6];
  tw7 = twiddle_ptr[twiddle_index*7];
#else
  if(twiddle_index*5 >= MAX_TWIDDLES)
  {
    tw5 = twiddle_ptr[(twiddle_index*5)%(MAX_TWIDDLES)];
    #ifdef VECTORIZATION
    tw5.v = -tw5.v;
    #else
    tw5.re = -tw5.re ;
    tw5.im = -tw5.im ;
    #endif
  }
  else
  {
    tw5 = twiddle_ptr[twiddle_index*5];
  }
  if(twiddle_index*6 >= MAX_TWIDDLES)
  {
    tw6 = twiddle_ptr[(twiddle_index*6)%(MAX_TWIDDLES)];
    #ifdef VECTORIZATION
    tw6.v = -tw6.v;
    #else
    tw6.re = -tw6.re ;
    tw6.im = -tw6.im ;
    #endif
  }
  else
  {
    tw6 = twiddle_ptr[twiddle_index*6];
  }
  if(twiddle_index*7 >= MAX_TWIDDLES)
  {
    #ifdef VECTORIZATION
    tw7.v = -tw7.v;
    #else
    tw7.re = -tw7.re ;
    tw7.im = -tw7.im ;
    #endif
  }
  else
  {
    tw7 = twiddle_ptr[twiddle_index*7];
  }
#endif // FULL_TWIDDLES

  input[index]            = r0;
  input[index+distance]   = complex_mul(tw1, r1);
  input[index+2*distance] = complex_mul(tw2, r2);
  input[index+3*distance] = complex_mul(tw3, r3);
  input[index+4*distance] = complex_mul_real(r4.re,tw4);
  input[index+5*distance] = complex_mul(tw5, r5);
  input[index+6*distance] = complex_mul(tw6, r6);
  input[index+7*distance] = complex_mul(tw7, r7);
}



void __attribute__ ((always_inline)) process_butterfly_radix8 (Complex_type* input, int twiddle_index, int index, int distance, Complex_type* twiddle_ptr)
{
  DTYPE d0, d1, d2, d3, d4, d5, d6, d7;
  DTYPE e0, e1, e2, e3, e4, e5, e6, e7;

  Complex_type tw1, tw2, tw3, tw4, tw5, tw6, tw7;

  Complex_type r0, r1, r2, r3, r4, r5, r6, r7;

#if defined(MIXED_RADIX) && !defined(FULL_TWIDDLES)
  twiddle_index = twiddle_index * 4;
  #define MAX_TWIDDLES (FFT_LEN_RADIX2)
#elif defined(FULL_TWIDDLES)
  #define MAX_TWIDDLES (FFT_LEN_RADIX8)
#else
  #define MAX_TWIDDLES (FFT_LEN_RADIX8/2)
#endif

#ifdef VECTORIZATION
  VDTYPE v0         = input[index].v;
  VDTYPE v1         = input[index+distance].v;
  VDTYPE v2         = input[index+2*distance].v;
  VDTYPE v3         = input[index+3*distance].v;
  VDTYPE v4         = input[index+4*distance].v;
  VDTYPE v5         = input[index+5*distance].v;
  VDTYPE v6         = input[index+6*distance].v;
  VDTYPE v7         = input[index+7*distance].v;

  VDTYPE t0, t1, t2, t3, t4, t5, t6, t7;
  VDTYPE q0, q1, q2, q3, q4, q5, q6, q7;
  VDTYPE s0, s1, s2, s3, s4, s5, s6, s7;
  //
  t0 = v0 + v4;
  t1 = v1 + v5;
  t2 = v2 + v6;
  t3 = v3 + v7;
  t4 = v0 - v4;
  t5 = v1 - v5;
  t6 = v2 - v6;
  t7 = v3 - v7;

  q0 = t0 + t2;
  q1 = t1 + t3;
  q2 = t0 - t2;
  q3 = t1 - t3;
  q4 = t4;
  q5 = t5 + t7;
  q6 = t6;
  q7 = t5 - t7;


  s0 = q0 + q1;
  s1 = q0 - q1;
  s2 = q2 + __builtin_shuffle(q3, (v2s){1,0})*ONE_MONE;
  s3 = q2 + __builtin_shuffle(q3, (v2s){1,0})*MONE_ONE;

  s4 = q4 + ROT_CONST_V * (__builtin_shuffle(q5, (v2s){1,0}))*ONE_MONE;
  s5 = q4 + ROT_CONST_V * (__builtin_shuffle(q5, (v2s){1,0}))*MONE_ONE;
  s6 = ROT_CONST_V * q7 + (__builtin_shuffle(q6, (v2s){1,0}))*ONE_MONE;
  s7 = ROT_CONST_V * q7 + (__builtin_shuffle(q6, (v2s){1,0}))*MONE_ONE;


  r0.v = s0;
  r1.v = s4 + s6;
  r2.v = s2;
  r3.v = s4 - s6;
  r4.v = s1;
  r5.v = s5 - s7;
  r6.v = s3;
  r7.v = s5 + s7;


  #else
  //int twiddle_index = 0;
  d0         = input[index].re;
  d1         = input[index+distance].re;
  d2         = input[index+2*distance].re;
  d3         = input[index+3*distance].re;
  d4         = input[index+4*distance].re;
  d5         = input[index+5*distance].re;
  d6         = input[index+6*distance].re;
  d7         = input[index+7*distance].re;

  e0         = input[index].im;
  e1         = input[index+distance].im;
  e2         = input[index+2*distance].im;
  e3         = input[index+3*distance].im;
  e4         = input[index+4*distance].im;
  e5         = input[index+5*distance].im;
  e6         = input[index+6*distance].im;
  e7         = input[index+7*distance].im;


  // Basic buttefly rotation
  /*

  r0   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i
  r1   1.0000 + 0.0000i   0.7071 - 0.7071i   0.0000 - 1.0000i  -0.7071 - 0.7071i  -1.0000 - 0.0000i  -0.7071 + 0.7071i   0.0000 + 1.0000i   0.7071 + 0.7071i
  r2   1.0000 + 0.0000i   0.0000 - 1.0000i  -1.0000 - 0.0000i   0.0000 + 1.0000i   1.0000 + 0.0000i   0.0000 - 1.0000i   -1.0000 - 0.0000i  0.0000 + 1.0000i
  r3   1.0000 + 0.0000i  -0.7071 - 0.7071i  -0.0000 + 1.0000i   0.7071 - 0.7071i  -1.0000 - 0.0000i   0.7071 + 0.7071i   0.0000 - 1.0000i  -0.7071 + 0.7071i
  r4   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i
  r5   1.0000 + 0.0000i  -0.7071 + 0.7071i   0.0000 - 1.0000i   0.7071 + 0.7071i  -1.0000 - 0.0000i   0.7071 - 0.7071i  -0.0000 + 1.0000i  -0.7071 - 0.7071i
  r6   1.0000 + 0.0000i   0.0000 + 1.0000i  -1.0000 - 0.0000i   0.0000 - 1.0000i   1.0000 + 0.0000i   0.0000 + 1.0000i  -1.0000 - 0.0000i   0.0000 - 1.0000i
  r7   1.0000 + 0.0000i   0.7071 + 0.7071i  -0.0000 + 1.0000i  -0.7071 + 0.7071i  -1.0000 - 0.0000i  -0.7071 - 0.7071i   0.0000 - 1.0000i   0.7071 - 0.7071i
  */

  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im


  DTYPE d13 = d1 + d3;
  DTYPE d1_3 = d1 - d3;
  DTYPE e13 = e1 + e3;
  DTYPE e57 = e5 + e7;
  DTYPE e1_3 = e1 - e3;
  DTYPE e_57 = e7 - e5;
  DTYPE d57 = d5 + d7;
  DTYPE d_57 = d7 - d5;
  DTYPE temp1 = ROT_CONST*(d1_3+d_57);
  DTYPE temp1b = ROT_CONST*(e57 - e13);
  DTYPE temp2 = ROT_CONST*(d57-d13);
  DTYPE temp2b = ROT_CONST*(e1_3 + e_57);
  DTYPE d04   = d0 + d4;
  DTYPE d0_4  = d0 - d4;
  DTYPE d26  = d2 + d6;
  DTYPE d_26  = d6 - d2;
  DTYPE d0246 = d04 + d26;
  DTYPE d1357 = d13 + d57;
  DTYPE e0246 = e0 + e2 + e4 + e6;
  DTYPE e0_4  = e0 - e4;
  DTYPE e0_24_6 = e0 - e2 + e4 - e6;
  DTYPE e1357 = e13 + e57;
  DTYPE e_13_57 = e_57 - e1_3;
  DTYPE e2_6 = e2 - e6 ;
  DTYPE e_26 = e6 - e2 ;

  r0.re = d0246 + d1357;

  //r1.re = d0 + ROT_CONST*d1 - ROT_CONST*d3 - d4 - ROT_CONST*d5 + ROT_CONST*d7;
  r1.re = d0_4 + temp1;
  r7.re = r1.re + (e_26 + temp1b);
  r1.re = r1.re - (e_26 + temp1b);

  r2.re = d04 - d26;
  r6.re = r2.re + e_13_57;
  r2.re = r2.re - e_13_57;


  //r3.re = d0 - ROT_CONST*d1 + ROT_CONST*d3 - d4 + ROT_CONST*d5 - ROT_CONST*d7;
  r3.re = d0_4 - temp1;
  r5.re = r3.re + (e2_6 + temp1b);
  r3.re = r3.re - (e2_6 + temp1b);


  r4.re = d0246 - d1357;


  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

  r0.im = e0246 + e1357;

  //r1.im = -ROT_CONST*d1 - d2 - ROT_CONST*d3 + ROT_CONST*d5 + d6 + ROT_CONST*d7;
  r1.im = d_26 + temp2;
  r7.im = -r1.im + ( e0_4 + temp2b);
  r1.im =  r1.im + ( e0_4 + temp2b);

  r2.im = d_57 - d1_3;
  r6.im = -r2.im + e0_24_6;
  r2.im =  r2.im + e0_24_6;

  //r3.im = -ROT_CONST*d1 + d2 - ROT_CONST*d3 + ROT_CONST*d5 - d6 + ROT_CONST*d7;
  r3.im =  temp2 - d_26;
  r5.im = -r3.im + (e0_4 - temp2b);
  r3.im =  r3.im + (e0_4 - temp2b);

  r4.im = e0246 - e1357;
  #endif

  // TWIDDLES
  tw1 = twiddle_ptr[twiddle_index*1];
  tw2 = twiddle_ptr[twiddle_index*2];
  tw3 = twiddle_ptr[twiddle_index*3];
  tw4 = twiddle_ptr[twiddle_index*4];
#ifdef FULL_TWIDDLES
  tw5 = twiddle_ptr[twiddle_index*5];
  tw6 = twiddle_ptr[twiddle_index*6];
  tw7 = twiddle_ptr[twiddle_index*7];
#else
  if(twiddle_index*5 >= MAX_TWIDDLES)
  {
    tw5 = twiddle_ptr[(twiddle_index*5)%(MAX_TWIDDLES)];
    tw5.re = -tw5.re;
    tw5.im = -tw5.im;
  }
  else
  {
    tw5 = twiddle_ptr[twiddle_index*5];
  }
  if(twiddle_index*6 >= MAX_TWIDDLES)
  {
    tw6 = twiddle_ptr[(twiddle_index*6)%(MAX_TWIDDLES)];
    tw6.re = -tw6.re;
    tw6.im = -tw6.im;
  }
  else
  {
    tw6 = twiddle_ptr[twiddle_index*6];
  }
  if(twiddle_index*7 >= MAX_TWIDDLES)
  {
    tw7 = twiddle_ptr[(twiddle_index*7)%(MAX_TWIDDLES)];
    tw7.re = -tw7.re;
    tw7.im = -tw7.im;
  }
  else
  {
    tw7 = twiddle_ptr[twiddle_index*7];
  }
#endif // FULL_TWIDDLES

  input[index]            = r0;
  input[index+distance]   = complex_mul(tw1, r1);
  input[index+2*distance] = complex_mul(tw2, r2);
  input[index+3*distance] = complex_mul(tw3, r3);
  input[index+4*distance] = complex_mul(tw4, r4);
  input[index+5*distance] = complex_mul(tw5, r5);
  input[index+6*distance] = complex_mul(tw6, r6);
  input[index+7*distance] = complex_mul(tw7, r7);
}

void __attribute__ ((always_inline)) process_butterfly_last_radix8 (Complex_type* input, Complex_type* output, int outindex )
{
  int index = 0;

  DTYPE d0, d1, d2, d3, d4, d5, d6, d7;
  DTYPE e0, e1, e2, e3, e4, e5, e6, e7;

  Complex_type r0, r1, r2, r3, r4, r7, r6, r5 ;

#ifdef VECTORIZATION

  VDTYPE v0         = input[index].v;
  VDTYPE v1         = input[index+1].v;
  VDTYPE v2         = input[index+2].v;
  VDTYPE v3         = input[index+3].v;
  VDTYPE v4         = input[index+4].v;
  VDTYPE v5         = input[index+5].v;
  VDTYPE v6         = input[index+6].v;
  VDTYPE v7         = input[index+7].v;

  VDTYPE t0, t1, t2, t3, t4, t5, t6, t7;
  VDTYPE q0, q1, q2, q3, q4, q5, q6, q7;
  VDTYPE s0, s1, s2, s3, s4, s5, s6, s7;
  //
  t0 = v0 + v4;
  t1 = v1 + v5;
  t2 = v2 + v6;
  t3 = v3 + v7;
  t4 = v0 - v4;
  t5 = v1 - v5;
  t6 = v2 - v6;
  t7 = v3 - v7;

  q0 = t0 + t2;
  q1 = t1 + t3;
  q2 = t0 - t2;
  q3 = t1 - t3;
  q4 = t4;
  q5 = t5 + t7;
  q6 = t6;
  q7 = t5 - t7;


  s0 = q0 + q1;
  s1 = q0 - q1;
  s2 = q2 + __builtin_shuffle(q3, (v2s){1,0})*ONE_MONE;
  s3 = q2 + __builtin_shuffle(q3, (v2s){1,0})*MONE_ONE;

  s4 = q4 + ROT_CONST_V * (__builtin_shuffle(q5, (v2s){1,0}))*ONE_MONE;
  s5 = q4 + ROT_CONST_V * (__builtin_shuffle(q5, (v2s){1,0}))*MONE_ONE;
  s6 = ROT_CONST_V * q7 + (__builtin_shuffle(q6, (v2s){1,0}))*ONE_MONE;
  s7 = ROT_CONST_V * q7 + (__builtin_shuffle(q6, (v2s){1,0}))*MONE_ONE;


  r0.v = s0;
  r1.v = s4 + s6;
  r2.v = s2;
  r3.v = s4 - s6;
  r4.v = s1;
  r5.v = s5 - s7;
  r6.v = s3;
  r7.v = s5 + s7;


#else

  d0         = input[index].re;
  d1         = input[index+1].re;
  d2         = input[index+2].re;
  d3         = input[index+3].re;
  d4         = input[index+4].re;
  d5         = input[index+5].re;
  d6         = input[index+6].re;
  d7         = input[index+7].re;

  e0         = input[index].im;
  e1         = input[index+1].im;
  e2         = input[index+2].im;
  e3         = input[index+3].im;
  e4         = input[index+4].im;
  e5         = input[index+5].im;
  e6         = input[index+6].im;
  e7         = input[index+7].im;

  /* twiddles are all 1s*/

  // Basic buttefly rotation
  /*

r0   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i
r1   1.0000 + 0.0000i   0.7071 - 0.7071i   0.0000 - 1.0000i  -0.7071 - 0.7071i  -1.0000 - 0.0000i  -0.7071 + 0.7071i   0.0000 + 1.0000i   0.7071 + 0.7071i
r2   1.0000 + 0.0000i   0.0000 - 1.0000i  -1.0000 - 0.0000i   0.0000 + 1.0000i   1.0000 + 0.0000i   0.0000 - 1.0000i   -1.0000 - 0.0000i  0.0000 + 1.0000i
r3   1.0000 + 0.0000i  -0.7071 - 0.7071i  -0.0000 + 1.0000i   0.7071 - 0.7071i  -1.0000 - 0.0000i   0.7071 + 0.7071i   0.0000 - 1.0000i  -0.7071 + 0.7071i

r4   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i   1.0000 + 0.0000i  -1.0000 - 0.0000i
r5   1.0000 + 0.0000i  -0.7071 + 0.7071i   0.0000 - 1.0000i   0.7071 + 0.7071i  -1.0000 - 0.0000i   0.7071 - 0.7071i  -0.0000 + 1.0000i  -0.7071 - 0.7071i
r6   1.0000 + 0.0000i   0.0000 + 1.0000i  -1.0000 - 0.0000i   0.0000 - 1.0000i   1.0000 + 0.0000i   0.0000 + 1.0000i  -1.0000 - 0.0000i   0.0000 - 1.0000i
r7   1.0000 + 0.0000i   0.7071 + 0.7071i  -0.0000 + 1.0000i  -0.7071 + 0.7071i  -1.0000 - 0.0000i  -0.7071 - 0.7071i   0.0000 - 1.0000i   0.7071 - 0.7071i

  */

  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im

  DTYPE d13 = d1 + d3;
  DTYPE d1_3 = d1 - d3;
  DTYPE e13 = e1 + e3;
  DTYPE e57 = e5 + e7;
  DTYPE e1_3 = e1 - e3;
  DTYPE e_57 = e7 - e5;
  DTYPE d57 = d5 + d7;
  DTYPE d_57 = d7 - d5;
  DTYPE temp1 = ROT_CONST*(d1_3+d_57);
  DTYPE temp1b = ROT_CONST*(e57 - e13);
  DTYPE temp2 = ROT_CONST*(d57-d13);
  DTYPE temp2b = ROT_CONST*(e1_3 + e_57);
  DTYPE d04   = d0 + d4;
  DTYPE d0_4  = d0 - d4;
  DTYPE d26  = d2 + d6;
  DTYPE d_26  = d6 - d2;
  DTYPE d0246 = d04 + d26;
  DTYPE d1357 = d13 + d57;
  DTYPE e0246 = e0 + e2 + e4 + e6;
  DTYPE e0_4  = e0 - e4;
  DTYPE e0_24_6 = e0 - e2 + e4 - e6;
  DTYPE e1357 = e13 + e57;
  DTYPE e_13_57 = e_57 - e1_3;
  DTYPE e2_6 = e2 - e6 ;
  DTYPE e_26 = e6 - e2 ;

  r0.re = d0246 + d1357;

  //r1.re = d0 + ROT_CONST*d1 - ROT_CONST*d3 - d4 - ROT_CONST*d5 + ROT_CONST*d7;
  r1.re = d0_4 + temp1;
  r7.re = r1.re + (e_26 + temp1b);
  r1.re = r1.re - (e_26 + temp1b);

  r2.re = d04 - d26;
  r6.re = r2.re + e_13_57;
  r2.re = r2.re - e_13_57;


  //r3.re = d0 - ROT_CONST*d1 + ROT_CONST*d3 - d4 + ROT_CONST*d5 - ROT_CONST*d7;
  r3.re = d0_4 - temp1;
  r5.re = r3.re + (e2_6 + temp1b);
  r3.re = r3.re - (e2_6 + temp1b);


  r4.re = d0246 - d1357;


  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

  r0.im = e0246 + e1357;

  //r1.im = -ROT_CONST*d1 - d2 - ROT_CONST*d3 + ROT_CONST*d5 + d6 + ROT_CONST*d7;
  r1.im = d_26 + temp2;
  r7.im = -r1.im + ( e0_4 + temp2b);
  r1.im =  r1.im + ( e0_4 + temp2b);

  r2.im = d_57 - d1_3;
  r6.im = -r2.im + e0_24_6;
  r2.im =  r2.im + e0_24_6;

  //r3.im = -ROT_CONST*d1 + d2 - ROT_CONST*d3 + ROT_CONST*d5 - d6 + ROT_CONST*d7;
  r3.im =  temp2 - d_26;
  r5.im = -r3.im + (e0_4 - temp2b);
  r3.im =  r3.im + (e0_4 - temp2b);

  r4.im = e0246 - e1357;
#endif

  /* In the Last step, twiddle factors are all 1 */
#ifndef SORT_OUTPUT
#ifdef BITREV_LUT
#if defined(MIXED_RADIX) && !defined(RADIX_8)
  unsigned int index12 = *((unsigned int *)(&bit_rev_2_8_LUT[outindex]));
  unsigned int index34 = *((unsigned int *)(&bit_rev_2_8_LUT[outindex+2]));
  unsigned int index56 = *((unsigned int *)(&bit_rev_2_8_LUT[outindex+4]));
  unsigned int index78 = *((unsigned int *)(&bit_rev_2_8_LUT[outindex+6]));
#else // !MIXED_RADIX
  unsigned int index12 = *((unsigned int *)(&bit_rev_radix8_LUT[outindex]));
  unsigned int index34 = *((unsigned int *)(&bit_rev_radix8_LUT[outindex+2]));
  unsigned int index56 = *((unsigned int *)(&bit_rev_radix8_LUT[outindex+4]));
  unsigned int index78 = *((unsigned int *)(&bit_rev_radix8_LUT[outindex+6]));
#endif // MIXED_RADIX
  unsigned int index1  = index12 & 0x0000FFFF;
  unsigned int index2  = index12 >> 16;
  unsigned int index3  = index34 & 0x0000FFFF;
  unsigned int index4  = index34 >> 16;
  unsigned int index5  = index56 & 0x0000FFFF;
  unsigned int index6  = index56 >> 16;
  unsigned int index7  = index78 & 0x0000FFFF;
  unsigned int index8  = index78 >> 16;
  output[index1] = r0;
  output[index2] = r1;
  output[index3] = r2;
  output[index4] = r3;
  output[index5] = r4;
  output[index6] = r5;
  output[index7] = r6;
  output[index8] = r7;
#else // !BITREV_LUT
#if defined(MIXED_RADIX) && !defined(RADIX_8)
  output[bit_rev_2_8(outindex    )] = r0;
  output[bit_rev_2_8(outindex+1  )] = r1;
  output[bit_rev_2_8(outindex+2*1)] = r2;
  output[bit_rev_2_8(outindex+3*1)] = r3;
  output[bit_rev_2_8(outindex+4*1)] = r4;
  output[bit_rev_2_8(outindex+5*1)] = r5;
  output[bit_rev_2_8(outindex+6*1)] = r6;
  output[bit_rev_2_8(outindex+7*1)] = r7;
#else // !MIXED_RADIX

  output[bit_rev_radix8(outindex    )] = r0;
  output[bit_rev_radix8(outindex+1  )] = r1;
  output[bit_rev_radix8(outindex+2*1)] = r2;
  output[bit_rev_radix8(outindex+3*1)] = r3;
  output[bit_rev_radix8(outindex+4*1)] = r4;
  output[bit_rev_radix8(outindex+5*1)] = r5;
  output[bit_rev_radix8(outindex+6*1)] = r6;
  output[bit_rev_radix8(outindex+7*1)] = r7;
#endif // MIXED_RADIX
#endif // BITREV_LUT
#else // SORT_OUTPUT
   output[outindex    ] = r0;
   output[outindex+1  ] = r1;
   output[outindex+2*1] = r2;
   output[outindex+3*1] = r3;
   output[outindex+4*1] = r4;
   output[outindex+5*1] = r5;
   output[outindex+6*1] = r6;
   output[outindex+7*1] = r7;
#endif // !SORT_OUTPUT

}


#if !defined(SORT_OUTPUT)
void __attribute__ ((always_inline)) fft_radix8 (Complex_type * Inp_signal, Complex_type * Out_signal, int output_index_base)
#else
void __attribute__ ((always_inline)) fft_radix8 (Complex_type * Inp_signal, Complex_type * Out_signal)
#endif
{
  int k,j,stage, step, d, index;
  Complex_type * _in;
  Complex_type * _out;
  Complex_type  temp;

  int dist = FFT_LEN_RADIX8 >> 3;
  int nbutterfly = FFT_LEN_RADIX8 >> 3;
  int butt = 8;
  Complex_type * _in_ptr;
  Complex_type * _out_ptr;
  Complex_type * _tw_ptr;
  _in  = &(Inp_signal[0]);
  _out = &(Out_signal[0]);

  // FIRST STAGE
  stage = 1;

  _in_ptr = _in;

  #ifdef FULL_TWIDDLES
  _tw_ptr = twiddle_factors8;
  #else
  _tw_ptr = twiddle_factors;
  #endif

  for(j=0;j<nbutterfly;j++)
  {
    #if defined(MIXED_RADIX) && !defined(RADIX_8)
    process_butterfly_radix8(_in_ptr, j, 0, dist, _tw_ptr);
    #else
    process_butterfly_real_radix8(_in_ptr, j, dist, _tw_ptr);
    #endif
    _in_ptr++;
  } //j

  stage = stage + 1;
  dist  = dist >> 3;

  // SECOND STAGE
  while(dist > 1)
  {
    step = dist << 3;
    for(j=0;j<butt;j++)
    {
      _in_ptr  = _in;
      for(d = 0; d < dist; d++)
      {
        process_butterfly_radix8(_in_ptr,  d*butt, j*step, dist, _tw_ptr);
        _in_ptr++;
      } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 3;
    butt = butt << 3;
  }

  _in_ptr  = _in;

  // LAST STAGE
#if defined(MIXED_RADIX) && !defined(RADIX_8) && !defined(SORT_OUTPUT)
  index = output_index_base;
#else
  index = 0;
#endif
  for(j=0;j<FFT_LEN_RADIX8/8; j++) //FFT_LEN_RADIX8/8 o FFT_LEN_RADIX8 >> 3
  {
    process_butterfly_last_radix8(_in_ptr, _out, index);
    _in_ptr +=8;
    index   +=8;
  } //j

  // ORDER VALUES
#if defined(SORT_OUTPUT) && !defined(MIXED_RADIX)
  for(j=0; j<FFT_LEN_RADIX8; j+=8)
  {
#ifdef BITREV_LUT
    unsigned int index12 = *((unsigned int *)(&bit_rev_radix8_LUT[j]));
    unsigned int index34 = *((unsigned int *)(&bit_rev_radix8_LUT[j+2]));
    unsigned int index56 = *((unsigned int *)(&bit_rev_radix8_LUT[j+4]));
    unsigned int index78 = *((unsigned int *)(&bit_rev_radix8_LUT[j+6]));
    unsigned int index1  = index12 & 0x0000FFFF;
    unsigned int index2  = index12 >> 16;
    unsigned int index3  = index34 & 0x0000FFFF;
    unsigned int index4  = index34 >> 16;
    unsigned int index5  = index56 & 0x0000FFFF;
    unsigned int index6  = index56 >> 16;
    unsigned int index7  = index78 & 0x0000FFFF;
    unsigned int index8  = index78 >> 16;
#else
    int index1 = bit_rev_radix8(j);
    int index2 = bit_rev_radix8(j+1);
    int index3 = bit_rev_radix8(j+2);
    int index4 = bit_rev_radix8(j+3);
    int index5 = bit_rev_radix8(j+4);
    int index6 = bit_rev_radix8(j+5);
    int index7 = bit_rev_radix8(j+6);
    int index8 = bit_rev_radix8(j+7);
#endif // BITREV_LUT

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
    if(index5 > j+4)
    {
      temp         = _out[j+4];
      _out[j+4]    = _out[index5];
      _out[index5] = temp;
    }
    if(index6 > j+5)
    {
      temp         = _out[j+5];
      _out[j+5]    = _out[index6];
      _out[index6] = temp;
    }
    if(index7 > j+6)
    {
      temp         = _out[j+6];
      _out[j+6]    = _out[index7];
      _out[index7] = temp;
    }
    if(index8 > j+7)
    {
      temp         = _out[j+7];
      _out[j+7]    = _out[index8];
      _out[index8] = temp;
    }
  }
#endif
}

// PARALLEL
#if !defined(SORT_OUTPUT)
void __attribute__ ((always_inline)) par_fft_radix8 (Complex_type * Inp_signal, Complex_type * Out_signal, int output_index_base)
#else
void __attribute__ ((always_inline)) par_fft_radix8 (Complex_type * Inp_signal, Complex_type * Out_signal)
#endif
{
  int k,j,stage, step, d, index;
  Complex_type * _in;
  Complex_type * _out;
  Complex_type  temp;
  int dist = FFT_LEN_RADIX8 >> 3;
  int nbutterfly = FFT_LEN_RADIX8 >> 3;
  int butt = 8;
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
#ifdef FULL_TWIDDLES
  _tw_ptr = twiddle_factors8;
#else
  _tw_ptr = twiddle_factors;
#endif


  stage = 1;

  for(j=0; j < nbutterfly/NUM_CORES; j++)
  {
#if defined(MIXED_RADIX) && !defined(RADIX_8)
    process_butterfly_radix8(_in_ptr, j*NUM_CORES+core_id, 0, dist, _tw_ptr);
#else
    process_butterfly_real_radix8(_in_ptr,  j*NUM_CORES+core_id, dist, _tw_ptr);
#endif
    _in_ptr+=NUM_CORES;
  } //j

  stage = stage + 1;
  dist  = dist >> 3;

  // STAGES 2 -> n-1
  while(dist >= NUM_CORES)
  {
    pi_cl_team_barrier();
    step = dist << 3;
    for(j=0;j<butt;j++)
    {
      _in_ptr = &_in[core_id];
      for(d = 0; d < dist/NUM_CORES; d++)
      {
        process_butterfly_radix8(_in_ptr, (d*NUM_CORES+core_id)*butt, j*step, dist, _tw_ptr);
        _in_ptr+=NUM_CORES;
      } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 3;
    butt = butt << 3;
  }
  while(dist > 1)
  {
    pi_cl_team_barrier();
    step = dist << 3;
    for(j=0;j<butt/NUM_CORES;j++)
    {
        _in_ptr = _in;
        for(d = 0; d < dist; d++)
        {
          process_butterfly_radix8(_in_ptr, d*butt, (j*NUM_CORES+core_id)*step, dist, _tw_ptr);
          _tw_ptr+=8*NUM_CORES;
          _in_ptr+=NUM_CORES;
        } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 3;
    butt = butt << 3;
  }

  pi_cl_team_barrier();

  // LAST STAGE
  _in_ptr  = &_in[8*core_id];
#if defined(MIXED_RADIX) && !defined(RADIX_8) && !defined(SORT_OUTPUT)
  index = output_index_base + 8*core_id;
#else
  index = 8*core_id;
#endif
  //N must be at least 64 for multicore
  for(j=0;j<FFT_LEN_RADIX8/(8*NUM_CORES);j++)
  {
    process_butterfly_last_radix8(_in_ptr,_out,index);
    _in_ptr += 8*NUM_CORES;
    index   += 8*NUM_CORES;
  } //j

  // ORDER VALUES
#if defined(SORT_OUTPUT) && !defined(MIXED_RADIX)
  pi_cl_team_barrier();
  for(j = 8*core_id; j < FFT_LEN_RADIX8; j+=NUM_CORES*8)
  {
#ifdef BITREV_LUT
    unsigned int index12 = *((unsigned int *)(&bit_rev_radix8_LUT[j]));
    unsigned int index34 = *((unsigned int *)(&bit_rev_radix8_LUT[j+2]));
    unsigned int index56 = *((unsigned int *)(&bit_rev_radix8_LUT[j+4]));
    unsigned int index78 = *((unsigned int *)(&bit_rev_radix8_LUT[j+6]));
    unsigned int index1  = index12 & 0x0000FFFF;
    unsigned int index2  = index12 >> 16;
    unsigned int index3  = index34 & 0x0000FFFF;
    unsigned int index4  = index34 >> 16;
    unsigned int index5  = index56 & 0x0000FFFF;
    unsigned int index6  = index56 >> 16;
    unsigned int index7  = index78 & 0x0000FFFF;
    unsigned int index8  = index78 >> 16;
#else
    int index1 = bit_rev_radix8(j);
    int index2 = bit_rev_radix8(j+1);
    int index3 = bit_rev_radix8(j+2);
    int index4 = bit_rev_radix8(j+3);
    int index5 = bit_rev_radix8(j+4);
    int index6 = bit_rev_radix8(j+5);
    int index7 = bit_rev_radix8(j+6);
    int index8 = bit_rev_radix8(j+7);
#endif // BITREV_LUT

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
    if(index5 > j+4)
    {
      temp         = _out[j+4];
      _out[j+4]    = _out[index5];
      _out[index5] = temp;
    }
    if(index6 > j+5)
    {
      temp         = _out[j+5];
      _out[j+5]    = _out[index6];
      _out[index6] = temp;
    }
    if(index7 > j+6)
    {
      temp         = _out[j+6];
      _out[j+6]    = _out[index7];
      _out[index7] = temp;
    }
    if(index8 > j+7)
    {
      temp         = _out[j+7];
      _out[j+7]    = _out[index8];
      _out[index8] = temp;
    }
  }
  pi_cl_team_barrier();
  #endif
}
