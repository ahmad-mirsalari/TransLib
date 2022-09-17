#include <stdio.h>
#include "pulp.h"
#include "fft.h"

#ifdef MIXED_RADIX
extern Complex_type twiddle_factors[];
#else
#include "twiddle_factor.h"
#endif

#include "print_float.h"

#ifdef BITREV_LUT
#include "bit_reverse.h"
#endif

#ifdef VECTORIZATION
RT_L1_DATA static VDTYPE ONE_MONE = (VDTYPE) {1.0f, -1.0f};
RT_L1_DATA static VDTYPE MONE_ONE = (VDTYPE) {-1.0f, 1.0f};
#endif


#ifdef VECTORIZATION
static Complex_type __attribute__ ((always_inline)) complex_mul (Complex_type A, Complex_type B)
{
  Complex_type c_tmp;

  // c_tmp.re = A.re * B.re - A.im * B.im;
  // c_tmp.im = A.re * B.im + A.im * B.re;
  // return c_tmp;

  VDTYPE P0, P1, P2, P3;

  P0 = A.v * B.v;
  B.v = __builtin_pulp_shuffleh(B.v, (v2s){1,0});
  P1 = A.v * B.v;
  P2 =__builtin_pulp_shuffle2h(P0, P1, (v2s) {0,2});
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

#ifdef MIXED_RADIX
int __attribute__ ((always_inline)) bit_rev_2_4(int value) //digit rev between 2 bit
{
  int i;
  unsigned int new_value = 0;
  for (i = 0; i < LOG2_FFT_LEN_RADIX4/2; i++)
  {
    new_value <<= 2;
    new_value |= (value & 0x3);
    value >>= 2;
  }
  new_value = (new_value << 1) | value;
  return new_value;
}
#endif

int __attribute__ ((always_inline)) bit_rev_radix4(int value) //digit reverse 2 bit blocks
{
  int i;
  unsigned int new_value = 0;
  for (i = 0; i < LOG2_FFT_LEN_RADIX4/2; i++)
  {
      new_value <<= 2;
      new_value |= (value & 0x3);
      value >>= 2;
  }
  return new_value;
}


void __attribute__ ((always_inline)) process_butterfly_real_radix4 (Complex_type* input, int twiddle_index, int distance, Complex_type* twiddle_ptr)
{
  DTYPE d0, d1, d2, d3;

  Complex_type r0, r1, r2, r3;

  int index = 0;

  #ifdef MIXED_RADIX
  twiddle_index = twiddle_index * 2;
  #define MAX_TWIDDLES (FFT_LEN_RADIX4)
  #else
  #define MAX_TWIDDLES (FFT_LEN_RADIX4/2)
  #endif


  #ifdef VECTORIZATION
  VDTYPE v0 = input[index].v;
  VDTYPE v1 = input[index+distance].v;
  VDTYPE v2 = input[index+2*distance].v;
  VDTYPE v3 = input[index+3*distance].v;

  VDTYPE v1_s, v3_s;

  r0.v = v0 + v1 + v2 + v3;
  v1_s = __builtin_shuffle(v1, (v2s){1,0});
  v3_s = __builtin_shuffle(v3, (v2s){1,0});
  r1.v = v0 - v1_s - v2 + v3_s;
  r2.v = v0 - v1 + v2 - v3;
  r3.v = r1.v * ONE_MONE;

  #else
  //int twiddle_index = 0; //now twiddles are set to a nonzero value
  d0 = input[index].re;
  d1 = input[index+distance].re;
  d2 = input[index+2*distance].re;
  d3 = input[index+3*distance].re;


  // Basic buttefly rotation
  /*

r0   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i
r1   1.0000 + 0.0000i   0.7071 - 0.7071i   0.0000 - 1.0000i  -0.7071 - 0.7071i  -1.0000 - 0.0000i  -0.7071 + 0.7071i   0.0000 + 1.0000i   0.7071 + 0.7071i
r2   1.0000 + 0.0000i   0.0000 - 1.0000i  -1.0000 - 0.0000i   0.0000 + 1.0000i   1.0000 + 0.0000i   0.0000 - 1.0000i   -1.0000 - 0.0000i  0.0000 + 1.0000i
r3   1.0000 + 0.0000i  -0.7071 - 0.7071i   0.0000 + 1.0000i   0.7071 - 0.7071i  -1.0000 - 0.0000i   0.7071 + 0.7071i   0.0000 - 1.0000i  -0.7071 + 0.7071i

  */

  //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im, since c1 is real = c1.re*c2.re
  r0.re = d0 + d1 + d2 + d3;
  r1.re = d0 - d2;
  r2.re = d0 - d1 + d2 - d3;
  r3.re = r1.re;

  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re, since c1 is real = c1.re*c2.im
  r0.im = 0.0f;
  r1.im = -d1 + d3;
  r2.im = 0.0f;
  r3.im = d1 - d3;
  #endif

  Complex_type tw1, tw2, tw3;

  tw1 = twiddle_ptr[twiddle_index];
  tw2 = twiddle_ptr[twiddle_index*2];
  input[index]            = r0;

  register int idx3 = twiddle_index*3 - MAX_TWIDDLES;
  if(idx3 >= 3)
  {
    tw3 = twiddle_ptr[idx3];
    #ifdef VECTORIZATION
    tw3.v = -tw3.v;
    #else
    tw3.re = -tw3.re ;
    tw3.im = -tw3.im ;
    #endif
  }
  else
  {
    tw3 = twiddle_ptr[twiddle_index*3];
  }

  input[index+distance]   = complex_mul(tw1, r1);
  input[index+2*distance] = complex_mul_real(r2.re,tw2);
  input[index+3*distance] = complex_mul(tw3, r3);
}

void __attribute__ ((always_inline)) process_butterfly_radix4 (Complex_type* input, int twiddle_index, int index, int distance, Complex_type* twiddle_ptr)
{
     Complex_type r0, r1, r2, r3;

     Complex_type tw1, tw2, tw3;

#ifdef VECTORIZATION
     VDTYPE v0 = input[index].v;
     VDTYPE v1 = input[index+distance].v;
     VDTYPE v2 = input[index+2*distance].v;
     VDTYPE v3 = input[index+3*distance].v;

     VDTYPE v1_s;
     VDTYPE v3_s;

     r0.v = v0 + v1 + v2 + v3;
     r2.v = v0 - v1 + v2 - v3;
     v1_s = __builtin_shuffle(v1, (v2s){1,0}) * ONE_MONE;
     v3_s = __builtin_shuffle(v3, (v2s){1,0}) * MONE_ONE;
     r1.v = v0 - v2 +  v1_s + v3_s;
     r3.v = v0 - v2 - (v1_s + v3_s);
 #else

     DTYPE d0, d1, d2, d3;
     DTYPE e0, e1, e2, e3;


     //int twiddle_index = 0; //lo levo perché sto facendo il mix con la radix 2 quindi ora la radix 4 è in mezzo quindi all'inizio della radix4 i twiddle adesso ci sono
     d0         = input[index].re;
     d1         = input[index+distance].re;
     d2         = input[index+2*distance].re;
     d3         = input[index+3*distance].re;
     // printf("### [%d] %d %d %d %d / 0 %d %d %d  / %x %x\n", get_core_id(), index, index+distance, index+2*distance,index+3*distance, twiddle_index, twiddle_index*2, twiddle_index*3, &input, &twiddle_index);

     e0         = input[index].im;
     e1         = input[index+distance].im;
     e2         = input[index+2*distance].im;
     e3         = input[index+3*distance].im;


     //Complex_type tw0 = twiddle_ptr[0];

     // Basic buttefly rotation
     /*

     r0   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i
     r1   1.0000 + 0.0000i   0.7071 - 0.7071i   0.0000 - 1.0000i  -0.7071 - 0.7071i  -1.0000 - 0.0000i  -0.7071 + 0.7071i   0.0000 + 1.0000i   0.7071 + 0.7071i
     r2   1.0000 + 0.0000i   0.0000 - 1.0000i  -1.0000 - 0.0000i   0.0000 + 1.0000i   1.0000 + 0.0000i   0.0000 - 1.0000i   -1.0000 - 0.0000i  0.0000 + 1.0000i
     r3   1.0000 + 0.0000i  -0.7071 - 0.7071i  -0.0000 + 1.0000i   0.7071 - 0.7071i  -1.0000 - 0.0000i   0.7071 + 0.7071i   0.0000 - 1.0000i  -0.7071 + 0.7071i
     */

     //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im

     r0.re = d0 + d1 + d2 + d3;
     r1.re = d0 - d2 - ( e3 - e1 );
     r2.re = d0 - d1 + d2 - d3;
     r3.re = d0 - d2 - ( e1 - e3 );

     //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

     r0.im = e0 + e1 + e2 + e3;
     r1.im = -d1 + d3 + e0 - e2;
     r2.im = e0 - e1 + e2 - e3;
     r3.im = d1 - d3 + e0 - e2;
  #endif

     #ifdef MIXED_RADIX
     twiddle_index = twiddle_index * 2;
     #define MAX_TWIDDLES (FFT_LEN_RADIX4)
     #else
     #define MAX_TWIDDLES (FFT_LEN_RADIX4/2)
     #endif

     tw1 = twiddle_ptr[twiddle_index];
     tw2 = twiddle_ptr[twiddle_index*2];
     input[index]           = r0;

     int idx3  = twiddle_index*3 - MAX_TWIDDLES;
     if(idx3 >= 0)
       {
         tw3 = twiddle_ptr[idx3];
         #ifdef VECTORIZATION
         tw3.v = -tw3.v;
         #else
         tw3.re = -tw3.re;
         tw3.im = -tw3.im;
         #endif
       }
       else
       {
         tw3 = twiddle_ptr[twiddle_index*3];
       }

       //input[index]           = r0; //complex_mul(tw0, r0)
       input[index+distance]   = complex_mul(tw1, r1);
       input[index+2*distance] = complex_mul(tw2, r2);
       input[index+3*distance] = complex_mul(tw3, r3);
     }




void __attribute__ ((always_inline)) process_butterfly_last_radix4 (Complex_type* input, Complex_type* output, int outindex )
{
  int index = 0;

  Complex_type r0, r1, r2, r3;

#ifdef VECTORIZATION
  VDTYPE v0         = input[index].v;
  VDTYPE v1         = input[index+1].v;
  VDTYPE v2         = input[index+2].v;
  VDTYPE v3         = input[index+3].v;

  VDTYPE v1_s;
  VDTYPE v3_s;

  r0.v = v0 + v1 + v2 + v3;
  r2.v = v0 - v1 + v2 - v3;
  v1_s = __builtin_shuffle(v1, (v2s){1,0}) * ONE_MONE;
  v3_s = __builtin_shuffle(v3, (v2s){1,0}) * MONE_ONE;
  r1.v = v0 - v2 +  v1_s + v3_s;
  r3.v = v0 - v2 - (v1_s + v3_s);

#else
  DTYPE d0         = input[index].re;
  DTYPE d1         = input[index+1].re;
  DTYPE d2         = input[index+2].re;
  DTYPE d3         = input[index+3].re;


  DTYPE e0         = input[index].im;
  DTYPE e1         = input[index+1].im;
  DTYPE e2         = input[index+2].im;
  DTYPE e3         = input[index+3].im;


  /* twiddles are all 1s*/

  // Basic buttefly rotation
  /*

r0   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i   1.0000 + 0.0000i
r1   1.0000 + 0.0000i   0.7071 - 0.7071i   0.0000 - 1.0000i  -0.7071 - 0.7071i  -1.0000 - 0.0000i  -0.7071 + 0.7071i   0.0000 + 1.0000i   0.7071 + 0.7071i
r2   1.0000 + 0.0000i   0.0000 - 1.0000i  -1.0000 - 0.0000i   0.0000 + 1.0000i   1.0000 + 0.0000i   0.0000 - 1.0000i   -1.0000 - 0.0000i  0.0000 + 1.0000i
r3   1.0000 + 0.0000i  -0.7071 - 0.7071i  -0.0000 + 1.0000i   0.7071 - 0.7071i  -1.0000 - 0.0000i   0.7071 + 0.7071i   0.0000 - 1.0000i  -0.7071 + 0.7071i

  */

   //Re(c1*c2) = c1.re*c2.re - c1.im*c2.im

  r0.re = d0 + d1 + d2 + d3;
  r1.re = d0 - d2 - ( e3 - e1 );
  r2.re = d0 - d1 + d2 - d3;
  r3.re = d0 - d2 - ( e1 - e3 );

  //Im(c1*c2) = c1.re*c2.im + c1.im*c2.re

  r0.im = e0 + e1 + e2 + e3;
  r1.im = -d1 + d3 + e0 - e2;
  r2.im = e0 - e1 + e2 - e3;
  r3.im = d1 - d3 + e0 - e2;
#endif

#ifndef SORT_OUTPUT
 // /* In the Last step, twiddle factors are all 1 */
#ifdef BITREV_LUT
#ifdef MIXED_RADIX
  unsigned int index12 = *((unsigned int *)(&bit_rev_2_4_LUT[outindex]));
  unsigned int index34 = *((unsigned int *)(&bit_rev_2_4_LUT[outindex+2]));
#else // !MIXED_RADIX
  unsigned int index12 = *((unsigned int *)(&bit_rev_radix4_LUT[outindex]));
  unsigned int index34 = *((unsigned int *)(&bit_rev_radix4_LUT[outindex+2]));
#endif // MIXED_RADIX
  unsigned int index1  = index12 & 0x0000FFFF;
  unsigned int index2  = index12 >> 16;
  unsigned int index3  = index34 & 0x0000FFFF;
  unsigned int index4  = index34 >> 16;
  output[index1] = r0;
  output[index2] = r1;
  output[index3] = r2;
  output[index4] = r3;
#else // !BITREV_LUT
#ifdef MIXED_RADIX
  output[bit_rev_2_4(outindex  )] = r0;
  output[bit_rev_2_4(outindex+1)] = r1;
  output[bit_rev_2_4(outindex+2)] = r2;
  output[bit_rev_2_4(outindex+3)] = r3;
#else // !MIXED_RADIX
  output[bit_rev_radix4(outindex  )] = r0;
  output[bit_rev_radix4(outindex+1)] = r1;
  output[bit_rev_radix4(outindex+2)] = r2;
  output[bit_rev_radix4(outindex+3)] = r3;
#endif // MIXED_RADIX
#endif // BITREV_LUT
#else // SORT_OUTPUT
  output[outindex  ] = r0;
  output[outindex+1] = r1;
  output[outindex+2] = r2;
  output[outindex+3] = r3;
#endif // !SORT_OUTPUT
}


#if defined(MIXED_RADIX) && !defined(SORT_OUTPUT)
void __attribute__ ((noinline)) fft_radix4 (Complex_type * Inp_signal, Complex_type * Out_signal, int output_index_base)
#else
void __attribute__ ((noinline)) fft_radix4 (Complex_type * Inp_signal, Complex_type * Out_signal)
#endif
{
  int k,j,stage, step, d, index;
  Complex_type * _in;
  Complex_type * _out;
  Complex_type  temp;
  int dist = FFT_LEN_RADIX4 >> 2;
  int butt = 4; //number of butterfly in the same group
  int nbutterfly = FFT_LEN_RADIX4 >> 2;
  Complex_type * _in_ptr;
  Complex_type * _out_ptr;
  Complex_type * _tw_ptr;
  _in  = &(Inp_signal[0]);
  _out = &(Out_signal[0]);

  // FIRST STAGE input is real, stage=1
  stage = 1;

  _in_ptr = _in;
  _tw_ptr = twiddle_factors;

  for(j=0;j<nbutterfly;j++)
  {
    #ifdef MIXED_RADIX
    process_butterfly_radix4(_in_ptr, j, 0, dist, _tw_ptr);
    #else
    process_butterfly_real_radix4(_in_ptr, j, dist, _tw_ptr);
    #endif
    _in_ptr++;
  } //j

  stage = stage + 1;
  dist  = dist >> 2;

  // STAGE 2 -> n-1
  while(dist > 1)
  {
    step = dist << 2;
    for(j=0;j<butt;j++)
    {
      _in_ptr  = _in;
      for(d = 0; d < dist; d++)
      {
        process_butterfly_radix4(_in_ptr, d*butt, j*step, dist, _tw_ptr);
        _in_ptr++;
      } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 2;
    butt = butt << 2;
  }

  _in_ptr  = _in;

  // last stage
#if defined(MIXED_RADIX) && !defined(SORT_OUTPUT)
  index=output_index_base;
#else
  index=0;
#endif
  for(j=0;j<FFT_LEN_RADIX4>>2;j++)
  {
    process_butterfly_last_radix4(_in_ptr, _out, index);
    _in_ptr +=4;
    index   +=4;
  } //j

  // ORDER VALUES
#if defined(SORT_OUTPUT) && !defined(MIXED_RADIX)
   for(j=0; j<FFT_LEN_RADIX4; j+=4)
   {
#ifdef BITREV_LUT
     unsigned int index12 = *((unsigned int *)(&bit_rev_radix4_LUT[j]));
     unsigned int index34 = *((unsigned int *)(&bit_rev_radix4_LUT[j+2]));
     unsigned int index1  = index12 & 0x0000FFFF;
     unsigned int index2  = index12 >> 16;
     unsigned int index3  = index34 & 0x0000FFFF;
     unsigned int index4  = index34 >> 16;
#else
     int index1 = bit_rev_radix4(j);
     int index2 = bit_rev_radix4(j+1);
     int index3 = bit_rev_radix4(j+2);
     int index4 = bit_rev_radix4(j+3);
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
#endif
}

//parallel
#if defined(MIXED_RADIX) && !defined(SORT_OUTPUT)
void __attribute__ ((noinline)) par_fft_radix4 (Complex_type * Inp_signal, Complex_type * Out_signal, int output_index_base)
#else
void __attribute__ ((noinline)) par_fft_radix4 (Complex_type * Inp_signal, Complex_type * Out_signal)
#endif
{
  int k, j, stage, step, d, index;
  Complex_type * _in;
  Complex_type * _out;
  Complex_type  temp;
  int dist = FFT_LEN_RADIX4 >> 2;
  int butt = 4;
  int nbutterfly = FFT_LEN_RADIX4 >> 2;
  int core_id = get_core_id();
  Complex_type * _in_ptr;
  Complex_type * _out_ptr;
  Complex_type * _tw_ptr;

  _in  = &(Inp_signal[0]);
  _out = &(Out_signal[0]);

  // FIRST STAGE
  _in_ptr = &_in[core_id];
  _tw_ptr = twiddle_factors;

  stage = 1;
  for(j=0; j < nbutterfly/NUM_CORES; j++)
  {
#ifdef MIXED_RADIX
    process_butterfly_radix4(_in_ptr, j*NUM_CORES+core_id, 0, dist, _tw_ptr);
#else
    process_butterfly_real_radix4(_in_ptr,  j*NUM_CORES+core_id, dist, _tw_ptr);
#endif
    _in_ptr+=NUM_CORES;
  } //j

  stage = stage + 1;
  dist  = dist >> 2;

  // STAGES 2 -> n-1
  while(dist >= NUM_CORES)
  {
    synch_barrier();
    step = dist << 2;
    for(j=0;j<butt;j++)
    {
      _in_ptr = &_in[core_id];
      for(d = 0; d < dist/NUM_CORES; d++)
      {
        process_butterfly_radix4(_in_ptr, (d*NUM_CORES+core_id)*butt, j*step, dist, _tw_ptr);
        _in_ptr+=NUM_CORES;
      } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 2;
    butt = butt << 2;
  }
  while(dist > 1)
  {
    synch_barrier();
    step = dist << 2;
    for(j=0;j<butt/NUM_CORES;j++)
    {
      _in_ptr  = _in;
      for(d = 0; d < dist; d++)
      {
          process_butterfly_radix4(_in_ptr, d*butt, (j*NUM_CORES+core_id)*step, dist, _tw_ptr);
          _in_ptr++;
      } //d
    } //j
    stage = stage + 1;
    dist  = dist >> 2;
    butt = butt << 2;
  }

  synch_barrier();

  // LAST STAGE
  _in_ptr  = &_in[4*core_id];
#if defined(MIXED_RADIX) && !defined(SORT_OUTPUT)
  index = output_index_base + 4*core_id;
#else
  index = 4*core_id;
#endif
  for(j=0; j<FFT_LEN_RADIX4/(4*NUM_CORES); j++)
  {
    process_butterfly_last_radix4(_in_ptr , _out, index);
    _in_ptr +=4*NUM_CORES;
    index   +=4*NUM_CORES;
  }

  synch_barrier();

  // ORDER VALUES
#if defined(SORT_OUTPUT) && !defined(MIXED_RADIX)
  for(j = 4*core_id; j < FFT_LEN_RADIX4; j+=NUM_CORES*4)
  {
#ifdef BITREV_LUT
    unsigned int index12 = *((unsigned int *)(&bit_rev_radix4_LUT[j]));
    unsigned int index34 = *((unsigned int *)(&bit_rev_radix4_LUT[j+2]));
    unsigned int index1  = index12 & 0x0000FFFF;
    unsigned int index2  = index12 >> 16;
    unsigned int index3  = index34 & 0x0000FFFF;
    unsigned int index4  = index34 >> 16;
#else
    int index1 = bit_rev_radix4(j);
    int index2 = bit_rev_radix4(j+1);
    int index3 = bit_rev_radix4(j+2);
    int index4 = bit_rev_radix4(j+3);
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
  synch_barrier();
#endif
}
