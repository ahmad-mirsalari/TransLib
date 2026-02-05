#include "pmsis.h"

#include <stdio.h>

#include "fft_mixed.h"

#include "../fft_radix8/fft_radix8.c"

#ifdef STAGE_2_2_8
#define MAX_C  4
#define ELEMS  6
#else
#define MAX_C  5
#define ELEMS  12
#endif

void process_butterfly_real_radix2 (Complex_type* input, int twiddle_index, int distance, Complex_type* twiddle_ptr);
void process_butterfly_radix2 (Complex_type* input, int twiddle_index, int index, int distance, Complex_type* twiddle_ptr);
int bit_rev_radix2(int index_torevert);
#if !defined(SORT_OUTPUT)
void fft_radix8 (Complex_type * Inp_signal, Complex_type * Out_signal, int output_index_base);
#else
void fft_radix8 (Complex_type * Inp_signal, Complex_type * Out_signal);
#endif
extern Complex_type twiddle_factors[];
extern short bit_rev_radix2_LUT[];

extern int bitrev_perm_count[];
extern int bitrev_perm_steps[];

// SEQUENTIAL VERSION
void __attribute__ ((noinline)) fft_mixed_2_8 (Complex_type * Inp_signal, Complex_type * Out_signal)
{
  int j, d;
  int step;
  int dist = FFT_LEN_RADIX2 >> 1;
  int nbutterfly = FFT_LEN_RADIX2 >> 1;
  int butt = 2;
  Complex_type * _in_ptr;
  Complex_type temp;

  #if !defined (RADIX_8)
  // STAGE 1 (input is real, stage=1)
  _in_ptr = &(Inp_signal[0]);
  for(j=0;j<nbutterfly;j++)
  {
    process_butterfly_real_radix2(_in_ptr, j, dist, twiddle_factors);
    _in_ptr++;

  }
  #endif


#ifdef STAGE_2_2_8
  // STAGE 2
  dist  = dist >> 1;
  step = dist << 1;
  for(j=0;j<butt;j++)
  {
    _in_ptr = &(Inp_signal[0]);
    for(d = 0; d < dist; d++)
    {
      process_butterfly_radix2(_in_ptr, d*butt, j*step, dist, twiddle_factors);
      _in_ptr++;
    } //d
  } //j

  // STAGE 3 -> n (radix-8)
  for(j=0; j<FFT_LEN; j+=FFT_LEN/4)
  {
    #if !defined(SORT_OUTPUT)
    fft_radix8(Inp_signal+j, Out_signal, j);
    #else
    fft_radix8(Inp_signal+j, Out_signal+j);
    #endif
   }
#else // !STAGE_2_2_8
  #if defined (RADIX_8)  
    #if !defined(SORT_OUTPUT)
    fft_radix8(Inp_signal, Out_signal, 0);
    #else
    fft_radix8(Inp_signal, Out_signal);
    #endif
  #else

  // STAGE 2 -> n (radix-8)
  for(j=0; j<FFT_LEN; j+=FFT_LEN/2)
    #if !defined(SORT_OUTPUT)
    fft_radix8(Inp_signal+j, Out_signal, j);
    #else
    fft_radix8(Inp_signal+j, Out_signal+j);
    #endif
  #endif
#endif // STAGE_2_2_8



  // ORDER VALUES
#ifdef SORT_OUTPUT
#ifdef BITREV_LUT
  unsigned int index[ELEMS];
  unsigned int val, base;
  base = 0;
  for(int c = 0; c < MAX_C; c++)
  {
    for(int cnt = 0; cnt < bitrev_perm_count[c]; cnt++)
    {
      // Read a sequence of permutation frm the LUT
      for(int s = 0; s < bitrev_perm_steps[c]; s+=2)
      {
        val = *((unsigned int *)(&bit_rev_2_8_LUT[base+bitrev_perm_steps[c]*cnt+s]));
        index[s] = val & 0x0000FFFF;
        index[s+1] = val >> 16;
      }
      // Apply permutations
      temp = Out_signal[index[bitrev_perm_steps[c]-1]];
      for(int k=bitrev_perm_steps[c]-1; k>0; k--)
      {
        Out_signal[index[k]] = Out_signal[index[k-1]];
      }
      Out_signal[index[0]] = temp;
    }
    base += bitrev_perm_steps[c]*bitrev_perm_count[c];
  }
#else // !BITREV_LUT
  for(j=0; j<FFT_LEN; j++)
  {
    int k, skip;
    unsigned int index[ELEMS+1];
    skip = 0;
    index[0] = j;
    // Find a closed set of permutations
    for(k=0; k<ELEMS; k++)
    {
      index[k+1] = bit_rev_2_8(index[k]);
      if(index[k+1] == j) { break; }
      if(index[k+1] > j) { skip=1; break; }
    }
    // Apply permutations
    if(!skip)
    {
      temp = Out_signal[index[k]];
      for(int l=k; l>0; l--)
      {
        Out_signal[index[l]] = Out_signal[index[l-1]];
      }
      Out_signal[index[0]] = temp;
    }
  }
#endif // BITREV_LUT
#endif // SORT_OUTPUT
}

// PARALLEL VERSION
void __attribute__ ((noinline)) par_fft_mixed_2_8 (Complex_type * Inp_signal, Complex_type * Out_signal)
{
  int j, d;
  int step;
  int dist = FFT_LEN_RADIX2 >> 1;
  int nbutterfly = FFT_LEN_RADIX2 >> 1;
  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif
  int butt = 2;
  Complex_type * _in_ptr;
  Complex_type temp;

  #if !defined (RADIX_8)
  // STAGE 1 (input is real, stage=1)
  _in_ptr = &(Inp_signal[core_id]);
  for(j=0;j<nbutterfly/NUM_CORES;j++)
  {
     process_butterfly_real_radix2(_in_ptr, j*NUM_CORES+core_id, dist, twiddle_factors);
    _in_ptr+=NUM_CORES;
  }
  #endif

  pi_cl_team_barrier();

#ifdef STAGE_2_2_8
  // STAGE 2
  dist  = dist >> 1;
  step = dist << 1;
  for(j=0;j<butt;j++)
  {
    _in_ptr = &(Inp_signal[core_id]);
    for(d = 0; d < dist/NUM_CORES; d++)
    {
      process_butterfly_radix2(_in_ptr, (d*NUM_CORES+core_id)*butt, j*step,dist, twiddle_factors);
      _in_ptr+=NUM_CORES;
     } //d
  } //j

  pi_cl_team_barrier();

  // STAGE 3 -> n (radix-8)
  for(j=0; j<FFT_LEN; j+=FFT_LEN/4)
  {
#if !defined(SORT_OUTPUT)
    par_fft_radix8(Inp_signal+j, Out_signal, j);
#else
    par_fft_radix8(Inp_signal+j, Out_signal+j);
#endif
  }
#else // !STAGE_2_2_8
#if defined (RADIX_8)
#if !defined(SORT_OUTPUT)
    par_fft_radix8(Inp_signal, Out_signal, 0);
#else
    par_fft_radix8(Inp_signal, Out_signal);
#endif
#else   
  // STAGE 2 -> n (radix-8)
  for(j=0; j<FFT_LEN; j+=FFT_LEN/2)
#if !defined(SORT_OUTPUT)
    par_fft_radix8(Inp_signal+j, Out_signal, j);
#else
    par_fft_radix8(Inp_signal+j, Out_signal+j);
#endif
#endif
#endif // STAGE_2_2_8

  // ORDER VALUES
#ifdef SORT_OUTPUT
#ifdef BITREV_LUT
  unsigned int index[ELEMS];
  unsigned int val, base;
  base = 0;
  for(int c = 0; c < MAX_C; c++)
  {
    for(int cnt = core_id; cnt < bitrev_perm_count[c]; cnt+=NUM_CORES)
    {
      // Read a sequence of permutation frm the LUT
      for(int s = 0; s < bitrev_perm_steps[c]; s+=2)
      {
        val = *((unsigned int *)(&bit_rev_2_8_LUT[base+bitrev_perm_steps[c]*cnt+s]));
        index[s] = val & 0x0000FFFF;
        index[s+1] = val >> 16;
      }
      // Apply permutations
      temp = Out_signal[index[bitrev_perm_steps[c]-1]];
      for(int k=bitrev_perm_steps[c]-1; k>0; k--)
      {
        Out_signal[index[k]] = Out_signal[index[k-1]];
      }
      Out_signal[index[0]] = temp;
    }
    base += bitrev_perm_steps[c]*bitrev_perm_count[c];
  }
#else // !BITREV_LUT
  for(j=core_id; j<FFT_LEN; j+=NUM_CORES)
  {
    int k, skip;
    unsigned int index[ELEMS+1];
    skip = 0;
    index[0] = j;
    // Find a closed set of permutations
    for(k=0; k<ELEMS; k++)
    {
      index[k+1] = bit_rev_2_8(index[k]);
      if(index[k+1] == j) { break; }
      if(index[k+1] > j) { skip=1; break; }
    }
    // Apply permutations
    if(!skip)
    {
      temp = Out_signal[index[k]];
      for(int l=k; l>0; l--)
      {
        Out_signal[index[l]] = Out_signal[index[l-1]];
      }
      Out_signal[index[0]] = temp;
    }
  }
#endif // BITREV_LUT
  pi_cl_team_barrier();
#endif // SORT_OUTPUT
}
