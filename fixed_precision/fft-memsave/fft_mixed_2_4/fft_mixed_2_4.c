#include <stdio.h>
#include "pulp.h"
#include "fft_mixed.h"
#include "print_float.h"

#include "../fft_radix4/fft_radix4.c"


extern Complex_type twiddle_factors[];
extern short bit_rev_radix2_LUT[];

void process_butterfly_real_radix2 (Complex_type* input, int twiddle_index, int distance, Complex_type* twiddle_ptr);
int bit_rev_radix2(int index_torevert);
#if !defined(SORT_OUTPUT)
void fft_radix4 (Complex_type * Inp_signal, Complex_type * Out_signal, int output_index_base);
#else
void fft_radix4 (Complex_type * Inp_signal, Complex_type * Out_signal);
#endif

#define ELEMS  ((((LOG2_FFT_LEN)+1)<<1) >> 1)

// SEQUENTIAL VERSION
void __attribute__ ((noinline)) fft_mixed_2_4 (Complex_type * Inp_signal, Complex_type * Out_signal)
{
  int j;
  int dist = FFT_LEN_RADIX2 >> 1;
  int nbutterfly = FFT_LEN_RADIX2 >> 1;
  Complex_type * _in_ptr;
  Complex_type temp;

  // STAGE 1
  _in_ptr = &(Inp_signal[0]);
  for(j=0;j<nbutterfly;j++)
  {
    process_butterfly_real_radix2(_in_ptr, j, dist, twiddle_factors);
    _in_ptr++;
  }

  // STAGES 2 -> n (radix-4)
  for(j=0; j<FFT_LEN; j+=FFT_LEN/2)
#if !defined(SORT_OUTPUT)
    fft_radix4(Inp_signal+j, Out_signal, j);
#else
    fft_radix4(Inp_signal+j, Out_signal+j);
#endif

  // ORDER VALUES
#ifdef SORT_OUTPUT
#ifdef BITREV_LUT
  for(j=0; j<FFT_LEN; j+=LOG2_FFT_LEN)
  {
    unsigned int index[ELEMS];
    unsigned int val;
    // Read a sequence of permutation frm the LUT
    val = *((unsigned int *)(&bit_rev_2_4_LUT[j]));
    if(val == 0) break;
    index[0] = val & 0x0000FFFF;
    index[1] = val >> 16;
    for(int k=2; k<ELEMS; k+=2)
    {
      val = *((unsigned int *)(&bit_rev_2_4_LUT[j+k]));
      index[k] = val & 0x0000FFFF;
      index[k+1] = val >> 16;
    }
    // Apply permutations
    temp = Out_signal[index[LOG2_FFT_LEN-1]];
    for(int k=LOG2_FFT_LEN-1; k>0; k--)
    {
      Out_signal[index[k]] = Out_signal[index[k-1]];
    }
    Out_signal[index[0]] = temp;
  }
#else // !BITREV_LUT
  for(j=0; j<FFT_LEN; j++)
  {
    int skip;
    unsigned int index[LOG2_FFT_LEN];
    index[0] = j;
    skip = 0;
    // Find a closed set of permutations
    for(int k=0; k<LOG2_FFT_LEN-1; k++)
    {
      index[k+1] = bit_rev_2_4(index[k]);
      if(index[k+1] > j) { skip=1; break; }
    }
    // Apply permutations
    if(!skip)
    {
    temp = Out_signal[index[LOG2_FFT_LEN-1]];
    for(int k=LOG2_FFT_LEN-1; k>0; k--)
    {
      Out_signal[index[k]] = Out_signal[index[k-1]];
    }
    Out_signal[index[0]] = temp;
    }
  }
#endif // BITREV_LUT
#endif // SORT_OUTPUT
}


// PARALLEL VERSION
void __attribute__ ((noinline)) par_fft_mixed_2_4 (Complex_type * Inp_signal, Complex_type * Out_signal)
{
  int j;
  int dist = FFT_LEN_RADIX2 >> 1;
  int nbutterfly = FFT_LEN_RADIX2 >> 1;
  int core_id = get_core_id();
  Complex_type * _in_ptr;
  Complex_type temp;

  // STAGE 1 (input is real, stage=1)
  _in_ptr = &(Inp_signal[core_id]);
  for(j=0;j<nbutterfly/NUM_CORES;j++)
  {
     process_butterfly_real_radix2(_in_ptr, j*NUM_CORES+core_id, dist, twiddle_factors);
    _in_ptr+=NUM_CORES;
  }

  synch_barrier();

  // STAGE 2 -> n (radix-4)
  for(j=0; j<FFT_LEN; j+=FFT_LEN/2)
#if !defined(SORT_OUTPUT)
    par_fft_radix4(Inp_signal+j, Out_signal, j);
#else
    par_fft_radix4(Inp_signal+j, Out_signal+j);
#endif

  // ORDER VALUES
#ifdef SORT_OUTPUT
#ifdef BITREV_LUT
  for(j=LOG2_FFT_LEN*core_id; j<FFT_LEN; j+=LOG2_FFT_LEN*NUM_CORES)
  {
    unsigned int index[ELEMS];
    unsigned int val;
    // Read a sequence of permutation frm the LUT
    val = *((unsigned int *)(&bit_rev_2_4_LUT[j]));
    if(val == 0) break; // LUT ends with zero values
    index[0] = val & 0x0000FFFF;
    index[1] = val >> 16;
    for(int k=2; k<ELEMS; k+=2)
    {
      val = *((unsigned int *)(&bit_rev_2_4_LUT[j+k]));
      index[k] = val & 0x0000FFFF;
      index[k+1] = val >> 16;
    }
    // Apply permutations
    temp = Out_signal[index[LOG2_FFT_LEN-1]];
    for(int k=LOG2_FFT_LEN-1; k>0; k--)
    {
      Out_signal[index[k]] = Out_signal[index[k-1]];
    }
    Out_signal[index[0]] = temp;
  }
#else // !BITREV_LUT
  for(j=core_id; j<FFT_LEN; j+=NUM_CORES)
  {
    int skip;
    unsigned int index[LOG2_FFT_LEN];
    index[0] = j;
    skip = 0;
    // Find a closed set of permutations
    for(int k=0; k<LOG2_FFT_LEN-1; k++)
    {
      index[k+1] = bit_rev_2_4(index[k]);
      if(index[k+1] > j) { skip=1; break; }
    }
    // Apply permutations
    if(!skip)
    {
      temp = Out_signal[index[LOG2_FFT_LEN-1]];
      for(int k=LOG2_FFT_LEN-1; k>0; k--)
      {
        Out_signal[index[k]] = Out_signal[index[k-1]];
      }
      Out_signal[index[0]] = temp;
    }
  }
#endif // BITREV_LUT
  synch_barrier();
#endif // SORT_OUTPUT
}
