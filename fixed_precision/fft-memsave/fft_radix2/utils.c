#include <math.h>

#ifndef MIXED_RADIX
#include "fft.h"
#else
#include "../fft_mixed_2_8/fft_mixed.h" 
#endif

extern Complex_type twiddle_factors[FFT_LEN_RADIX2/2];

void compute_twiddles()
{
  int i;
  float Theta = -(2*M_PI)/FFT_LEN_RADIX2; //M_PI = pigreco
  for (i=0; i<FFT_LEN_RADIX2/2; i++)
  {
    float Phi = Theta*i;
    twiddle_factors[i].re = cosf(Phi);
    twiddle_factors[i].im = sinf(Phi);

  }
}
