#include <math.h>
#include "fft.h"

extern Complex_type twiddle_factors[FFT_LEN_RADIX4/2];

void compute_twiddles()
{
  int i;
  DTYPE Theta = -2*(2*M_PI)/(FFT_LEN_RADIX4*2); //solo i pari cioè FFT_LEN_RADIX2/2 => *2 (e FFT_LEN_RADIX4*2=FFT_LEN_RADIX2)
  for (i=0; i<FFT_LEN_RADIX4/2; i++)
  {
    DTYPE Phi = Theta*i;
    twiddle_factors[i].re = cosf(Phi);
    twiddle_factors[i].im = sinf(Phi);
  }
}
