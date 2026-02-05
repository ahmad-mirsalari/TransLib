#include <stdio.h>
#include <math.h>

#include "ecg_processing.h"

#ifndef VECTORIAL
OUT_TYPE __attribute__ ((noinline)) single_convolution(OUT_TYPE *x, int n, FIL_TYPE *h, int nc)
{
  
  int k = 0;

  OUT_TYPE result = 0;
  int min = MIN(n, nc);
  for (k = 0; k < min; k++)  // It updates the buffersize with #define MIN
  {
    // result += ((INP_TYPE) x[n-1-k]) * h[min -1 - k];
    result += ((OUT_TYPE) x[n - min  + k]) * (OUT_TYPE)h[k];

  }
  return (OUT_TYPE) result;
}
#else

#if  defined(FP16_VECTORIATION)

OUT_TYPE __attribute__ ((noinline)) single_convolution(OUT_TYPE *x, int n, FIL_TYPE *h, int nc)
{
  int min = MIN(n, nc);
  int j = 0;
  OUT_TYPE result = 0;
  OUT_VTYPE temp1 = (OUT_VTYPE){0, 0};

  INP_VTYPE *Vs1 = (INP_VTYPE*)&x[n - min];
  FIL_VTYPE *Vf = (FIL_VTYPE*)&h[0]; 
  
  for (j = 0; j < min/2; j++)
  {
    temp1 += Vf[j] * Vs1[j];

  }
  result =  temp1[0] + temp1[1];
  return result;
}


#elif defined(FP8_VECTORIATION)

OUT_TYPE single_convolution(OUT_TYPE *x, int n, FIL_TYPE *h, int nc)
{
  int min = MIN(n, nc);
  int remainder = min % 4;
  int j = 0;
  OUT_TYPE result = 0;
  OUT_VTYPE temp1 = (OUT_VTYPE){0, 0, 0, 0};

  INP_VTYPE *Vs1 = (INP_VTYPE*)&x[min-4];
  FIL_VTYPE *Vf = (FIL_VTYPE*)&h[min-4]; 
  
  for (j = 0; j < min/4; j++)
  {
    temp1 += Vf[-j] * Vs1[-j];

  }
  // for (j = 0; j < remainder; j++) {
  //     temp1[0] += x[n-1-(min-1-j)] * h[n-1-(min-1-j)];
  //   }
  
  result = temp1[0] + temp1[1]+ temp1[2]+ temp1[3];
  return result;
}

#endif

#endif


OUT_TYPE __attribute__ ((noinline)) single_convolution_mix(dataType *x, int n, FIL_TYPE *h, int nc)
{
  
  int k = 0;

  OUT_TYPE result = 0;
  int min = MIN(n, nc);
  for (k = 0; k < min; k++)  // It updates the buffersize with #define MIN
  {
    // result += ((INP_TYPE) x[n-1-k]) * h[min -1 - k];
    // #ifdef DEBUG
    //     printf("x[%d] = %f, h[%d] = %f\n", n - min + k, (OUT_TYPE)x[n - min + k], k, (OUT_TYPE)h[k]);
    // #endif
    result += ((OUT_TYPE) x[n - min  + k]) * (OUT_TYPE)h[k];

  }
  return (OUT_TYPE)result;
}