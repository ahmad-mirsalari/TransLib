#include "pmsis.h"

#include "config.h"

#if DATA_WIDTH == 32 | !defined(VECTORIZATION)
void __attribute__((noinline)) convolve(
        INP_TYPE Signal[],
        FIL_TYPE *Filter, int FilterLength,
        OUT_TYPE *Output, int OutputLength)
{
  int i, j, z;
  //int L=OutputLength;

  OUT_TYPE sum=0;

#if NUM_CORES > 1
  for (i = pi_core_id(); i < OutputLength - FilterLength; i+=NUM_CORES)
#else
  for (i = 0; i < OutputLength - FilterLength; i++)
#endif
  {
    sum = 0;
#if IS_FLOAT
    for (j = 0; j < FilterLength; j+=2)
    {
      INP_TYPE a, c;
      FIL_TYPE b, d; 
      a = Signal[i + j];
      b = Filter[FilterLength - 1 - j];
      c = Signal[i + j + 1];
      d = Filter[FilterLength - 1 - j - 1];
      asm volatile("":::"memory");
      sum += (OUT_TYPE)(a * b);
      sum += (OUT_TYPE)(c * d);
    }
#else
    for (j = 0; j < FilterLength; j+=2)
    {
      INP_TYPE a, c;
      FIL_TYPE b, d; 
      a = Signal[i + j];
      b = Filter[FilterLength - 1 - j];
      c = Signal[i + j + 1];
      d = Filter[FilterLength - 1 - j - 1];
      asm volatile("":::"memory");
      sum += (a * b) >> SHIFT;
      sum += (c * d) >> SHIFT;      
    }
#endif
    Output[i] = sum;
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif

}
#else
static inline OUT_VTYPE pack_f16alt(float16alt a, float16alt b) {
  OUT_VTYPE result;
  __asm__ __volatile__ ("pv.pack.h %0, %1, %2" : "=f" (result): "f" (a), "f" (b) : );
  return result;
}


void __attribute__((always_inline)) convolve(
        INP_TYPE Signal[],
        FIL_TYPE *Filter, int FilterLength,
        OUT_TYPE *Output, int OutputLength)
{
  int i, j, z;

  OUT_TYPE sum=0;

#if NUM_CORES > 1
  for (i = 2*pi_core_id(); i < OutputLength - FilterLength; i+=2*NUM_CORES)
#else
  for (i = 0; i < OutputLength - FilterLength; i+=2)
#endif
  {
    sum = 0;
#if IS_FLOAT
    OUT_VTYPE temp1 = (OUT_VTYPE){0, 0};
    OUT_VTYPE temp2 = (OUT_VTYPE){0, 0};
    INP_VTYPE *Vs1 = (INP_VTYPE*)&Signal[i];
    INP_VTYPE *Vs2 = (INP_VTYPE*)&Signal[i+1];
    FIL_VTYPE *Vf = (FIL_VTYPE*)&Filter[FilterLength - 2];
    OUT_VTYPE *Vout = (OUT_VTYPE*)&Output[i];
    for (j = 0; j < FilterLength/2; j++)
    {
      FIL_VTYPE Fval = (FIL_VTYPE)(__builtin_shuffle(Vf[-j], (v2s){1,0}));
      temp1 += Vs1[j] * Fval;
      temp2 += Vs2[j] * Fval;
    }
    
    *Vout = (OUT_VTYPE) {temp1[0] + temp1[1], temp2[0] + temp2[1]};
#else
    v2s temp1 = (v2s){0, 0};
    v2s temp2 = (v2s){0, 0};
    v2s *Vs1 = (v2s*)&Signal[i];
    v2s *Vs2 = (v2s*)&Signal[i+1];
    v2s *Vf = (v2s*)&Filter[FilterLength - 2];
    v2s *Vout = (v2s*)&Output[i];
    for (j = 0; j < FilterLength/2; j++)
    {
      v2s Fval = (v2s)(__builtin_shuffle(Vf[-j], (v2s){1,0}));
      temp1 += (Vs1[j] * Fval) >> SHIFT;
      temp2 += (Vs2[j] * Fval) >> SHIFT;
    }


    *Vout = pack_f16alt(temp1[0] + temp1[1], temp2[0] + temp2[1]);
#endif
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif

}

#endif
