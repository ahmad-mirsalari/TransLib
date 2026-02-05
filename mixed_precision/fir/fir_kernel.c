#include "pmsis.h"

#include "config.h"
#include "flags.h"

#if DATA_WIDTH == 32 | !defined(VECTORIZATION) // Non-vectorized version
void __attribute__((noinline)) convolve(
    INP_TYPE Signal[],
    FIL_TYPE *Filter, int FilterLength,
    OUT_TYPE *Output, int OutputLength)
{
  int i, j, z;
// int L=OutputLength;
#ifndef FABRIC
  int core_id = pi_core_id();
#else
  int core_id = 0;
#endif

#ifdef HWMIXED
float sum=0;
#else
OUT_TYPE sum = 0;
#endif

#if NUM_CORES > 1
  for (i = core_id; i < OutputLength - FilterLength; i += NUM_CORES)
#else
  for (i = 0; i < OutputLength - FilterLength; i++)
#endif
  {
    sum = 0;
    for (j = 0; j < (FilterLength & 0xfffffffe); j += 2)
    {

      #ifdef HWMIXED
      asm volatile("" ::: "memory");
      sum += (float)Signal[i + j] * (float)Filter[FilterLength - 1 - j];
      sum += (float)Signal[i + j + 1] * (float)Filter[FilterLength - 1 - j - 1];
      #else
      asm volatile("" ::: "memory");
      sum += (OUT_TYPE)(Signal[i + j]) * (OUT_TYPE)Filter[FilterLength - 1 - j];
      sum += (OUT_TYPE)(Signal[i + j + 1]) * (OUT_TYPE)Filter[FilterLength - 1 - j - 1];


      // INP_TYPE a, c;
      // FIL_TYPE b, d;
      // a = (INP_TYPE)Signal[i + j];
      // b = (FIL_TYPE)Filter[FilterLength - 1 - j];

      // c = (INP_TYPE)Signal[i + j + 1];
      // d = (FIL_TYPE)Filter[FilterLength - 1 - j - 1];

      // asm volatile("" ::: "memory");
      // sum += (OUT_TYPE)(a * b);
      // sum += (OUT_TYPE)(c * d);
      #endif
    }
    if (FilterLength & 0x00000001)
        {
          #ifdef HWMIXED
          sum += (float)Signal[i + FilterLength - 1] * (float)Filter[0];
          #else
          sum += (OUT_TYPE)(Signal[i + FilterLength - 1]) * (OUT_TYPE)(Filter[0]);
          #endif
        }

    Output[i] = (OUT_TYPE)sum;
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}

#else // Vectorized version

#ifdef FP8 // FP8 vectorized

// static inline OUT_VTYPE pack_f16alt(float16alt a, float16alt b)
// {
//   OUT_VTYPE result;
//   __asm__ __volatile__("pv.pack.h %0, %1, %2" : "=f"(result) : "f"(a), "f"(b) :);
//   return result;
// }

void __attribute__((always_inline)) convolve(
    INP_TYPE Signal[],
    FIL_TYPE *Filter, int FilterLength,
    OUT_TYPE *Output, int OutputLength)
{
#ifndef FABRIC
  int core_id = pi_core_id();
#else
  int core_id = 0;
#endif
int i, j, z;

#ifdef MIXED_VECTOR
float temp[4];
#else
OUT_VTYPE temp1, temp2, temp3, temp4;
#endif
int remainder = FilterLength & 0x3;
int vect_limit = FilterLength & ~0x3;
#if NUM_CORES > 1
  for (i = 4 * core_id; i < OutputLength - FilterLength; i += 4 * NUM_CORES)
#else
  for (i = 0; i < OutputLength - FilterLength; i += 4)
#endif
  {
#if IS_FLOAT

    #ifdef MIXED_VECTOR
    memset(temp, 0, sizeof(temp));
    #else
    temp1 = (OUT_VTYPE){0, 0, 0, 0};
    temp2 = (OUT_VTYPE){0, 0, 0, 0};
    temp3 = (OUT_VTYPE){0, 0, 0, 0};
    temp4 = (OUT_VTYPE){0, 0, 0, 0};
    #endif

    INP_VTYPE *Vs1 = (INP_VTYPE *)&Signal[i];
    INP_VTYPE *Vs2 = (INP_VTYPE *)&Signal[i + 1];
    INP_VTYPE *Vs3 = (INP_VTYPE *)&Signal[i + 2];
    INP_VTYPE *Vs4 = (INP_VTYPE *)&Signal[i + 3];
    
    #ifndef MIXED_VECTOR
    OUT_VTYPE *Vout = (OUT_VTYPE *)&Output[i];
    #endif


    #if defined(REVERSED)
    FIL_VTYPE *Vf = (FIL_VTYPE *)&Filter[0];
    #else
    FIL_VTYPE *Vf = (FIL_VTYPE *)&Filter[FilterLength - 4];
    #endif

    for (j = 0; j < FilterLength / 4; j++)
    {
      #if defined(REVERSED)
        #ifdef MIXED_VECTOR
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[0]) : "f"(Vs1[j]), "f"(Vf[j]) :);
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[1]) : "f"(Vs2[j]), "f"(Vf[j]) :);
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[2]) : "f"(Vs3[j]), "f"(Vf[j]) :);
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[3]) : "f"(Vs4[j]), "f"(Vf[j]) :);
        #else
          temp1 += Vs1[j] * Vf[j];
          temp2 += Vs2[j] * Vf[j];
          temp3 += Vs3[j] * Vf[j];
          temp4 += Vs4[j] * Vf[j];
        #endif
      #else // Not reversed

      FIL_VTYPE Fval = (FIL_VTYPE)(__builtin_shuffle(Vf[-j], (v4s){3, 2, 1, 0}));
        #ifdef MIXED_VECTOR
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[0]) : "f"(Vs1[j]), "f"(Fval) :);
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[1]) : "f"(Vs2[j]), "f"(Fval) :);
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[2]) : "f"(Vs3[j]), "f"(Fval) :);
          __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[3]) : "f"(Vs4[j]), "f"(Fval) :);
        #else
          temp1 += Vs1[j] * Fval;
          temp2 += Vs2[j] * Fval;
          temp3 += Vs3[j] * Fval;
          temp4 += Vs4[j] * Fval;
        #endif
      
      #endif
    }


    #if defined(REVERSED)
    for (z = FilterLength - remainder; z < FilterLength; z++)
    {
      #ifdef MIXED_VECTOR
        temp[0] += (float)Signal[i +  z] * (float)Filter[z];
        temp[1] += (float)Signal[i + 1 +  z] * (float)Filter[z];
        temp[2] += (float)Signal[i + 2 +  z] * (float)Filter[z];
        temp[3] += (float)Signal[i + 3 +  z] * (float)Filter[z];
      #else
        temp1[0] += Signal[i +  z] * Filter[z];
        temp2[0] += Signal[i + 1 +  z] * Filter[z];
        temp3[0] += Signal[i + 2 +  z] * Filter[z];
        temp4[0] += Signal[i + 3 +  z] * Filter[z];
      #endif
    }

    #else // not REVERSED
      for (z = remainder -1 ; z >=0 ; z--) {
        #ifdef MIXED_VECTOR
          temp[0] += (float)Signal[i - z + FilterLength - 1] * (float)Filter[z];
          temp[1] += (float)Signal[i - z + 1 + FilterLength - 1 ] * (float)Filter[z];
          temp[2] += (float)Signal[i - z + 2 + FilterLength - 1 ] * (float)Filter[z];
          temp[3] += (float)Signal[i - z + 3 + FilterLength - 1 ] * (float)Filter[z];
        #else
          temp1[0] += Signal[i - z + FilterLength - 1] * Filter[z];
          temp2[0] += Signal[i - z + 1 + FilterLength - 1 ] * Filter[z];
          temp3[0] += Signal[i - z + 2 + FilterLength - 1 ] * Filter[z];
          temp4[0] += Signal[i - z + 3 + FilterLength - 1 ] * Filter[z]; 
        #endif
    }
    #endif

    #ifdef MIXED_VECTOR
      Output[i] = (OUT_TYPE)temp[0];
      Output[i + 1] = (OUT_TYPE)temp[1];
      Output[i + 2] = (OUT_TYPE)temp[2];
      Output[i + 3] = (OUT_TYPE)temp[3];
    #else
    *Vout = (OUT_VTYPE){temp1[0] + temp1[1] + temp1[2] + temp1[3],
                        temp2[0] + temp2[1] + temp2[2] + temp2[3],
                        temp3[0] + temp3[1] + temp3[2] + temp3[3],
                        temp4[0] + temp4[1] + temp4[2] + temp4[3]};
    #endif
#endif
  }
#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}

#else //FP16 or FP16ALT vectorized
  // static inline OUT_VTYPE pack_f16alt(float16alt a, float16alt b)
  // {
  //   OUT_VTYPE result;
  //   __asm__ __volatile__("pv.pack.h %0, %1, %2" : "=f"(result) : "f"(a), "f"(b) :);
  //   return result;
  // }

  void __attribute__((always_inline)) convolve(
      INP_TYPE Signal[],
      FIL_TYPE *Filter, int FilterLength,
      OUT_TYPE *Output, int OutputLength)
  {

    #ifdef MIXED_VECTOR
    float temp[2];
    float sum = 0;
    #else
    OUT_VTYPE temp1,temp2;
    OUT_TYPE sum = 0;
    #endif


    #ifndef FABRIC
      int core_id = pi_core_id();
    #else
      int core_id = 0;
    #endif
      int i, j, z;



    #if NUM_CORES > 1
      for (i = 2 * core_id; i < OutputLength - FilterLength; i += 2 * NUM_CORES)
    #else
      for (i = 0; i < OutputLength - FilterLength; i += 2)
    #endif
      {

        #ifdef MIXED_VECTOR
        temp[0] = 0;
        temp[1] = 0;
        #else
        temp1 = (OUT_VTYPE){0, 0};
        temp2 = (OUT_VTYPE){0, 0};
        #endif

        sum = 0;
        INP_VTYPE *Vs1 = (INP_VTYPE *)&Signal[i];
        INP_VTYPE *Vs2 = (INP_VTYPE *)&Signal[i + 1];

        #ifndef MIXED_VECTOR
        OUT_VTYPE *Vout = (OUT_VTYPE *)&Output[i];
        #endif

        #if defined(REVERSED)
            FIL_VTYPE *Vf = (FIL_VTYPE *)&Filter[0];
        #else
            FIL_VTYPE *Vf = (FIL_VTYPE *)&Filter[FilterLength - 2];
        #endif

        for (j = 0; j < FilterLength / 2; j++)
        {
          #if defined(REVERSED)
            #ifdef MIXED_VECTOR
              #ifdef FP16
              __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[0]) : "f"(Vs1[j]), "f"(Vf[j]) :);
              __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[1]) : "f"(Vs2[j]), "f"(Vf[j]) :);
              #else
              __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[0]) : "f"(Vs1[j]), "f"(Vf[j]) :);
              __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[1]) : "f"(Vs2[j]), "f"(Vf[j]) :);
              #endif
            #else
                
                temp1 += Vs1[j] * Vf[j];
                temp2 += Vs2[j] * Vf[j];
            #endif
          #else
                FIL_VTYPE Fval = (FIL_VTYPE)(__builtin_shuffle(Vf[-j], (v2s){1, 0}));

                #ifdef MIXED_VECTOR
                  #ifdef FP16
                    __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[0]) : "f"(Vs1[j]), "f"(Fval) :);
                    __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[1]) : "f"(Vs2[j]), "f"(Fval) :);
                  #else
                  
                    __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[0]) : "f"(Vs1[j]), "f"(Fval) :);
                    __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[1]) : "f"(Vs2[j]), "f"(Fval) :);
                  #endif
                #else
                  temp1 += Vs1[j] * Fval;
                  temp2 += Vs2[j] * Fval;
                #endif
          #endif
        } 

        if (FilterLength & 0x00000001)
        {
          #if defined(REVERSED)
            #ifdef MIXED_VECTOR
                temp[0] += (float)Signal[i + FilterLength - 1] * (float)Filter[FilterLength - 1];
                temp[1] += (float)Signal[i + 1 + FilterLength - 1] * (float)Filter[FilterLength - 1];
            #else
                temp1[0] += Signal[i + FilterLength - 1] * Filter[FilterLength - 1];
                temp2[0] += Signal[i + 1 + FilterLength - 1] * Filter[FilterLength - 1];
            #endif

          #else // Not reversed
            #ifdef MIXED_VECTOR
              temp[0] += (float)Signal[i + FilterLength - 1] * (float)Filter[0];
              temp[1] += (float)Signal[i + 1 + FilterLength - 1] * (float)Filter[0];
            #else
                temp1[0] += Signal[i + FilterLength - 1] * Filter[0];
                temp2[0] += Signal[i + 1 + FilterLength - 1] * Filter[0];
            #endif
          #endif
        }
        #ifdef MIXED_VECTOR
          Output[i] = (OUT_TYPE)temp[0];
          Output[i + 1] = (OUT_TYPE)temp[1];
        #else
          *Vout = (OUT_VTYPE){temp1[0] + temp1[1], temp2[0] + temp2[1]};
        #endif
      }

      #if NUM_CORES > 1
        pi_cl_team_barrier();
      #endif
    }
  #endif // end if for fp16 or fp16alt
#endif // end if for vectorized
