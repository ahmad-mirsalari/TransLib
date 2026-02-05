#include "pmsis.h"
#include <stdio.h>
#include <math.h>
#include "wavelet.h"
#include "config.h"
#include "stats.h"

#include "kernels.def" //filters

#pragma GCC push_options
#if (NUM_CORES > 1) && (NC > 2)
#pragma GCC optimize("no-tree-ch")
#pragma GCC optimize("no-tree-loop-im")
#endif

#define MADD_F16(c, a, b, temp)                                              \
  __asm__ __volatile__("fmul.h %0, %1, %2" : "=f"(temp) : "f"(a), "f"(b) :); \
  __asm__ __volatile__("fadd.h %0, %1, %2" : "=f"(c) : "f"(c), "f"(temp) :);


#ifdef VECTORIAL
void dwt_step(INP_TYPE *input, OUT_TYPE *output, size_t n, size_t idx_level) // change name of idx_level?
{
  
  size_t i, ii;
  int j;


  #ifdef PARALLEL
    int core_id = pi_core_id();
  #else
    int core_id = 0;
  #endif

  ii = 0;
  #ifdef MIXED_VECTOR
  float h, g, h2, g2;
  float h_v, g_v, h2_v, g2_v;
  #else
  OUT_TYPE h, g, h2, g2;
  OUT_VTYPE h_v, g_v, h2_v, g2_v;
  #endif
  INP_VTYPE in, in1;
  FIL_VTYPE temp, temp_n;

  
  size_t n_it;

#if NC == 2 //-----------------------------------------  NC=2
  temp[0] = R2_2;
  temp[1] = R2_2;
  temp_n[0] = R2_2;
  temp_n[1] = -R2_2;
  n_it = (n / 4) * 4;
  size_t n_rest = n - n_it; // n_rest = n%4
#ifdef PARALLEL
  for (i = 4 * core_id; i < n_it; i += NUM_CORES * 4) // loop unrolling (without it was *2)
#else
  for (i = 0; i < n_it; i += 4) // loop unrolling (without it was +=2, for the downsampling)
#endif
  {
    in = *((INP_VTYPE *)&input[i]);
    in1 = *((INP_VTYPE *)&input[i + 2]);
    #ifdef MIXED_VECTOR
        #ifdef FP16
        __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(h_v) : "f"(in), "f"(temp) :);
        __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(h2_v) : "f"(in1), "f"(temp) :);
        __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(g_v) : "f"(in), "f"(temp_n) :);
        __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(g2_v) : "f"(in1), "f"(temp_n) :);
        #else
        __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(h_v) : "f"(in), "f"(temp) :);
        __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(h2_v) : "f"(in1), "f"(temp) :);
        __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(g_v) : "f"(in), "f"(temp_n) :);
        __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(g2_v) : "f"(in1), "f"(temp_n) :);
        #endif
    output[ii + core_id * 2] = (OUT_TYPE)h_v; // following levels input
    output[ii + core_id * 2 + idx_level] = (OUT_TYPE)g_v;
    output[ii + 1 + core_id * 2] = (OUT_TYPE)h2_v;
    output[ii + 1 + core_id * 2 + idx_level] = (OUT_TYPE)g2_v;
    #else // Not MIXED_VECTOR
    h_v = in * temp;
    h2_v = in1 * temp;
    g_v = in * temp_n;
    g2_v = in1 * temp_n;

    output[ii + core_id * 2] = h_v[0] + h_v[1]; // following levels input
    output[ii + core_id * 2 + idx_level] = g_v[0] + g_v[1];
    output[ii + 1 + core_id * 2] = h2_v[0] + h2_v[1];
    output[ii + 1 + core_id * 2 + idx_level] = g2_v[0] + g2_v[1];
    #endif

    ii += NUM_CORES * 2;
  } /// while (i < n_it);
  #ifdef MIXED_VECTOR
  h = h_v;
  h2 = h2_v;
  g2 = g2_v;
  g = g_v;
  #else
  h = h_v[0] + h_v[1];
  h2 = h2_v[0] + h2_v[1];
  g2 = g2_v[0] + g2_v[1];
  g = g_v[0] + g_v[1];
  #endif
  ii = n / 4 * 2; // following levels input size (ii=n_it/2, for the downsampling)

#ifdef PARALLEL
  if (core_id == 0)
  {
#endif

    if (n_rest > 0)
    {
      switch (n_rest)
      {
      case 1:
        #ifdef MIXED_VECTOR
        g = h = (float)R2_2 * (float)input[n - 1];
        #else // Not MIXED_VECTOR
        g = h = R2_2 * input[n - 1];
        #endif //END MIXED_VECTOR
        break;
      case 2:
        in1 = *((INP_VTYPE *)&input[i - 2]);
        #ifdef MIXED_VECTOR
          #ifdef FP16
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(h_v) : "f"(in1), "f"(temp) :);
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(g_v) : "f"(in1), "f"(temp_n) :);
          #else
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(h_v) : "f"(in1), "f"(temp) :);
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(g_v) : "f"(in1), "f"(temp_n) :);
          #endif
        h = h_v;
        g = g_v;
        #else // Not MIXED_VECTOR
        h_v = in1 * temp;
        g_v = in1 * temp_n;
        h = h_v[0] + h_v[1];
        g = g_v[0] + g_v[1];
        #endif //END MIXED_VECTOR
        break;
      default:
        in1 = *((INP_VTYPE *)&input[i - 3]);
        #ifdef MIXED_VECTOR
          #ifdef FP16
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(h_v) : "f"(in1), "f"(temp) :);
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(g_v) : "f"(in1), "f"(temp_n) :);
          #else
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(h_v) : "f"(in1), "f"(temp) :);
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(g_v) : "f"(in1), "f"(temp_n) :);
          #endif
        h = h_v;
        g = g_v;
        h2 = (float)R2_2 * (float)input[n - 1];
        
        #else // Not MIXED_VECTOR
        h_v = in1 * temp;
        g_v = in1 * temp_n;
        h = h_v[0] + h_v[1];
        g = g_v[0] + g_v[1];
        h2 = R2_2 * input[n - 1];
        #endif   //END MIXED_VECTOR

        output[ii + 1] = (OUT_TYPE)h2;
        output[ii + 1 + idx_level] = (OUT_TYPE)h2;
      }
      output[ii] = (OUT_TYPE)h;
      output[ii + idx_level] = (OUT_TYPE)g;
    }

#ifdef PARALLEL
  } // core 0
  pi_cl_team_barrier();
#endif

  int next_inputs = (n + NC - 1) / 2;
#ifdef PARALLEL
  for (i = 2 * core_id; i < next_inputs / 2 * 2; i += 2 * NUM_CORES)
#else
  for (i = 0; i < next_inputs / 2 * 2; i += 2)
#endif
  {
    OUT_TYPE a = output[i];
    OUT_TYPE b = output[i + 1];
    asm volatile("" ::: "memory");
    input[i] = (INP_TYPE)a;
    input[i + 1] = (INP_TYPE)b;
  }
#ifdef PARALLEL
  if (core_id == 0)
#endif
    if (next_inputs & 0x1)
      input[next_inputs - 1] = output[next_inputs - 1];
  ;

#else // -------------------------------------- NC > 4

  #ifdef MIXED_VECTOR
  float a=0, b=0;
  #else
  OUT_VTYPE a = (OUT_VTYPE){0, 0};
  OUT_VTYPE b = (OUT_VTYPE){0, 0};
  #endif

  #ifdef PARALLEL
    for (i = core_id; i < NC / 2 - 1; i += NUM_CORES)
  #else
    for (i = 0; i < NC / 2 - 1; i++)
  #endif
        {

          #ifdef MIXED_VECTOR
          a = 0, b = 0;
          #else
          a = (OUT_VTYPE){0, 0};
          b = (OUT_VTYPE){0, 0};
          #endif
          for (j = 2 * (i + 1) - 1; j >= 1; j -= 2) // j<2*(i+1)=NC/2-1 above respect to i and here respect to j;
          {
            INP_VTYPE in = *((INP_VTYPE *)&input[2 * (i + 1) - j - 1]);
            FIL_VTYPE lo = *((FIL_VTYPE *)&Lo[(NC - 1) - j]);
            FIL_VTYPE hi = *((FIL_VTYPE *)&Hi[(NC - 1) - j]);
            asm volatile("" ::: "memory");
            #ifdef MIXED_VECTOR
                #ifdef FP16
                __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
                __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
                #else
                __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
                __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
                #endif
            #else      
              a += in * lo; // cA next levels, reversed Lo
              b += in * hi; // cD final output, rev Hi (in matlab is the 2nd array, i.e. [lo hi]=wavefilters('db4'))
            #endif

          }
          #ifdef MIXED_VECTOR
          output[ii + core_id] = (OUT_TYPE)a;
          output[ii + core_id + idx_level] = (OUT_TYPE)b;
          #else
          output[ii + core_id] = a[0] + a[1];
          output[ii + core_id + idx_level] = b[0] + b[1];
          #endif
          ii += NUM_CORES;
        }
  #ifdef PARALLEL
    ii = NC / 2 - 1;
  #endif
  // middle and final part of the array
  #ifdef PARALLEL
    for (i = (NC - 1) + core_id * 2; i < n; i += NUM_CORES * 2)
  #else
    for (i = NC - 1; i < n; i += 2)
  #endif
    {
      #ifdef MIXED_VECTOR
      a = 0, b = 0;
      #else
      a = (OUT_VTYPE){0, 0};
      b = (OUT_VTYPE){0, 0};
      #endif
      for (j = (NC - 1); j >= 1; j -= 2)
      {
        INP_VTYPE in = *((INP_VTYPE *)&input[i - j]);
        FIL_VTYPE lo = *((FIL_VTYPE *)&Lo[(NC - 1) - j]);
        FIL_VTYPE hi = *((FIL_VTYPE *)&Hi[(NC - 1) - j]);
        asm volatile("" ::: "memory");
        #ifdef MIXED_VECTOR
            #ifdef FP16
            __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
            __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
            #else
            __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
            __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
            #endif
        #else      
          a += in * lo; // cA next levels, reversed Lo
          b += in * hi; // cD final output, rev Hi (in matlab is the 2nd array, i.e. [lo hi]=wavefilters('db4'))
        #endif
      }

      if (NC & 0x00000001) // odd NC
      {
        INP_TYPE in = input[i - 0];
        FIL_TYPE lo = Lo[(NC - 1) - 0];
        FIL_TYPE hi = Hi[(NC - 1) - 0];
        asm volatile("" ::: "memory");

        #ifdef MIXED_VECTOR
        a += (float)in * (float)lo;
        b += (float)in * (float)hi;
        #else
        a[0] += in * lo;
        b[0] += in * hi;
        #endif
      } // END odd NC
      #ifdef MIXED_VECTOR
      output[ii + core_id] = (OUT_TYPE)a;
      output[ii + core_id + idx_level] = (OUT_TYPE)b;
      #else
      output[ii + core_id] = a[0] + a[1];
      output[ii + core_id + idx_level] = b[0] + b[1];
      #endif
      ii += NUM_CORES;
    } 

#ifdef PARALLEL
  ii = n / 2;
#endif
  if (n % 2 == 0) // n is even
  {
    #ifdef PARALLEL
        for (i = core_id; i < NC / 2 - 1; i += NUM_CORES)
    #else
        for (i = 0; i < NC / 2 - 1; i++)
    #endif
          {
            #ifdef MIXED_VECTOR
            a = 0, b = 0;
            #else
            a = (OUT_VTYPE){0, 0};
            b = (OUT_VTYPE){0, 0};
            #endif
            for (j = NC - 2 * (i + 1) - 1; j >= 1; j -= 2)
            {

              INP_VTYPE in = *((INP_VTYPE *)&input[n - j - 1]);
              FIL_VTYPE lo = *((FIL_VTYPE *)&Lo[(NC - 1) - (2 * (i + 1) + j)]);
              FIL_VTYPE hi = *((FIL_VTYPE *)&Hi[(NC - 1) - (2 * (i + 1) + j)]);
              asm volatile("" ::: "memory");
              #ifdef MIXED_VECTOR
                  #ifdef FP16
                  __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
                  __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
                  #else
                  __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
                  __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
                  #endif
              #else      
                a += in * lo; // cA next levels, reversed Lo
                b += in * hi; // cD final output, rev Hi (in matlab is the 2nd array, i.e. [lo hi]=wavefilters('db4'))
              #endif
            } // while(j < NC-2*(i+1));

            if (NC & 0x00000001)
            {
              INP_TYPE in = input[i - 0];
              FIL_TYPE lo = Lo[(NC - 1) - (2 * (i + 1))];
              FIL_TYPE hi = Hi[(NC - 1) - (2 * (i + 1))];
              asm volatile("" ::: "memory");
              #ifdef MIXED_VECTOR
              a += (float)in * (float)lo;
              b += (float)in * (float)hi;
              #else
              a[0] += in * lo;
              b[0] += in * hi;
              #endif
            }
            #ifdef MIXED_VECTOR
            output[ii + core_id] = (OUT_TYPE)a;
            output[ii + core_id + idx_level] = (OUT_TYPE)b;
            #else
            output[ii + core_id] = a[0] + a[1];
            output[ii + core_id + idx_level] = b[0] + b[1];
            #endif
            ii += NUM_CORES;
          }
    #ifdef PARALLEL
        ii = n / 2 + (NC / 2 - 1);
    #endif
  } //END n is even
  else // n is odd
  {
    #ifdef PARALLEL
        for (i = core_id; i < NC / 2; i += NUM_CORES)
    #else
        for (i = 0; i < NC / 2; i++)
    #endif
        {
          #ifdef MIXED_VECTOR
          a = 0, b = 0;
          #else
          a = (OUT_VTYPE){0, 0};
          b = (OUT_VTYPE){0, 0};
          #endif
          for (j = NC - 2 * (i + 1); j >= 1; j -= 2)

          {

            INP_VTYPE in = *((INP_VTYPE *)&input[n - j - 1]);
            FIL_VTYPE lo = *((FIL_VTYPE *)&Lo[(NC - 1) - (2 * (i + 1) + j - 1)]);
            FIL_VTYPE hi = *((FIL_VTYPE *)&Hi[(NC - 1) - (2 * (i + 1) + j - 1)]);

            asm volatile("" ::: "memory");
            #ifdef MIXED_VECTOR
              #ifdef FP16
              __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
              __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
              #else
              __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(a) : "f"(in), "f"(lo) :);
              __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(b) : "f"(in), "f"(hi) :);
              #endif
            #else      
              a += in * lo; // cA next levels, reversed Lo
              b += in * hi; // cD final output, rev Hi (in matlab is the 2nd array, i.e. [lo hi]=wavefilters('db4'))
            #endif
          } 

          if (NC & 0x00000001)
            {
            }
          else
          {
            INP_TYPE in = input[n - 0 - 1];
            FIL_TYPE lo = Lo[(NC - 1) - (2 * (i + 1) + 0 - 1)];
            FIL_TYPE hi = Hi[(NC - 1) - (2 * (i + 1) + 0 - 1)];
            asm volatile("" ::: "memory");
            #ifdef MIXED_VECTOR
            a += (float)in * (float)lo;
            b += (float)in * (float)hi;
            #else
            a[0] += in * lo;
            b[0] += in * hi;
            #endif
          }
        #ifdef MIXED_VECTOR
        output[ii + core_id] = (OUT_TYPE)a;
        output[ii + core_id + idx_level] = (OUT_TYPE)b;
        #else
        output[ii + core_id] = a[0] + a[1];
        output[ii + core_id + idx_level] = b[0] + b[1];
        #endif
        ii += NUM_CORES;
      }
    #ifdef PARALLEL
        ii = n / 2 + NC / 2;
    #endif
  } // END odd

  int next_inputs = (n + NC - 1) / 2;
  #ifdef PARALLEL
    pi_cl_team_barrier();

    i = 2 * core_id;
    if (i < next_inputs / 2 * 2)
      do
  #else
  
    i = 0;
    do
  #endif // END PARALLEL
    {
      OUT_TYPE a0 = output[i];
      OUT_TYPE b0 = output[i + 1];
      asm volatile("" ::: "memory");
      input[i] = (INP_TYPE)a0;
      input[i + 1] = (INP_TYPE)b0;
      i += 2 * NUM_CORES;
    } while (i < next_inputs / 2 * 2);
  #ifdef PARALLEL
    if (core_id == 0)
  #endif
    if (next_inputs & 0x1)
      input[next_inputs - 1] = (INP_TYPE)output[next_inputs - 1];
#endif // END NC>4

#ifdef PARALLEL
  pi_cl_team_barrier();
#endif
}

#else // SCALAR
void dwt_step(INP_TYPE *input, OUT_TYPE *output, size_t n, size_t idx_level) // change name of idx_level?
{
  size_t i, ii, j;

  #ifdef PARALLEL
    int core_id = pi_core_id();
  #else
    int core_id = 0;
  #endif
  ii = 0;
  
  #ifdef HWMIXED
  float h, g, h2, g2;
  #else
  OUT_TYPE h, g, h2, g2;
  #endif
  size_t n_it;

#if NC == 2 // -----------------------------------------  NC=2

  n_it = (n / 4) * 4;
  size_t n_rest = n - n_it; // n_rest = n%4

  #ifdef PARALLEL
    i = 4 * core_id;
    if (i < n_it)
      do
  #else // NO PARALLEL
    i = 0;
    if (n_it > 0)
      do
  #endif // END PARALLEL
  {
    #ifdef HWMIXED
    h = (float)R2_2 * (float)input[i] + (float)R2_2 * (float)input[i + 1]; // approssimation coeff (reversed low filter Lo)
    h2 = (float)R2_2 * (float)input[i + 2] + (float)R2_2 * (float)input[i + 3];
    g = (float)R2_2 * (float)input[i] - (float)R2_2 * (float)input[i + 1]; // detail coeff (reversed high filter Hi)
    g2 = (float)R2_2 * (float)input[i + 2] - (float)R2_2 * (float)input[i + 3];
    #else
    h = (OUT_TYPE)R2_2 * (OUT_TYPE)input[i] + (OUT_TYPE)R2_2 * (OUT_TYPE)input[i + 1]; // approssimation coeff (reversed low filter Lo)
    h2 = (OUT_TYPE)R2_2 * (OUT_TYPE)input[i + 2] + (OUT_TYPE)R2_2 * (OUT_TYPE)input[i + 3];
    g = (OUT_TYPE)R2_2 * (OUT_TYPE)input[i] - (OUT_TYPE)R2_2 * (OUT_TYPE)input[i + 1]; // detail coeff (reversed high filter Hi)
    g2 = (OUT_TYPE)R2_2 * (OUT_TYPE)input[i + 2] - (OUT_TYPE)R2_2 * (OUT_TYPE)input[i + 3];
    #endif
    output[ii + core_id * 2] = (OUT_TYPE)h;             // following levels input
    output[ii + core_id * 2 + idx_level] = (OUT_TYPE)g; // final output
    output[ii + 1 + core_id * 2] =(OUT_TYPE) h2;
    output[ii + 1 + core_id * 2 + idx_level] = (OUT_TYPE) g2;

    ii += NUM_CORES * 2;
    i += NUM_CORES * 4;
  } while (i < n_it);

  ii = n / 4 * 2; // following levels input size (ii=n_it/2, for the downsampling)

#ifdef PARALLEL
  if (core_id == 0)
  {
#endif // End PARALLEL

if (n_rest > 0)
{
  switch (n_rest)
  {
  case 1:
    #ifdef HWMIXED
    g = h = (float)R2_2 * (float)input[n - 1];
    #else
    g = h = (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 1];
    #endif
    break;
  case 2:
    
    #ifdef HWMIXED
    h = (float)R2_2 * (float)input[n - 2] + (float)R2_2 * (float)input[n - 1];
    g = (float)R2_2 * (float)input[n - 2] - (float)R2_2 * (float)input[n - 1];
    #else
    h = (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 2] + (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 1];
    g = (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 2] - (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 1];
    #endif
    break;
  default:
    #ifdef HWMIXED
    h = (float)R2_2 * (float)input[n - 3] + (float)R2_2 * (float)input[n - 2];
    h2 = (float)R2_2 * (float)input[n - 1];
    g = (float)R2_2 * (float)input[n - 3] - (float)R2_2 * (float)input[n - 2];
    #else
    h = (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 3] + (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 2];
    h2 = (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 1];
    g = (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 3] - (OUT_TYPE)R2_2 * (OUT_TYPE)input[n - 2];
    #endif
    output[ii + 1] = (OUT_TYPE)h2;
    output[ii + 1 + idx_level] = (OUT_TYPE)h2;
  }
  output[ii] = (OUT_TYPE)h;
  output[ii + idx_level] = (OUT_TYPE)g;
}

#ifdef PARALLEL
  } // core 0
  pi_cl_team_barrier();
#endif

int next_inputs = (n + NC - 1) / 2;
#ifdef PARALLEL
  for (i = 2 * core_id; i < next_inputs / 2 * 2; i += 2 * NUM_CORES)
#else
  for (i = 0; i < next_inputs / 2 * 2; i += 2)
#endif

{
  OUT_TYPE a = output[i];
  OUT_TYPE b = output[i + 1];
  asm volatile("" ::: "memory");
  input[i] = (OUT_TYPE)a;
  input[i + 1] = (OUT_TYPE)b;
}

#ifdef PARALLEL
  if (core_id == 0)
#endif

if (next_inputs & 0x1)
  input[next_inputs - 1] = output[next_inputs - 1];

#else // -----------------------------------------  NC > 4
  OUT_TYPE temp;
#ifdef PARALLEL
  for (i = core_id; i < NC / 2 - 1; i += NUM_CORES)
#else
  for (i = 0; i < NC / 2 - 1; i++)
#endif
  {
    #ifdef HWMIXED
    float a = 0.0f;
    float b = 0.0f;
    #else
    OUT_TYPE a = 0.0f;
    OUT_TYPE b = 0.0f;
    #endif
    j = 0;
    do
    {
      // beginning part of each layer
      INP_TYPE in = input[2 * (i + 1) - j - 1];
      FIL_TYPE lo = Lo[j];
      FIL_TYPE hi = Hi[j];
      asm volatile("" ::: "memory");

      #ifdef HWMIXED
      a += (float)in * (float)lo; // cA next levels, reversed Lo
      b += (float)in * (float)hi; // cD final output,
      #else
      a += (OUT_TYPE)in * (OUT_TYPE)lo; // cA next levels, reversed Lo
      b += (OUT_TYPE)in * (OUT_TYPE)hi; // cD final output, rev Hi (in matlab is the 2nd array, i.e. [lo hi]=wavefilters('db4'))
      #endif
      j++;
    } while (j < 2 * (i + 1));
    output[ii + core_id] = (OUT_TYPE)a;
    output[ii + core_id + idx_level] = (OUT_TYPE)b;

    ii += NUM_CORES;
  }

#ifdef PARALLEL
  ii = NC / 2 - 1;
#endif

// middle and final part of the array
#ifdef PARALLEL
  for (i = (NC - 1) + core_id * 2; i < n; i += NUM_CORES * 2)
#else
  for (i = NC - 1; i < n; i += 2)
#endif
  {
    #ifdef HWMIXED
    float a = 0.0f;
    float b = 0.0f;
    #else
    OUT_TYPE a = 0.0f;
    OUT_TYPE b = 0.0f;
    #endif
    for (j = 0; j < NC; j++)
    {

      INP_TYPE in = input[i - j];
      FIL_TYPE lo = Lo[j];
      FIL_TYPE hi = Hi[j];

      asm volatile("" ::: "memory");
      #ifdef HWMIXED
      a += (float)in * (float)lo;
      b += (float)in * (float)hi;
      #else
      a += (OUT_TYPE)in * (OUT_TYPE)lo;
      b += (OUT_TYPE)in * (OUT_TYPE)hi;
      #endif
    }
    output[ii + core_id] = (OUT_TYPE)a;
    output[ii + core_id + idx_level] = (OUT_TYPE)b;

    ii += NUM_CORES;
  }
#ifdef PARALLEL
  ii = n / 2;
#endif
  if (n % 2 == 0) // ------------ even
  {
    #ifdef PARALLEL
        for (i = core_id; i < NC / 2 - 1; i += NUM_CORES)
    #else
        for (i = 0; i < NC / 2 - 1; i++)
    #endif
          {
            #ifdef HWMIXED
            float a = 0.0f;
            float b = 0.0f;
            #else
            OUT_TYPE a = 0.0f;
            OUT_TYPE b = 0.0f;
            #endif
            // for(j = 0; j < NC-2*(i+1); j++)
            j = 0;
            do
            {
              INP_TYPE in = input[n - j - 1];
              FIL_TYPE lo = Lo[2 * (i + 1) + j];
              FIL_TYPE hi = Hi[2 * (i + 1) + j];
              asm volatile("" ::: "memory");
              #ifdef HWMIXED
              a += (float)in * (float)lo;
              b += (float)in * (float)hi;
              #else
              a += (OUT_TYPE)in * (OUT_TYPE)lo;
              b += (OUT_TYPE)in * (OUT_TYPE)hi;
              #endif
              j++;
            } while (j < NC - 2 * (i + 1));
            output[ii + core_id] = (OUT_TYPE)a;
            output[ii + core_id + idx_level] = (OUT_TYPE)b;
            ii += NUM_CORES;
          }
          #ifdef PARALLEL
              ii = n / 2 + (NC / 2 - 1);
          #endif
  }
  else // --------------- odd
  {
    #ifdef PARALLEL
        for (i = core_id; i < NC / 2; i += NUM_CORES)
    #else
        for (i = 0; i < NC / 2; i++)
    #endif
          {
            #ifdef HWMIXED
            float a = 0.0f;
            float b = 0.0f;
            #else
            OUT_TYPE a = 0.0f;
            OUT_TYPE b = 0.0f;
            #endif
            j = 0;
            do
            {
              INP_TYPE in = input[n - j - 1];
              FIL_TYPE lo = Lo[2 * (i + 1) + j - 1];
              FIL_TYPE hi = Hi[2 * (i + 1) + j - 1];
              asm volatile("" ::: "memory");
              #ifdef HWMIXED
              a += (float)in * (float)lo;
              b += (float)in * (float)hi;
              #else
              a += (OUT_TYPE)in * (OUT_TYPE)lo;
              b += (OUT_TYPE)in * (OUT_TYPE)hi;
              #endif
              j++;
            } while (j < NC - 2 * (i + 1) + 1);
            output[ii + core_id] = (OUT_TYPE)a;
            output[ii + core_id + idx_level] = (OUT_TYPE)b;
            ii += NUM_CORES;
          }
      #ifdef PARALLEL
          ii = n / 2 + NC / 2;
      #endif
  }

int next_inputs = (n + NC - 1) / 2;
#ifdef PARALLEL
  pi_cl_team_barrier();
  i = 2 * core_id;
    if (i < next_inputs / 2 * 2)
      do
#else // NO PARALLEL
  i = 0;
  do
#endif // END PARALLEL
    {
      OUT_TYPE a = output[i];
      OUT_TYPE b = output[i + 1];
      asm volatile("" ::: "memory");
      input[i] = (INP_TYPE)a;
      input[i + 1] = (INP_TYPE)b;
      i += 2 * NUM_CORES;
    } while (i < next_inputs / 2 * 2);
#ifdef PARALLEL
  if (core_id == 0)
#endif
    if (next_inputs & 0x1)
      input[next_inputs - 1] = (INP_TYPE)output[next_inputs - 1];

#endif // ALL NC CASES

#ifdef PARALLEL
  pi_cl_team_barrier();
#endif
}
#endif

int gsl_wavelet_transform(INP_TYPE *data, OUT_TYPE *output, size_t n) // output_dim inserirlo tra i parametri
{
  size_t i;

  size_t output_dim = DWT_LEN_OUT;
  size_t level_dim = n;
  for (i = 0; i < LEVELS; i++)
  {
    size_t input_dim = level_dim;
    level_dim = (level_dim + NC - 1) / 2;
    output_dim -= level_dim;
    asm volatile("" ::: "memory");
    dwt_step(data, output, input_dim, output_dim);
  }

  return 0;
}
#pragma GCC pop_options
