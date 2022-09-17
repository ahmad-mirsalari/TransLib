#include "pmsis.h"
#include <stdio.h>
#include <math.h>
#include "wavelet.h"
#include "config.h"
#include "stats.h"

#include "kernels.def" //filters

#pragma GCC push_options
#if (NUM_CORES>1) && (NC > 2)
#pragma GCC optimize ("no-tree-ch")
#pragma GCC optimize ("no-tree-loop-im")
#endif

#define MADD_F16(c, a, b, temp) \
__asm__ __volatile__ ("fmul.h %0, %1, %2": "=f"(temp): "f"(a), "f"(b) :); \
__asm__ __volatile__ ("fadd.h %0, %1, %2": "=f"(c): "f"(c), "f"(temp) :);


#ifdef PARALLEL
DATA_LOCATION volatile DATA_TYPE sum_core[NUM_CORES];
#endif

#ifdef VECTORIAL
void dwt_step (DATA_TYPE *input, DATA_TYPE *output, size_t n, size_t idx_level) //change name of idx_level?
{
  size_t i, ii;
  int j;
  VDTYPE temp,temp_n;
  #ifdef PARALLEL
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif
  ii = 0;
  DATA_TYPE h, g, h2, g2;
  VDTYPE h_v, g_v, h2_v, g2_v,in,in1;
  size_t n_it;

  #if NC == 2
    temp[0] = R2_2;
    temp[1] = R2_2;
    temp_n[0] = R2_2;
    temp_n[1] = -R2_2;
    n_it = (n/4)*4;
    size_t n_rest = n - n_it; // n_rest = n%4
    #ifdef PARALLEL
    for (i = 4*core_id; i < n_it; i += NUM_CORES*4) //loop unrolling (without it was *2)
    //i = 4*core_id;
    //if(i < n_it)
    //do
    #else
    for (i = 0; i < n_it; i += 4) //loop unrolling (without it was +=2, for the downsampling)
    //i = 0;
    //if(n_it > 0)
    //do
    #endif
    {
      //h = R2_2 * input[i] + R2_2 * input[i+1]; //approssimation coeff (reversed low filter Lo)
       in = *((VDTYPE*)&input[i]);
      h_v = in * temp;

      //h2 = R2_2 * input[i+2] + R2_2 * input[i+3];
       in1 = *((VDTYPE*)&input[i+2]);
      h2_v = in1 * temp;

      //g = R2_2 * input[i] - R2_2 * input[i+1]; //detail coeff (reversed high filter Hi)
      g_v = in * temp_n;

      //g2 = R2_2 * input[i+2] - R2_2 *input[i+3];
      g2_v = in1 * temp_n;
      //output[ii + core_id * 2] = h;           //following levels input
      output[ii + core_id * 2] = h_v[0] + h_v[1];           //following levels input


      //output[ii + core_id * 2 + idx_level] = g; //final output
      output[ii + core_id * 2 + idx_level] = g_v[0] + g_v[1];
      //output[ii + 1 + core_id*2] = h2
      output[ii + 1 + core_id*2] = h2_v[0] + h2_v[1];
      //output[ii + 1 + core_id * 2 + idx_level] = g2;
      output[ii + 1 + core_id * 2 + idx_level] = g2_v[0] + g2_v[1];


      ii += NUM_CORES*2;
    } ///while (i < n_it);
    h = h_v[0] + h_v[1];
    h2 = h2_v[0] + h2_v[1];
    g2 = g2_v[0] + g2_v[1];
    g = g_v[0] + g_v[1];
    ii = n/4*2; //following levels input size (ii=n_it/2, for the downsampling)

    #ifdef PARALLEL
    if(core_id == 0)
    {
    #endif


    if (n_rest > 0)
    {
      switch(n_rest)
      {
        case 1:
          g = h = R2_2 * input[n-1];
          break;
        case 2:
          //h = R2_2 * input[n - 2] + R2_2 * input[n - 1];
               in1 = *((VDTYPE*)&input[i-2]);
              h_v = in1 * temp;
              h = h_v[0] + h_v[1];
          //g = R2_2 * input[n - 2] - R2_2 * input[n - 1];
              g_v = in1 * temp_n;
              g = g_v[0] + g_v[1];
          break;
        default:
          //h  = R2_2 * input[n-3] + R2_2 * input[n-2];
           in1 = *((VDTYPE*)&input[i-3]);
          h_v = in1 * temp;
          h = h_v[0] + h_v[1];
          h2 = R2_2 * input[n-1];
          //g  = R2_2 * input[n-3] - R2_2 * input[n-2];
          g_v = in1 * temp_n;
          g = g_v[0] + g_v[1];
          output[ii+1] = h2;
          output[ii+1+idx_level]  = h2;
      }
      output[ii] = h;
      output[ii+idx_level] = g;
    }

    #ifdef PARALLEL
    } // core 0
    pi_cl_team_barrier();
    #endif


    int next_inputs = (n + NC - 1) / 2;
    #ifdef PARALLEL
    for (i = 2*core_id; i < next_inputs/2 * 2; i += 2*NUM_CORES)
    #else
    for (i = 0; i < next_inputs/2 * 2; i += 2)
    #endif
    {
      DATA_TYPE a = output[i];
      DATA_TYPE b = output[i+1];
      //in = *((VDTYPE*)&output[i]);
      asm volatile("":::"memory");
      input[i] =a; //in[0];
      input[i+1] =b; //in[1];
    }
    #ifdef PARALLEL
    if(core_id == 0)
    #endif
    if(next_inputs & 0x1)
      input[next_inputs-1] = output[next_inputs-1];;


  #else //NC > 4

    #ifdef PARALLEL
    for(i = core_id; i < NC/2-1; i += NUM_CORES)
    #else
    for(i = 0; i < NC/2 - 1; i++)
    #endif
    {

      VDTYPE a = (VDTYPE){0, 0};
      VDTYPE b = (VDTYPE){0, 0};
;
      for(j = 2 *(i+1)-1 ; j>=1; j-=2) //j<2*(i+1)=NC/2-1 above respect to i and here respect to j;
      {
      //DATA_TYPE in = input[2*(i+1)-j-1];
        VDTYPE in = *((VDTYPE*)&input[2*(i+1)-j-1]);
        //DATA_TYPE lo = Lo[j];
        VDTYPE lo = *((VDTYPE*)&Lo[(NC-1)-j]);
        //DATA_TYPE hi = Hi[j];
        VDTYPE hi = *((VDTYPE*)&Hi[(NC-1)-j]);
        asm volatile("":::"memory");
        a += in * lo;  //cA next levels, reversed Lo
        b += in * hi;  //cD final output, rev Hi (in matlab is the 2nd array, i.e. [lo hi]=wavefilters('db4'))
      }
      output[ii+core_id] = a[0]+a[1];
      output[ii+core_id+idx_level] = b[0]+b[1];
     ii += NUM_CORES;
    }
    #ifdef PARALLEL
    ii = NC/2-1;
    #endif
    //middle and final part of the array
    #ifdef PARALLEL
    for (i = (NC-1)+core_id*2; i < n; i += NUM_CORES*2)
    #else
    for (i = NC-1; i < n; i += 2)
    #endif
    {
      //DATA_TYPE a = 0.0f;
      //DATA_TYPE b = 0.0f;
      VDTYPE a = (VDTYPE){0, 0};
      VDTYPE b = (VDTYPE){0, 0};
      for(j=(NC-1) ; j >=1; j-=2)
      {
        //DATA_TYPE in = input[i-j];
        //DATA_TYPE lo = Lo[j];
        //DATA_TYPE hi = Hi[j];
        VDTYPE in = *((VDTYPE*)&input[i-j]);
        VDTYPE lo = *((VDTYPE*)&Lo[(NC-1) - j]);
        VDTYPE hi = *((VDTYPE*)&Hi[(NC-1) -j]);
        asm volatile("":::"memory");
        a += in * lo;
        b += in * hi;
      }

      if (NC & 0x00000001) 
       {
        DATA_TYPE in = input[i-0];
        DATA_TYPE lo = Lo[(NC-1) - 0];
        DATA_TYPE hi = Hi[(NC-1) - 0];
        asm volatile("":::"memory");

        a[0] += in * lo;
        b[0] += in * hi;
       }
      output[ii+core_id] = a[0]+a[1];
      output[ii+core_id+idx_level] = b[0]+b[1];
    ii += NUM_CORES;
    }


    #ifdef PARALLEL
    ii = n/2;
    #endif
    if(n%2==0) //even
    {
      #ifdef PARALLEL
      for(i = core_id; i < NC/2-1; i += NUM_CORES)
      #else
      for(i = 0; i < NC/2-1; i++)
      #endif
      {
        //DATA_TYPE a = 0.0f;
        //DATA_TYPE b = 0.0f;
        VDTYPE a = (VDTYPE){0, 0};
        VDTYPE b = (VDTYPE){0, 0};
        for(j = NC-2*(i+1) -1; j >=1; j-=2)
        {
          //DATA_TYPE in = input[n-j-1] ;
          //DATA_TYPE lo = Lo[2*(i+1)+j];
          //DATA_TYPE hi = Hi[2*(i+1)+j];

          VDTYPE in = *((VDTYPE*)&input[n-j-1]);
          VDTYPE lo = *((VDTYPE*)&Lo[(NC -1) - (2*(i+1)+j)]);
          VDTYPE hi = *((VDTYPE*)&Hi[(NC -1)-(2*(i+1)+j)]);
          asm volatile("":::"memory");
          a += in * lo;
          b += in * hi;
        } //while(j < NC-2*(i+1));

      if (NC & 0x00000001) 
       {
        //printf("I am here in $$$$$$$\n");
        DATA_TYPE in = input[i-0];
        DATA_TYPE lo = Lo[(NC-1) - (2*(i+1))];
        DATA_TYPE hi = Hi[(NC-1) - (2*(i+1))];
        asm volatile("":::"memory");

        a[0] += in * lo;
        b[0] += in * hi;
       }
        output[ii+core_id] = a[0] + a[1];
        output[ii+core_id+idx_level] = b[0] + b[1]; 
        ii+=NUM_CORES;
      }
      #ifdef PARALLEL
      ii = n/2 + (NC/2-1);
      #endif
    }
    else //odd
    {
      #ifdef PARALLEL
      for(i = core_id; i < NC/2; i += NUM_CORES)
      #else
      for(i = 0; i < NC/2; i++)
      #endif
      {
        //DATA_TYPE a = 0.0f;
        //DATA_TYPE b = 0.0f;
        VDTYPE a = (VDTYPE){0, 0};
        VDTYPE b = (VDTYPE){0, 0};
        for(j = NC-2*(i+1); j >=1; j-=2)

        {
          //DATA_TYPE in = input[n-j-1];
          //DATA_TYPE lo = Lo[2*(i+1)+j-1];
          //DATA_TYPE hi = Hi[2*(i+1)+j-1];

          VDTYPE in = *((VDTYPE*)&input[n-j-1]);
          VDTYPE lo = *((VDTYPE*)&Lo[(NC -1) -(2*(i+1)+j-1)]);
          VDTYPE hi = *((VDTYPE*)&Hi[(NC -1) -(2*(i+1)+j-1)]);

          asm volatile("":::"memory");
          a += in * lo;
          b += in * hi;
        } //while(j < NC-2*(i+1)+1);
      if (NC & 0x00000001) 
       {
       }
       else{
                
        DATA_TYPE in = input[n-0-1];
        DATA_TYPE lo = Lo[(NC -1) -(2*(i+1)+0-1)];
        DATA_TYPE hi = Hi[(NC -1) -(2*(i+1)+0-1)];
        asm volatile("":::"memory");

        a[0] += in * lo;
        b[0] += in * hi;
       }
        output[ii+core_id] = a[0] + a[1];
        output[ii+core_id+idx_level] = b[0] +b[1];
        ii += NUM_CORES;
      }
      #ifdef PARALLEL
      ii = n/2 + NC/2;
      #endif
    }

    int next_inputs = (n + NC - 1) / 2;
    #ifdef PARALLEL
    pi_cl_team_barrier();

    //for (i = 2*core_id; i < next_inputs/2 * 2; i += 2*NUM_CORES) //instead of i<ii because in parallel, ii are not execute in the same way by all cores
    i = 2*core_id;
    if (i < next_inputs/2 * 2)
    do
    #else
    //for (i = 0; i < next_inputs/2 * 2; i+=2)
    i = 0;
    do
    #endif
    {
      DATA_TYPE a = output[i];
      DATA_TYPE b = output[i+1];

      //VDTYPE a = *((VDTYPE*)&output[i]);
      //*((VDTYPE*)&input[n-j-1]);
      //VDTYPE b = *((VDTYPE*)&output[i+2]);
      asm volatile("":::"memory");
      input[i] = a;
      input[i+1] = b;
      /*input[i] = a[0];
      input[i+1] = a[1];
      input[i+2] = b[0];
      input[i+3] = b[1];*/
      //VDTYPE *Vout = (VDTYPE*)&input[i];
      i += 2*NUM_CORES;
      //*Vout =  {temp1[0] + temp1[1], temp2[0] + temp2[1]};
    } while(i < next_inputs/2 * 2);
    #ifdef PARALLEL
    if(core_id == 0)
    #endif
    if(next_inputs & 0x1)
      input[next_inputs-1] = output[next_inputs-1];
    #endif //ALL NC CASES

    #ifdef PARALLEL
    pi_cl_team_barrier();
    #endif
}

#else
void dwt_step (DATA_TYPE *input, DATA_TYPE *output, size_t n, size_t idx_level) //change name of idx_level?
{
  size_t i, ii, j;

  #ifdef PARALLEL
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif
  ii = 0;
  DATA_TYPE h, g, h2, g2;
  size_t n_it;

  #if NC == 2

    n_it = (n/4)*4;
    size_t n_rest = n - n_it; // n_rest = n%4

    #ifdef PARALLEL
    //for (i = 4*core_id; i < n_it; i += NUM_CORES*4) //loop unrolling (without it was *2)
    i = 4*core_id;
    if(i < n_it)
    do
    #else
    //for (i = 0; i < n_it; i += 4) //loop unrolling (without it was +=2, for the downsampling)
    i = 0;
    if(n_it > 0)
    do
    #endif
    {
      h = R2_2 * input[i] + R2_2 * input[i+1]; //approssimation coeff (reversed low filter Lo)
      h2 = R2_2 * input[i+2] + R2_2 * input[i+3];
      g = R2_2 * input[i] - R2_2 * input[i+1]; //detail coeff (reversed high filter Hi)
      g2 = R2_2 * input[i+2] - R2_2 *input[i+3];
      output[ii + core_id * 2] = h;           //following levels input
      output[ii + core_id * 2 + idx_level] = g; //final output
     output[ii + 1 + core_id*2] = h2;
      output[ii + 1 + core_id * 2 + idx_level] = g2;


      ii += NUM_CORES*2;
      i += NUM_CORES*4;
    } while (i < n_it);

    ii = n/4*2; //following levels input size (ii=n_it/2, for the downsampling)

    #ifdef PARALLEL
    if(core_id == 0)
    {
    #endif


    if (n_rest > 0)
    {
      switch(n_rest)
      {
        case 1:
          g = h = R2_2 * input[n-1];
          break;
        case 2:
          h = R2_2 * input[n - 2] + R2_2 * input[n - 1];
          g = R2_2 * input[n - 2] - R2_2 * input[n - 1];
          break;
        default:
          h  = R2_2 * input[n-3] + R2_2 * input[n-2];
          h2 = R2_2 * input[n-1];
          g  = R2_2 * input[n-3] - R2_2 * input[n-2];
          output[ii+1] = h2;
          output[ii+1+idx_level]  = h2;
      }
      output[ii] = h;
      output[ii+idx_level] = g;
    }

    #ifdef PARALLEL
    } // core 0
    pi_cl_team_barrier();
    #endif


    int next_inputs = (n + NC - 1) / 2;
    #ifdef PARALLEL
    for (i = 2*core_id; i < next_inputs/2 * 2; i += 2*NUM_CORES)
    #else
    for (i = 0; i < next_inputs/2 * 2; i += 2)
    #endif
    {
      DATA_TYPE a = output[i];
      DATA_TYPE b = output[i+1];
      asm volatile("":::"memory");
      input[i] = a;
      input[i+1] = b;
    }
    #ifdef PARALLEL
    if(core_id == 0)
    #endif
    if(next_inputs & 0x1)
      input[next_inputs-1] = output[next_inputs-1];;


  #else //NC > 4
    DATA_TYPE temp;
    #ifdef PARALLEL
    for(i = core_id; i < NC/2-1; i += NUM_CORES)
    #else
    for(i = 0; i < NC/2 - 1; i++)
    #endif
    {
      // printf("[%d] / %d\n", core_id, NC/2-1);
      DATA_TYPE a = 0.0f;
      DATA_TYPE b = 0.0f;
      //for(j = 0; j < 2*(i+1); j++) //j<2*(i+1)=NC/2-1 above respect to i and here respect to j;
      j = 0;
      do
      {
        //beginning part of each layer
        DATA_TYPE in = input[2*(i+1)-j-1];
        DATA_TYPE lo = Lo[j];
        DATA_TYPE hi = Hi[j];
        asm volatile("":::"memory");

        a += in * lo;  //cA next levels, reversed Lo
        b += in * hi;  //cD final output, rev Hi (in matlab is the 2nd array, i.e. [lo hi]=wavefilters('db4'))
        j++;
      } while(j < 2*(i+1));
      output[ii+core_id] = a;
      output[ii+core_id+idx_level] = b;

      
      ii += NUM_CORES;
    }
    #ifdef PARALLEL
    ii = NC/2-1;
    #endif

    //middle and final part of the array
    #ifdef PARALLEL
    for (i = (NC-1)+core_id*2; i < n; i += NUM_CORES*2)
    #else
    for (i = NC-1; i < n; i += 2)
    #endif
    {
      DATA_TYPE a = 0.0f;
      DATA_TYPE b = 0.0f;
      for(j=0; j<NC; j++)
      {
        DATA_TYPE in = input[i-j];
        DATA_TYPE lo = Lo[j];
        DATA_TYPE hi = Hi[j];

        asm volatile("":::"memory");
        a += in * lo;
        b += in * hi;
      }
      output[ii+core_id] = a;
      output[ii+core_id+idx_level] = b;

      ii += NUM_CORES;
    }
    #ifdef PARALLEL
    ii = n/2;
    #endif
    if(n%2==0) //even
    {
      #ifdef PARALLEL
      for(i = core_id; i < NC/2-1; i += NUM_CORES)
      #else
      for(i = 0; i < NC/2-1; i++)
      #endif
      {
        DATA_TYPE a = 0.0f;
        DATA_TYPE b = 0.0f;
        //for(j = 0; j < NC-2*(i+1); j++)
        j=0;
        do
        {
          DATA_TYPE in = input[n-j-1] ;
          DATA_TYPE lo = Lo[2*(i+1)+j];
          DATA_TYPE hi = Hi[2*(i+1)+j];
          asm volatile("":::"memory");
          a += in * lo;
          b += in * hi;
          j++;
        } while(j < NC-2*(i+1));
        output[ii+core_id] = a;
        output[ii+core_id+idx_level] = b;
        ii+=NUM_CORES;
      }
      #ifdef PARALLEL
      ii = n/2 + (NC/2-1);
      #endif
    }
    else //odd
    {
      #ifdef PARALLEL
      for(i = core_id; i < NC/2; i += NUM_CORES)
      #else
      for(i = 0; i < NC/2; i++)
      #endif
      {
        DATA_TYPE a = 0.0f;
        DATA_TYPE b = 0.0f;
        //for(j = 0; j < NC-2*(i+1)+1; j++)
        j = 0;
        do
        {
          DATA_TYPE in = input[n-j-1];
          DATA_TYPE lo = Lo[2*(i+1)+j-1];
          DATA_TYPE hi = Hi[2*(i+1)+j-1];
          asm volatile("":::"memory");
          a += in * lo;
          b += in * hi;
          j++;
        } while(j < NC-2*(i+1)+1);
        output[ii+core_id] = a;
        output[ii+core_id+idx_level] = b;
        ii += NUM_CORES;
      }
      #ifdef PARALLEL
      ii = n/2 + NC/2;
      #endif
    }

    int next_inputs = (n + NC - 1) / 2;
    #ifdef PARALLEL
    pi_cl_team_barrier();

    //for (i = 2*core_id; i < next_inputs/2 * 2; i += 2*NUM_CORES) //instead of i<ii because in parallel, ii are not execute in the same way by all cores
    i = 2*core_id;
    if (i < next_inputs/2 * 2)
    do
    #else
    //for (i = 0; i < next_inputs/2 * 2; i+=2)
    i = 0;
    do
    #endif
    {
      DATA_TYPE a = output[i];
      DATA_TYPE b = output[i+1];
      asm volatile("":::"memory");
      input[i] = a;
      input[i+1] = b;
      i += 2*NUM_CORES;
    } while(i < next_inputs/2 * 2);
    #ifdef PARALLEL
    if(core_id == 0)
    #endif
    if(next_inputs & 0x1)
      input[next_inputs-1] = output[next_inputs-1];

    #endif //ALL NC CASES

    #ifdef PARALLEL
    pi_cl_team_barrier();
    #endif
}
#endif

int gsl_wavelet_transform(DATA_TYPE *data, DATA_TYPE *output, size_t n) //output_dim inserirlo tra i parametri
{
  size_t i;

  size_t output_dim = DWT_LEN_OUT;
  size_t level_dim = n;
  for (i = 0; i < LEVELS; i++)
  {
    size_t input_dim = level_dim;
    level_dim = (level_dim + NC -1)/2;
    output_dim -= level_dim;
    asm volatile("":::"memory");
    dwt_step (data, output, input_dim, output_dim);

  }
  
  return 0;
}
#pragma GCC pop_options
