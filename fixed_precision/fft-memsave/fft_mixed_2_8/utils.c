#include "pmsis.h"

#include "fft_mixed.h" 
#ifdef FP32
#define THR 0.001f
#elif defined(FP16)
#define THR 0.00004f
#elif defined(FP16ALT)
#define THR 0.000157f
#endif

extern short bit_rev_2_8_LUT[FFT_LEN_RADIX2];
int bit_rev_2_8(int value);

#ifdef STAGE_2_2_8
#define MAX_PERM  6
DATA_LOCATION int bitrev_perm_count[4];
DATA_LOCATION int bitrev_perm_steps[] = {6, 4, 3, 2};
#else
#define MAX_PERM  12
DATA_LOCATION int bitrev_perm_count[5];
DATA_LOCATION int bitrev_perm_steps[] = {12, 6, 4, 3, 2};
#endif

void compute_2_8_LUT()
{
  int j, idx, skip, k, l, cnt;
  unsigned int index[MAX_PERM+1];

  idx = 0;
  for(int c = 0; c < sizeof(bitrev_perm_steps)/sizeof(bitrev_perm_steps[0]); c++)
  {
    cnt = 0;
    for(j=0; j<FFT_LEN_RADIX2; j++)
    {
      skip = 0;
      index[0] = j;
      //printf(">>> %d\n", j);
      // Find a closed set of permutations
      for(k=0; k<MAX_PERM; k++)
      {
        index[k+1] = bit_rev_2_8(index[k]);
       //printf("  *  %d\n", index[k+1]);
        if(index[k+1] == j) { break; }
        if(index[k+1] <  j) { skip=1; break; }
      }
      // Save permutations of length bitrev_perm_steps[c] into the LUT
      if(!skip && k == bitrev_perm_steps[c] - 1)
      {
         cnt++;
         for(l=0; l<=k; l++)
         {
           //printf("%5d ", index[l]);
           bit_rev_2_8_LUT[idx++] = index[l];
         }
         //printf("\n");
      }
    }
    bitrev_perm_count[c] = cnt;
    //printf("###### %d\n", cnt);
  }
  bit_rev_2_8_LUT[idx] = 0;
}
