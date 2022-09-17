
extern short bit_rev_2_4_LUT[FFT_LEN_RADIX2];
int bit_rev_2_4(int value);

void compute_2_4_LUT()
{
  int j, idx, skip;
  unsigned int index[LOG2_FFT_LEN_RADIX2];
  idx = 0;
  for(j=0; j<FFT_LEN_RADIX2; j++)
  {
    skip = 0;
    index[0] = j;
    // Find a closed set of permutations
    for(int k=0; k<LOG2_FFT_LEN_RADIX2-1; k++)
    {
      index[k+1] = bit_rev_2_4(index[k]);
      if(index[k+1] <= j) { skip=1; break; }
    }
    // Save permutations into the LUT
    if(!skip)
    {
      for(int k=0; k<LOG2_FFT_LEN_RADIX2; k++)
      {
         bit_rev_2_4_LUT[idx++] = index[k];
      }
    }
  }
  // Zero-terminated LUT
  bit_rev_2_4_LUT[idx] = 0;
}
