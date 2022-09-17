#else // NUM_CORES > 1, NO VECTORIZATION

// PARALLEL VERSION (no vectorization)
//extern FLOAT C_r[64];
//extern FLOAT D_r[64];
void __attribute__((noinline))
filter(FLOAT input_data[], FLOAT output_data[], FLOAT a[], FLOAT b[], int n, int n_taps)
{
 //int  BLOCKS_PAR = ((n+NUM_CORES-1)/NUM_CORES);
//int  LENGTH_PAR = (BLOCKS_PAR*NUM_CORES);
    int blockSize = (n+NUM_CORES-1)/NUM_CORES;
    int  LENGTH_PAR = (blockSize*NUM_CORES);
  int start = pi_core_id()*blockSize;
  int end = start + blockSize < n? start + blockSize : n;
  int i, j, k;
  int core_id = pi_core_id();
 printf("start is %d, end is %d\n", start, end);
  for(i = start; i<end; i+=1)
  {
    FLOAT val;
    val = b[0] * input_data[i];
    //printf(" i is %d,  bcoef0 %f,inputdata %d is %f, val %f\n",i,b[0],i,input_data[i], val);
    for(j=1; j<n_taps; j+=2)
    {
      val += b[j] * input_data[i-j];
       //printf(" i-j is %d and bj is %f and inputdata is %f and mult is %f val is %f\n" ,i - j, b[j], input_data[i - j],  b[j] * input_data[i - j], val);
    val += b[j+1] * input_data[i-j-1];
      //printf(" i-j is %d and aj is %f and mult is %f and val is %f\n" ,i - j, b[j+1], b[j+1] * input_data[i - j-1], val);
    }
    output_data[i] = val;
    printf("val [%d] is %f \n", i,val);
  }
  pi_cl_team_barrier();

  for(i = 0; i<LENGTH_PAR; i+=8)
  {
    FLOAT val;
    
    j = core_id;
#if NUM_CORES < 8
    FLOAT temp[8];
    //printf("j is %d \n", j);
    volatile int iterations = 8/NUM_CORES;
    for(int c=0; c<iterations; c++)
#elif NUM_CORES == 16
    if(core_id < 8)
#endif
    {
      int idx = j*8;
      //printf(" j is %d and idx is %d \n", j, idx);
      val  = D_r[idx]   * output_data[i];
      val += D_r[idx+1] * output_data[i+1];
      val += D_r[idx+2] * output_data[i+2];
      val += D_r[idx+3] * output_data[i+3];
      val += D_r[idx+4] * output_data[i+4];
      val += D_r[idx+5] * output_data[i+5];
      val += D_r[idx+6] * output_data[i+6];
      val += D_r[idx+7] * output_data[i+7];

      asm volatile ("":::"memory");

      val += C_r[idx]   * output_data[i-1];
      val += C_r[idx+1] * output_data[i-2];
      val += C_r[idx+2] * output_data[i-3];
      val += C_r[idx+3] * output_data[i-4];
      val += C_r[idx+4] * output_data[i-5];
      val += C_r[idx+5] * output_data[i-6];
      val += C_r[idx+6] * output_data[i-7];
      val += C_r[idx+7] * output_data[i-8];
      val += C_r[idx+8] * output_data[i-9];
      val += C_r[idx+9] * output_data[i-10];

      asm volatile ("":::"memory");

#if NUM_CORES < 8
      temp[j] = val;
      j+=NUM_CORES;
#endif

    }
    printf("val [%d] is %f \n", i,val);
    printf("temp [%d] is %f \n", core_id,temp[core_id]);
    pi_cl_team_barrier();

#if NUM_CORES == 16
    if(core_id < 8)
      output_data[i+core_id] = val;
#elif NUM_CORES == 8
    output_data[i+core_id] = val;
#elif NUM_CORES == 4
    output_data[i+core_id]   = temp[core_id];
    output_data[i+core_id+4] = temp[core_id+4];
#elif NUM_CORES == 2
    output_data[i+core_id]   = temp[core_id];
    output_data[i+core_id+2] = temp[core_id+2];
    output_data[i+core_id+4] = temp[core_id+4];
    output_data[i+core_id+6] = temp[core_id+6];
#elif num_cores == 1
    output_data[i+core_id]   = temp[core_id];
    output_data[i+core_id+1] = temp[core_id+1];
    output_data[i+core_id+2] = temp[core_id+2];
    output_data[i+core_id+3] = temp[core_id+3];
    output_data[i+core_id+4] = temp[core_id+4];
    output_data[i+core_id+5] = temp[core_id+5];
    output_data[i+core_id+6] = temp[core_id+6];
    output_data[i+core_id+7] = temp[core_id+7];
#else
    #error NUM_CORES is not valid
#endif
  
  }

}
#endif // VECTORIZATION

#endif
