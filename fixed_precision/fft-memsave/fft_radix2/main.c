#include "rt/rt_api.h"
#include "pulp.h"
#include <stdio.h>
#include "print_float.h"
#include "fft.h"
#include "data_signal.h"

#define STACK_SIZE      1024
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

#include "stats.h"

// End of computation
int done = 0;


void compute_twiddles();
int bit_rev_radix2(int);
extern Complex_type twiddle_factors[FFT_LEN_RADIX2/2];
extern short bit_rev_radix2_LUT[FFT_LEN_RADIX2];

void main_fn()
{
  int i;
  INIT_STATS();


  // Init input data
  if(get_core_id()==0) {

    compute_twiddles();

    #ifdef BITREV_LUT
      for (i=0; i<FFT_LEN_RADIX2;i++)
        bit_rev_radix2_LUT[i] = bit_rev_radix2(i);
    #endif

    #ifndef SORT_OUTPUT
    for(i=0;i<FFT_LEN_RADIX2;i++) {
      Buffer_Signal_Out[i] = (Complex_type){0.0f, 0.0f};
    }
    #endif
  }

  #ifdef SORT_OUTPUT
  Complex_type * Buffer_Signal_Out = Input_Signal;
  #endif

  synch_barrier();

  ENTER_LOOP_STATS();
  START_STATS();
  #ifdef PARALLEL
  par_fft_radix2(Input_Signal, Buffer_Signal_Out);
  #else
  fft_radix2(Input_Signal, Buffer_Signal_Out);
  #endif
  STOP_STATS();
  EXIT_LOOP_STATS();

  synch_barrier();

  #ifdef PRINT_RESULTS
  if(get_core_id()==0) {
    float real_acc = 0;
    float imag_acc = 0;
    for(i=0;i<FFT_LEN_RADIX2;i++)
    {
      printf("Output_Signal[%d] = (",i);
      printFloat(Buffer_Signal_Out[i].re);
      printf(", ");
      printFloat(Buffer_Signal_Out[i].im);
      printf(")\n");
      real_acc+=Buffer_Signal_Out[i].re;
      imag_acc+=Buffer_Signal_Out[i].im;
    }
    printf("Real Acc = ");
    printFloat(real_acc);
    printf("\n");
    printf("Imag Acc = ");
    printFloat(imag_acc);
    printf("\n");
  }
  synch_barrier();
  #endif
}

#ifndef FABRIC
static void cluster_entry(void *arg)
{
  rt_team_fork(NUM_CORES, main_fn, (void *)0x0);
}
#endif

static void end_of_call(void *arg)
{
  done = 1;
}


//RT_L1_DATA char stacks[STACK_SIZE*NUM_CORES];

int main()
{
#ifdef FABRIC
  main_fn();
#else
  rt_event_sched_t * psched = rt_event_internal_sched();
  if (rt_event_alloc(psched, 4)) return -1;

  rt_cluster_mount(MOUNT, CID, 0, NULL);

  void *stacks = rt_alloc(RT_ALLOC_CL_DATA, STACK_SIZE*rt_nb_pe());
  if (stacks == NULL) return -1;

  rt_cluster_call(NULL, CID, cluster_entry, NULL, stacks, STACK_SIZE, STACK_SIZE, 0, rt_event_get(psched, end_of_call, (void *) CID));

  while(!done)
    rt_event_execute(psched, 1);

  rt_cluster_mount(UNMOUNT, CID, 0, NULL);
#endif

  return 0;
}
