#include "pmsis.h"
#include <stdio.h>

#include "fft_mixed.h"
#include "data_signal.h"
#include "data_out.h"

#define STACK_SIZE      1024
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

#include "stats.h"




// End of computation
int done = 0;
int retval = 0;

extern Complex_type twiddle_factors[FFT_LEN_RADIX2/2];
#ifdef FULL_TWIDDLES
extern Complex_type twiddle_factors8[FFT_LEN_RADIX8];
#endif
extern short bit_rev_2_8_LUT[FFT_LEN];
void compute_twiddles();
void compute_full_twiddles();
void compute_2_8_LUT();
int  bit_rev_2_8(int value);
int  bit_rev_radix8(int value);
extern short bit_rev_radix8_LUT[FFT_LEN_RADIX8];

void main_fn() {
  int i;

  #ifdef SORT_OUTPUT
  Complex_type * Buffer_Signal_Out = Input_Signal;
  #endif

  #ifdef STATS
    INIT_STATS();
    PRE_START_STATS();
  #endif
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif

  // Init input data
  if(pi_core_id()==0) {
    compute_twiddles();
    #ifdef FULL_TWIDDLES
    compute_full_twiddles();
    #endif

    #if defined(BITREV_LUT) && !defined(RADIX_8) 
    #ifndef SORT_OUTPUT
    for (i=0; i<FFT_LEN; i++)
      bit_rev_2_8_LUT[i] = bit_rev_2_8(i);
    #else
    compute_2_8_LUT();
    #endif
    #endif

    #if defined(BITREV_LUT) && defined(RADIX_8) 
    for (i=0; i<FFT_LEN_RADIX8;i++)
      bit_rev_radix8_LUT[i] = bit_rev_radix8(i);
    printf("bit_rev_radix8 %d \n", bit_rev_radix8_LUT[1]);
    #endif

    #ifndef SORT_OUTPUT
    for(i=0;i<FFT_LEN;i++) {
      Buffer_Signal_Out[i] = (Complex_type){0.0f, 0.0f};
    }
    #endif
  }

  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif
  #ifdef STATS
    START_STATS();
  #endif

  #if NUM_CORES>1
  par_fft_mixed_2_8(Input_Signal, Buffer_Signal_Out);
  #else
  fft_mixed_2_8(Input_Signal, Buffer_Signal_Out);
  #endif

  #ifndef FABRIC
  pi_cl_team_barrier();
  #endif
  #ifdef STATS
    STOP_STATS();
  #endif

  #ifdef PRINT_RESULTS
  if(pi_core_id()==0){
    float real_error = 0;
    float imag_error = 0;
    float diff =0;
    for(i=0;i<FFT_LEN;i++)
    {

      printf("Output_Signal[%d] = (",i);
      printf("%f",Buffer_Signal_Out[i].re);
      printf(", ");
      printf("%f",Buffer_Signal_Out[i].im);

      printf("\tref=");
      printf("%f",ref[i].re);
      printf(", ");
      printf("%f",ref[i].im);

      printf("\tr_error =");
      diff = fabs(ref[i].re - Buffer_Signal_Out[i].re);
      real_error += diff;
      printf("%f ",diff);
      //printf(")\n");
      printf("\ti_error =");
      diff = fabs(ref[i].im - Buffer_Signal_Out[i].im);
      imag_error += diff;
      printf("%f \n",diff);
      //printf(")\n");
    }
    printf("Real error = ");
    printf("%f",real_error/FFT_LEN);
    printf("\n");
    printf("Imag error = ");
    printf("%f",imag_error/FFT_LEN);
    printf("\n");
  }
  #ifndef FABRIC
  pi_cl_team_barrier();
  #endif
  #endif
}

#ifndef FABRIC
static int cluster_entry()
{
  pi_cl_team_fork(NUM_CORES, main_fn, (void *)0x0);

  return 0;
}

static void exec_cluster_stub(void *arg)
{
  int *desc = arg;
  int (*entry)() = (int (*)())desc[0];
  desc[1] = entry();
}

static int exec_cluster(int (*cluster_entry)())
{
  int desc[2] = { (int)cluster_entry, 0 };

  struct pi_device cluster_dev;
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task;

  pi_cluster_conf_init(&conf);

  pi_open_from_conf(&cluster_dev, &conf);

  pi_cluster_open(&cluster_dev);

  pi_cluster_task(&cluster_task, exec_cluster_stub, desc);

    // [OPTIONAL] specify the stack size for the task
  cluster_task.stack_size = STACK_SIZE;
  cluster_task.slave_stack_size = STACK_SIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  pi_cluster_close(&cluster_dev);

  return desc[1];
}
#endif

int main()
{

 #ifdef FABRIC
      main_fn();
  #else
    if (exec_cluster(cluster_entry))
      return -1;
  #endif
  return 0;
}
