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


#if defined(__GAP9__)
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x)
#endif
#ifndef FABRIC
PI_L2 uint32_t perf_values[ARCHI_CLUSTER_NB_PE];
#else
PI_L2 uint32_t perf_value;
#endif

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

  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif

  // Init input data
  #ifndef FABRIC
  if(pi_core_id() == 0) 
  #endif
  {
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

  uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
  
   // Performance measurement
   #ifdef STATS 
   #if !defined(__GAP9__)
   INIT_STATS();
   PRE_START_STATS();
   START_STATS();
   #else
   pi_perf_conf(1 << PI_PERF_ACTIVE_CYCLES); // PI_PERF_INSTR
   pi_perf_reset();
   pi_perf_start();
   #endif
   #endif
   // Start Power mesurement
   #if defined(__GAP9__)
   pi_pad_function_set(GPIOs, 1);
   pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
   pi_gpio_pin_write(GPIOs, 0);
   WRITE_GPIO(0);
   
   WRITE_GPIO(1);
   #endif


  #if NUM_CORES>1
  par_fft_mixed_2_8(Input_Signal, Buffer_Signal_Out);
  #else
  fft_mixed_2_8(Input_Signal, Buffer_Signal_Out);
  #endif

  #ifndef FABRIC
  pi_cl_team_barrier();
  #endif
  // End Power measurement
  #if defined(__GAP9__) 
  WRITE_GPIO(0);
  #endif
  // Stop performance measurement
  #ifdef STATS
  #if !defined(__GAP9__)
  STOP_STATS();
  #else
  pi_perf_stop();

  #ifndef FABRIC
  perf_values[core_id] = pi_perf_read(PI_PERF_ACTIVE_CYCLES); // PI_PERF_CYCLES
  #else
  perf_value = pi_perf_read(PI_PERF_ACTIVE_CYCLES); // PI_PERF_CYCLES
  #endif
  // uint32_t cycles = pi_perf_read(PI_PERF_CYCLES);
  #endif
  #endif

  #ifdef CHECK
  #ifndef FABRIC
  if(pi_core_id() == 0) 
  #endif
  {
    float real_error = 0;
    float imag_error = 0;
    float diff =0;
    for(i=0;i<FFT_LEN;i++)
    {

      #ifdef PRINT_RESULTS
      printf("Output_Signal[%d] = (",i);
      printf("%f",Buffer_Signal_Out[i].re);
      printf(", ");
      printf("%f",Buffer_Signal_Out[i].im);

      printf("\tref=");
      printf("%f",ref[i].re);
      printf(", ");
      printf("%f",ref[i].im);

      printf("\tr_error =");
      #endif
      diff = fabs(ref[i].re - Buffer_Signal_Out[i].re);
      real_error += diff;
      #ifdef PRINT_RESULTS
      printf("%f ",diff);
      //printf(")\n");
      printf("\ti_error =");
      #endif
      diff = fabs(ref[i].im - Buffer_Signal_Out[i].im);
      imag_error += diff;
      #ifdef PRINT_RESULTS
      printf("%f \n",diff);
      #endif
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
  #if !defined(__GAP9__)
    cluster_task.stack_size = STACK_SIZE;

  #endif
    cluster_task.slave_stack_size = STACK_SIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  pi_cluster_close(&cluster_dev);

  return desc[1];
}
#endif

int main()
{

  #if defined(__GAP9__)
  pi_time_wait_us(10000);

  pi_freq_set(PI_FREQ_DOMAIN_FC, 240*1000*1000);
  
  pi_time_wait_us(10000);
  
  pi_freq_set(PI_FREQ_DOMAIN_CL, 240*1000*1000);
  
  pi_time_wait_us(10000);
  #endif

 #ifdef FABRIC
      main_fn();
  #else
    if (exec_cluster(cluster_entry))
      return -1;
  #endif


    // Print the performance values if GAP9 is used
  #ifdef STATS
  #if defined(__GAP9__)
  #ifndef FABRIC
  for (uint32_t i = 0; i < ARCHI_CLUSTER_NB_PE; i++)
  {
      printf("[%d] Perf : %d cycles\n", i, perf_values[i]);
  }
  #else
      printf("Perf : %d cycles\n", perf_value);
  #endif
  #endif
  #endif
  return 0;
}
