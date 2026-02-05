#include "pmsis.h"

#include "wavelet.h"
#include "stats.h"
#include "config.h"
#include <stdio.h>

#include "input_ch2_off.h"

#if defined(__GAP9__)
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x)
#endif
#ifndef FABRIC
PI_L2 uint32_t perf_values[ARCHI_CLUSTER_NB_PE];
#else
PI_L2 uint32_t perf_value;
#endif

DATA_LOCATION OUT_TYPE Output_Signal[DWT_LEN_OUT];

#define STACK_SIZE 2048

// End of computation
int done = 0;
int retval = 0;

// #ifndef FP8
// double __attribute__((used)) __extendohfdf2(float16alt value)
// {
//   float result;
//   __asm__ __volatile__("fcvt.s.ah %0, %1" : "=f"(result) : "f"(value) :);
//   return (double)result;
// }

// double __attribute__((used)) __extendhfdf2(float16 value)
// {
//   float result;
//   __asm__ __volatile__("fcvt.s.h %0, %1" : "=f"(result) : "f"(value) :);
//   return (double)result;
// }
// #endif
void check_wavelet_transform();

void main_fn()
{

  check_wavelet_transform();
}

void check_wavelet_transform()
{

  int i, j;

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


  gsl_wavelet_transform((INP_TYPE *)Input_Signal, Output_Signal, DWT_LEN);

  #ifndef FABRIC
  pi_cl_team_barrier();
  #endif

  // End Power mesurement
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

    float diff;
    int err = 0;

    for (int i = 0; i < DWT_LEN_OUT; i++)
    {
      // printf(" at index %d:\t ref %f\t output %f\t error %f\n", i, ref[i], Output_Signal[i], diff);

      diff = fabs(Output_Signal[i] - ref[i]);
      if (diff > THR)
      {
        retval = 1;
        err++;
#ifdef VERBOSE
        printf("Error at index %d:\t ref %f\t output %f\t error %f\n", i, ref[i], Output_Signal[i], diff);
#endif
      }
#ifdef PRINT_RESULTS
      printf("Index %d:\t ref %f\t output %f\t error %f\n", i, ref[i], Output_Signal[i], diff);
#endif
    }

    if (err != 0)
#ifndef VECTORIAL
#ifdef FP32
      printf("FP32 TEST FAILED with %d errors!!\n", err);
#elif defined(FP16)
      printf("FP16 TEST FAILED with %d errors!!\n", err);
#elif defined(FP16ALT)
      printf("FP16ALT TEST FAILED with %d errors!!\n", err);

#elif defined(FP8)
      printf("FP8 TEST FAILED with %d errors!!\n", err);
#endif
#else
#ifdef FP16
      printf("FP16 VEC TEST FAILED with %d errors!!\n", err);
#elif defined(FP16ALT)
      printf("FP16ALT VEC TEST FAILED with %d errors!!\n", err);
#endif
#endif
    else
#ifndef VECTORIAL
#ifdef FP32
      printf("FP32 TEST PASSED!!\n");
#elif defined(FP16)
      printf("FP16 TEST PASSED!!\n");
#elif defined(FP16ALT)
      printf("FP16ALT TEST PASSED!!\n");
#elif defined(FP8)
      printf("FP8 TEST PASSED!!\n");
#endif
#else
#ifdef FP16
      printf("FP16 VEC TEST PASSED!!\n");
#elif defined(FP16ALT)
      printf("FP16ALT VEC TEST PASSED!!\n");
#endif
#endif

    return err;
  }
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
  int desc[2] = {(int)cluster_entry, 0};

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
return retval;
}
