// Copyright 2017 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pmsis.h"
#include "convolution.h"
#include "stats.h"

#include "data.h"
#define STACK_SIZE 2048

#if defined(__GAP9__)
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x)
#endif
#ifndef FABRIC
PI_L2 uint32_t perf_values[ARCHI_CLUSTER_NB_PE];
#else
PI_L2 uint32_t perf_value;
#endif

DATA_LOCATION OUT_TYPE Out[OUT_DIM];

int retval = 0;

int __attribute((noinline)) check_result(OUT_TYPE *__restrict__ result, int SIZE)
{
#ifndef FABRIC
  pi_cl_team_barrier();
#endif

#ifndef FABRIC
  if (pi_core_id() == 0)
#endif
  {
    float diff;
    int err = 0;

    for (int i = 0; i < SIZE; i++)
    {
      diff = fabs(result[i] - ref[i]);
      // printf(" index %d:\t expected %f\t real %f\t error %f\n", i, ref[i], result[i], diff);
      if (diff > THR)
      {
        err++;
#ifdef VERBOSE
        printf("Error at index %d:\t expected %f\t real %f\t error %f\n", i, ref[i], result[i], diff);
#endif
      }

#ifdef PRINT_RESULTS
      printf("at index %d:\t expected %f\t real %f\t error %f\n", i, ref[i], result[i], diff);
#endif
    }

    if (err != 0)
      printf("TEST FAILED with %d errors!!\n", err);
    else
      printf("TEST PASSED!!\n");

    return err;
  }
}

void __attribute__((noinline)) InitZero(OUT_TYPE *__restrict__ Img, int size)
{
  int i;

  for (i = 0; i < size; i++)
    Img[i] = 0;
}

void __attribute__((noinline)) InitOne(INP_TYPE *__restrict__ Img, int size)
{
  int i;

  for (i = 0; i < size; i++)
    Img[i] = 1.0f;
}

void main_fn()
{

  uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();



  #ifndef FABRIC
    if (pi_core_id() == 0)
  #endif
    {
      InitZero(Out, OUT_DIM);

      // InitOne(In_Img, IMG_DIM);
    }
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif

  // Start Performance mesurement
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

#ifdef VECTORIAL
  if (FILT_WIN == 5)
  {
    Conv5x5_Vector(In_Img, Out, OUT_ROW, OUT_COL, INP_COL, Filter_Kern);
  }
  else
  {
    Conv3x3_Vector(In_Img, Out, OUT_ROW, OUT_COL, INP_COL, Filter_Kern);
  }

#else
  Conv_Scalar(In_Img, Out, Filter_Kern, OUT_ROW, OUT_COL, STRIDE, INP_COL, FILT_WIN);
#endif

#ifndef FABRIC
  pi_cl_team_barrier();
#endif

// End Power mesurement
#if defined(__GAP9__) 
WRITE_GPIO(0);
#endif

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
  retval = check_result(Out, OUT_DIM);
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
