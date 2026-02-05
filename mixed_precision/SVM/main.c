#include "pmsis.h"
#include <stdio.h>
#include <stdlib.h>
#include "plp_SVM_predict.h"
#include "stats.h"
#include "defines.h"
#include "modelSVM.h"

// #ifndef FP8
// double __attribute__((used)) __extendohfdf2(float16alt value)
// {
//     float result;
//     __asm__ __volatile__("fcvt.s.ah %0, %1" : "=f"(result) : "f"(value) :);
//     return (double)result;
// }

// double __attribute__((used)) __extendhfdf2(float16 value)
// {
//     float result;
//     __asm__ __volatile__("fcvt.s.h %0, %1" : "=f"(result) : "f"(value) :);
//     return (double)result;
// }

// float16 __attribute__((used)) __truncdfhf2(double value)
// {
//     float16 result;
//     float temp = (float)value;
//     __asm__ __volatile__("fcvt.h.s %0, %1" : "=f"(result) : "f"(temp) :);
//     return result;
// }

// float16alt __attribute__((used)) __truncdfohf2(double value)
// {
//     float16alt result;
//     float temp = (float)value;
//     __asm__ __volatile__("fcvt.ah.s %0, %1" : "=f"(result) : "f"(temp) :);
//     return result;
// }
// #endif

DATA_LOCATION int predictions[N_DEC_VALUES_];
DATA_LOCATION int miss_classifications = 0;

#if defined(__GAP9__)
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x)
#endif
#ifndef FABRIC
PI_L2 uint32_t perf_values[ARCHI_CLUSTER_NB_PE];
#else
PI_L2 uint32_t perf_value;
#endif

void main_fn()
{

  svm_model model_par;

  model_par.KERNEL_TYPE = KERNEL_TYPE_;
  model_par.GAMMA1 = GAMMA1_;
  model_par.SVS = SVS_;
  model_par.COEF_DIM = COEF_DIM_;
  model_par.F_DIM = F_DIM_;
  model_par.N_CLASS = N_CLASS_;
  model_par.N_DEC_VALUES = N_DEC_VALUES_;
  int kernel_type = model_par.KERNEL_TYPE;

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

  // switch (kernel_type)
  // {
  // case 0:
  //   plp_SVM_linear(model_par, data_model, predictions, sv_coef, bias);
  //   break;
  // case 2:
  //   plp_SVM_RBF(model_par, data_model, sv_coef, bias, X_ref, predictions);
  // }

  #if KERNEL_TYPE_ == 2
  plp_SVM_RBF(model_par, data_model, sv_coef, bias, X_ref, predictions);

  #elif KERNEL_TYPE_ == 0
  plp_SVM_linear(model_par, data_model, predictions, sv_coef, bias);
  #else
  print("Unsupported kernel type");
  #endif
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
  float accuracy = 0.0f;
  #ifndef FABRIC
    if(pi_core_id() == 0) 
  #endif
    {
    for (int j = 0; j < model_par.SVS; j++)
    {

      #ifdef PRINT_RESULTS
      printf("indx: %d\t class: %d (%d)\n", j, predictions[j], check_result[j]);
      #endif
      if (predictions[j] != check_result[j])
      {
        miss_classifications++;
        #ifdef PRINT_RESULTS
          printf("Error at indx: %d\t class: %d (%d)\n", j, predictions[j], check_result[j]);
        #endif
      }


    }
    accuracy = (100.0f * (model_par.SVS - miss_classifications)) / model_par.SVS;
    if (ACCURACY_ == accuracy )
      printf("Classification accuracy of C code: %.2f/100.00 == %.2f/100.00 of Golden model\n", accuracy, ACCURACY_);
    else if (ACCURACY_ - 1.0 < accuracy && accuracy < ACCURACY_ + 1.0)
    {
      printf(" C Accuracy is in acceptable range (%.2f%% <= %.2f%% <= %.2f%%)\n", (ACCURACY_ - 1.0), accuracy, (ACCURACY_ + 1.0));
    }
    else
    {
      printf("Error: Accuracy is not as expected (%.2f%% != %.2f%%) \n", accuracy, ACCURACY_);
      pmsis_exit(-1);
    }
  }
#endif
}

#ifndef FABRIC
static int cluster_entry(void *arg)
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
