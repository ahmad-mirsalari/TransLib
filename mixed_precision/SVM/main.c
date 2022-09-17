#include "pmsis.h"
#include <stdio.h>
#include <stdlib.h>
#include "plp_SVM_predict.h"
#include "stats.h"
#include "defines.h"
#include "modelSVM.h"


double __attribute__ ((used)) __extendohfdf2(float16alt value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.ah %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}

double __attribute__ ((used))  __extendhfdf2(float16 value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.h %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}

DATA_LOCATION int predictions[N_DEC_VALUES_];  
DATA_LOCATION int miss_classifications = 0;



void main_fn()
{

  svm_model model_par;

  model_par.KERNEL_TYPE = KERNEL_TYPE_;
  model_par.GAMMA1  = GAMMA1_;
  model_par.SVS     = SVS_;
  model_par.COEF_DIM  = COEF_DIM_;
  model_par.F_DIM     = F_DIM_;
  model_par.N_CLASS   = N_CLASS_;
  model_par.N_DEC_VALUES =  N_DEC_VALUES_; 
  int kernel_type = model_par.KERNEL_TYPE;

  #ifdef STATS
      INIT_STATS();
      PRE_START_STATS();
      START_STATS();
  #endif
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif
  switch(kernel_type)
    {
      case 0:
           plp_SVM_linear (model_par,data_model,predictions,sv_coef, bias);
           break;
      case 2:
            plp_SVM_RBF(model_par,data_model,sv_coef, bias, X_ref,predictions);
          }
      #ifdef STATS
      STOP_STATS();
      #endif
      #ifndef FABRIC
        pi_cl_team_barrier();
      #endif

#ifdef CHECK
  if(pi_core_id() == 0){
            for (int j =0; j<model_par.SVS; j++)
            {
              //printf("indx: %d\t class: %d (%d)\n", j, predictions[j], check_result[j]);
              if(predictions[j] != check_result[j]) {
                  miss_classifications++;
                  printf("Error at indx: %d\t class: %d (%d)\n", j, predictions[j], check_result[j]);
                }


              #ifdef PRINT_RESULTS

                printf("indx: %d\t class: %d (%d)\n", j, predictions[j], check_result[j]);
              #endif
            }
              
            
    printf("%.2f/100.00 of accuracy in classification\n", (100.0f*(model_par.SVS - miss_classifications))/model_par.SVS);
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
  int desc[2] = { (int)cluster_entry, 0 };

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
  return 0;
}
