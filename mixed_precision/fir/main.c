#include "pmsis.h"

#include "stats.h"

#include <stdio.h>
#include "config.h"

#include "data.h"

#define STACK_SIZE      1024

double __extendohfdf2(float16alt value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.ah %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}

double __extendhfdf2(float16 value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.h %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}

int retval = 0;


int check_result(OUT_TYPE *x, int r){
  
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif
  
  if(pi_core_id()==0){

    float diff;
    int err = 0;

    for(int i=0; i<r; i++) {
      diff = fabs(x[i] - check[i]);
      if(diff > THR) {
        err++;
        #ifdef VERBOSE
        printf("Error at index %d:\t expected %f\t real %f\t error %f\n", i, check[i], x[i], diff);
        #endif
      }
      #ifdef PRINT_RESULTS
        printf(" at index %d:\t expected %f\t real %f\t error %f\n", i, check[i], x[i], diff);
      #endif

      //printf(" at index %d:\t ref %f\t output %f\t error %f\n", i, check[i], x[i], diff);

    }

    if(err != 0)
      printf("TEST FAILED with %d errors!!\n", err);
    else
      printf("TEST PASSED!!\n");

    return err;
  }
  
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif

}

void main_fn() {

  int i;

  #ifdef STATS
    INIT_STATS();
    PRE_START_STATS();
  #endif
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif
  #ifdef STATS
      START_STATS();
  #endif


  convolve(UnitImpulse, Filter0, ORDER, Buffer0, LENGTH);

  #ifndef FABRIC
  pi_cl_team_barrier();
  #endif
  #ifdef STATS
    STOP_STATS();
  #endif
  
#ifdef CHECK
  retval = check_result(Buffer0, LENGTH-ORDER);
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
  return retval;
}
