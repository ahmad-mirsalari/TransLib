#include "pmsis.h"
#include <stdio.h>
#include <stdlib.h>

#define STACK_SIZE      1024

//#define ICACHE_CTRL_UNIT 0x10201400 
//#define ICACHE_PREFETCH ICACHE_CTRL_UNIT + 0x18 

// Cluster entry point

int kmeans();
int retval = 0;

void main_fn() {
  retval = kmeans();
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

  // [OPTIONAL] specify the stack size for the task
  #if !defined(__GAP9__)
    cluster_task.stack_size = STACK_SIZE;

  #endif
    cluster_task.slave_stack_size = STACK_SIZE;

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
    #if defined(__GAP9__)
  pi_time_wait_us(10000);

  pi_freq_set(PI_FREQ_DOMAIN_FC, 240*1000*1000);
  
  pi_time_wait_us(10000);
  
  pi_freq_set(PI_FREQ_DOMAIN_CL, 240*1000*1000);
  
  pi_time_wait_us(10000);
  #endif

  #ifdef FABRIC
      retval = main_fn();
  #else
    if (exec_cluster(cluster_entry))
      return -1;
  #endif
  return retval;
}
