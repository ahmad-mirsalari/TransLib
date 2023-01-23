// License, Version 0.51 (the "License"); you may not use this file except in
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pmsis.h"
#include "config.h"
#include "stats.h"

#include <stdio.h>
#include <stdint.h>
#include <limits.h> /* for CHAR_BIT */

#include "data.h"

#define STACK_SIZE 2048

DATA_LOCATION MA_TYPE matA[M*N] __attribute__ ((aligned (4)));
DATA_LOCATION MB_TYPE matB[N*P] __attribute__ ((aligned (4)));
DATA_LOCATION OUT_TYPE matC[M*P] __attribute__ ((aligned (4)));

// End of computation
int done = 0;

int retval = 0;

void __attribute__ ((noinline)) matrix_init(MA_TYPE * __restrict__ A, MB_TYPE * __restrict__ B, OUT_TYPE * __restrict__ C) {
  for (int i = 0; i < M; i++) 
    for (int j = 0; j < N; j++){
      A[i*N+j] = A_mat[i*N+j];


    } 
      
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < P; j++){
      B[i*P+j] = B_mat[i*P+j];
    }
  for (int i = 0; i < M; i++) 
    for (int j = 0; j < P; j++)  
      C[i*P+j] = 0;
  
}

int __attribute ((noinline)) check_result(OUT_TYPE * __restrict__ result) {
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif

  if(pi_core_id() == 0) {
    float diff;
    int err = 0;

    for (int i = 0; i < (M*P); i++) {
      diff = fabs(result[i] - ref[i]);
      if(diff > THR) {
        err++;
      #ifdef VERBOSE

        printf("Error at index %d:\t refrence %f\t output %f\t error %.4f\n", i, ref[i], result[i], diff);
      #endif
      
      }

      #ifdef PRINT_RESULTS

        printf("index %d:\t refrence %f\t output %f\t error %f\n", i, ref[i], result[i], diff);
      #endif
    }
  
    if(err != 0)
      printf("TEST FAILED with %d errors!!\n", err);
    else
      printf("TEST PASSED!!\n");

    return err;

  }
}

void main_fn(){

  matrix_init(matA, matB, matC);
 
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif
  
  #ifdef STATS
  INIT_STATS();

  PRE_START_STATS();
  START_STATS();
  #endif
  matMul(matA, matB, matC, M, N, P);

  #ifdef STATS
  STOP_STATS();
  #endif

  #ifdef CHECK
  retval = check_result(matC);
  #endif
};

#ifndef FABRIC
static int cluster_entry(void *arg)
{
  pi_cl_team_fork(NUM_CORES, main_fn, (void *)0x0);

  return 0;
}

static void end_of_call(void *arg)
{
  done = 1;
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
