#include <stdlib.h>
#include <stdio.h>

#include "pthread_launch.h"
#include "spmm.h"
#include "spad.h"
#include "bind_defs.h"
#include "group_templates.h"


// actual kernel
void kernel(
     float *A_val, float *B_val, float *C_val, int *A_idx, int *A_ptr, int m, int k, int ptid)
{

  // start recording all stats (all cores)
  // use the last thread, b/c this wakes up last?
  /*if (ptid == 0 )
  {
    stats_on();
  }*/


  
  // need to wait after transpose!
  pthread_barrier_wait(&start_barrier);

  // if (cinfo.used == 0) return;


  MOVE_STACK_ONTO_SCRATCHPAD();

  spmm_manycore(A_val, B_val, C_val, A_idx, A_ptr, m, k, ptid);

  RECOVER_DRAM_STACK();
}

// helper functions
Kern_Args *construct_args(float *A_val, float *B_val, float *C_val, int *A_idx, int *A_ptr, int m,
     int k, int ptid)
{

  Kern_Args *args = (Kern_Args *)malloc(sizeof(Kern_Args));

  args->A_val = A_val;
  args->B_val = B_val;
  args->C_val = C_val;
  args->A_idx = A_idx;
  args->A_ptr = A_ptr;
  args->m = m;
  args->k = k;
  args->ptid = ptid;

  return args;
}

void *pthread_kernel(void *args)
{
  // guarentee one thread goes to each core, by preventing any threads
  // from finishing early

  pthread_barrier_wait(&start_barrier);

  // call the spmd kernel
  Kern_Args *a = (Kern_Args *)args;

  kernel(a->A_val, a->B_val, a->C_val, a->A_idx, a->A_ptr, a->m, a->k, a->ptid);

  // reset scratchpad config
  SET_PREFETCH_MASK(0, 0, &start_barrier);

  /*if (a->ptid == 0)
  {
    stats_off();
  }*/

  return NULL;
}
