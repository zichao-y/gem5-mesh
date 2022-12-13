#include <stdlib.h>
#include <stdio.h>

#include "pthread_launch.h"
#include "spmm.h"
#include "spad.h"
#include "bind_defs.h"
#include "group_templates.h"


// actual kernel
void kernel(
     const float *A_val, const float *B_val, float *C_val, const int *A_idx, const int *A_ptr, int m, int n, int k, int mat_nnz, int ptid, int num_cores, int *wr_mask)
{

  // start recording all stats (all cores)
  // use the last thread, b/c this wakes up last?
  if (ptid == 0 )
  {
    

  #ifdef CACHE_WARM

      float tempB, tempC, tempA;
      int tempAi, tempAp;
      for(int i=0; i<k*n; i++){
        if (B_val[i] > tempB)
          tempB = B_val[i]; 
      }

      for(int i=0; i<k*m; i++){
        if (C_val[i] > tempC)
          tempC = C_val[i];
      }

      for(int i=0; i<mat_nnz; i++){
        if (A_val[i] > tempA){
          tempA = A_val[i];
          tempAi = A_idx[i]; 
        }
      }

      for(int i=0; i<m+1; i++){
        if (A_ptr[i] > tempAp)
          tempAp = A_ptr[i];
      }
      printf("tempA: %f, tempB: %f, tempAi: %d, tempAp: %d", tempA, tempB, tempAi, tempAp);

  #endif

    stats_on();
  }

  PREFETCH_EPOCH((1 << PREFETCH_NUM_REGION_SHAMT) | (64 << PREFETCH_REGION_SIZE_SHAMT));
  
  // need to wait after transpose!
  pthread_barrier_wait(&start_barrier);

  // if (cinfo.used == 0) return;


  MOVE_STACK_ONTO_SCRATCHPAD();

  spmm_manycore(A_val, B_val, C_val, A_idx, A_ptr, m, n, k, ptid, num_cores,wr_mask);

  RECOVER_DRAM_STACK();
}

// helper functions
Kern_Args *construct_args(const float *A_val, const float *B_val, float *C_val, const int *A_idx, const int *A_ptr, int m, int n,
     int k, int mat_nnz, int ptid, int num_cores, int *wr_mask)
{

  Kern_Args *args = (Kern_Args *)malloc(sizeof(Kern_Args));

  args->A_val = A_val;
  args->B_val = B_val;
  args->C_val = C_val;
  args->A_idx = A_idx;
  args->A_ptr = A_ptr;
  args->m = m;
  args->m = n;
  args->k = k;
  args->mat_nnz = mat_nnz;
  args->ptid = ptid;
  args->num_cores = num_cores;
  args->wr_mask = wr_mask;

  return args;
}

void *pthread_kernel(void *args)
{
  // guarentee one thread goes to each core, by preventing any threads
  // from finishing early

  pthread_barrier_wait(&start_barrier);

  // call the spmd kernel
  Kern_Args *a = (Kern_Args *)args;

  kernel(a->A_val, a->B_val, a->C_val, a->A_idx, a->A_ptr, a->m, a->n, a->k, a->mat_nnz, a->ptid, a->num_cores, a->wr_mask);

  // reset scratchpad config
  SET_PREFETCH_MASK(0, 0, &start_barrier);

  if (a->ptid == 0)
  {
    stats_off();
  }

  return NULL;
}
