#include <stdlib.h>
#include <stdio.h>

#include "pthread_launch.h"
#include "spmm.h"
#include "spad.h"
#include "bind_defs.h"
#include "group_templates.h"


// actual kernel
void kernel(
     const float *A_val, const float *B_val, float *C_val, float **C_inter_val, int **C_inter_idx, const int *A_idx, const int *A_ptr, const int *B_idx, const int *B_ptr, int *C_idx, int *C_ptr,int m, int n, int k, int ptid, int num_cores, float *C_valout_tmp, int *C_idxout_tmp,float** helper_queue_valarray,int** helper_queue_idxarray, float *C_val_extend, int *C_idx_extend)
{

  // start recording all stats (all cores)
  // use the last thread, b/c this wakes up last?
  if (ptid == 0 )
  {

    stats_on();
  }

  PREFETCH_EPOCH((1 << PREFETCH_NUM_REGION_SHAMT) | (3840 << PREFETCH_REGION_SIZE_SHAMT));
  
  // need to wait after transpose!
  //pthread_barrier_wait(&start_barrier);

  // if (cinfo.used == 0) return;

  //printf("core %d enter kernel!\n",ptid);
  //MOVE_STACK_ONTO_SCRATCHPAD();

  spmm_manycore(A_val, B_val, C_val, C_inter_val, C_inter_idx, A_idx, A_ptr, B_idx, B_ptr, C_idx, C_ptr,m, n, k, ptid, num_cores, C_valout_tmp, C_idxout_tmp,helper_queue_valarray,helper_queue_idxarray,C_val_extend, C_idx_extend);
  /*pthread_barrier_wait(&start_barrier);
  if(ptid==0){
    int c_start=0;
    for(int i=0;i<m;i++){
        // printf("%d,\n",i);

      for(int j=0;j<row_len;j++){


        // printf("%d,\n",i);
        C_val[c_start+j]=C_inter_val[i*QUEUE_SIZE+j];
  
        C_idx[c_start+j]=C_inter_idx[i*QUEUE_SIZE+j];
        // if((c_start+j)<100){
        // printf("c_idx %d: %d, %d\n",i,C_idx[c_start+j],C_idx_orig[c_start+j]);

        // }
      }
      C_ptr[i]=c_start;
      c_start+=row_len;
      // if((C_ptr[i]!=C_ptr_orig[i])){

      //   printf("c_ptr %d: %d, %d\n",i,C_ptr[i],C_ptr_orig[i]);
      // }
    }
  }*/
  //RECOVER_DRAM_STACK();
}

// helper functions
Kern_Args *construct_args(const float *A_val, const float *B_val, float *C_val, float **C_inter_val, int **C_inter_idx, const int *A_idx, const int *A_ptr, const int *B_idx, const int *B_ptr , int *C_idx, int *C_ptr, int m, int n, int k, int ptid, int num_cores, float *C_valout_tmp, int *C_idxout_tmp,float** helper_queue_valarray,int** helper_queue_idxarray,float *C_val_extend, int *C_idx_extend)
{

  Kern_Args *args = (Kern_Args *)malloc(sizeof(Kern_Args));

  args->A_val = A_val;
  args->B_val = B_val;
  args->C_val = C_val;
  args->A_idx = A_idx;
  args->A_ptr = A_ptr;
  args->B_idx = B_idx;
  args->B_ptr = B_ptr;
  args->C_idx = C_idx;
  args->C_ptr = C_ptr;
  

  args->C_inter_val = C_inter_val;
  args->C_inter_idx = C_inter_idx;
  //args->C_inter_len = C_inter_len;
  args->m = m;
  args->n = n;
  args->k = k;
  args->ptid = ptid;
  args->num_cores = num_cores;
  //args->ini_acc_val = ini_acc_val;
  args->C_valout_tmp = C_valout_tmp;
  args->C_idxout_tmp = C_idxout_tmp;
  args->helper_queue_idxarray = helper_queue_idxarray;
  args->helper_queue_valarray = helper_queue_valarray;
  args->C_val_extend = C_val_extend;
  args->C_idx_extend = C_idx_extend;

  return args;
}


void *pthread_kernel(void *args)
{
  // guarentee one thread goes to each core, by preventing any threads
  // from finishing early

  pthread_barrier_wait(&start_barrier);

  // call the spmd kernel
  Kern_Args *a = (Kern_Args *)args;

  kernel(a->A_val, a->B_val, a->C_val, a->C_inter_val, a->C_inter_idx, a->A_idx, a->A_ptr, a->B_idx, a->B_ptr, a->C_idx, a->C_ptr, a->m, a->n, a->k, a->ptid, a->num_cores, a->C_valout_tmp, a->C_idxout_tmp, a->helper_queue_valarray,a->helper_queue_idxarray,a->C_val_extend, a->C_idx_extend);

  // reset scratchpad config
  SET_PREFETCH_MASK(0, 0, &start_barrier);

  if (a->ptid == 0)
  {
    stats_off();
  }

  return NULL;
}
