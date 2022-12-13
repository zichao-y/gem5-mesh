#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <stdatomic.h>

#include "spad.h"
#include "pthread_launch.h"
#include "spmm.h"
#include "util.h"
#include "data.h"
#include "bind_defs.h"




atomic_int workq;
atomic_int write_q;
atomic_int val_q;
atomic_int idx_q;
atomic_int compen[256];
int ini_acc_val=0;


float absVal(float a)
{
	if(a < 0)
	{
		return (a * -1);
	}
   	else
	{ 
		return a;
	}
}


#define SMALL_FLOAT_VAL 0.00000001f
float percentDiff(float val1, float val2)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		return 0.0f;
	}

	else
	{
    		return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
} 

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
// ret 1 if fail
// ret 0 if sucess
int polybenchCompare(float val1, float val2) {
  return (percentDiff(val1, val2) > PERCENT_DIFF_ERROR_THRESHOLD);
}




int main(int argc, char *argv[])
{

  /*--------------------------------------------------------------------
   * Setup scratchpads
   *------------------------------------------------------------------*/

  printf("=================================================   starting spmm! ===============================================================\n");

  initScratchpads();

  //printf("init scratchpads!\n");

  /*--------------------------------------------------------------------
  * Get info about manycore
  *-------------------------------------------------------------------*/

  //int cores_x, cores_y;
  //int num_cores = get_dimensions(&cores_x, &cores_y);
  int cores_x = 1;
  int cores_y = 1;
  if (argc > 1)
      cores_x = atoi(argv[1]);
  if (argc > 2)
      cores_y = atoi(argv[2]);
  int num_cores = cores_x * cores_y;
  
  


  
  /*--------------------------------------------------------------------
  * Pack argument for kernel
  *-------------------------------------------------------------------*/

  
  int* C_pointer_ptr;
  float *C_inter_val_ptr;
  int *C_inter_idx_ptr;
  int *C_inter_len_ptr;
  int *C_valout_tmp_ptr;
  int *C_idxout_tmp_ptr;
  float *C_val_ptr;
  int *C_idx_ptr;
  
  float *C_val_extend_ptr;
  int *C_idx_extend_ptr;
  

  float* C_val = (float*)malloc_cache_aligned(sizeof(float),matC_nnz,(void**)&C_val_ptr);
  int* C_index = (int*)malloc_cache_aligned(sizeof(int),matC_nnz,(void**)&C_idx_ptr);
  int* C_pointer= (int*)malloc_cache_aligned(sizeof(int),mat_m+1,(void**)&C_pointer_ptr);
  C_pointer[0] = 0;
  

  float** C_inter_val_pointer = (float**)malloc_cache_aligned(sizeof(float),mat_m,(void**)&C_inter_val_ptr);
  int** C_inter_idx_pointer = (int**)malloc_cache_aligned(sizeof(int),mat_m,(void**)&C_inter_idx_ptr);
  float* C_valout_tmp = (float*)malloc_cache_aligned(sizeof(float),mat_m * 2*QUEUE_SIZE,(void**)&C_valout_tmp_ptr);
  int* C_idxout_tmp = (int*)malloc_cache_aligned(sizeof(int),mat_m * 2*QUEUE_SIZE,(void**)&C_idxout_tmp_ptr);
  //printf("C_inter_val is: %d, C_inter_idx is: %d\n",C_inter_val,C_inter_idx);
  //int* C_inter_len = (int*)malloc_cache_aligned(sizeof(int),mat_m,(void*)&C_inter_len_ptr);

  //malloc a space for intermediate results written
  float* C_val_extend = (float*)malloc_cache_aligned(sizeof(float),matC_nnz,(void**)&C_val_ptr);
  int* C_index_extend = (int*)malloc_cache_aligned(sizeof(int),matC_nnz,(void**)&C_idx_ptr);

  //int ini_acc_val=0;
  //printf("come to line 131\n");

  //allocate helper queues in DRAM
  int QUEUE_SIZE_DRAM = (mat_k-QUEUE_SIZE) > 0 ? mat_k-QUEUE_SIZE : 1;
  //printf("QUEUE_SIZE_DRAM: %d\n",QUEUE_SIZE_DRAM);
  float *helper_queue_valarray_ptr;
  int *helper_queue_idxarray_ptr;
  //float** helper_queue_valarray = (float**)malloc_cache_aligned(2*sizeof(float),num_cores,(void**)&helper_queue_valarray_ptr);
  //int** helper_queue_idxarray = (int**)malloc_cache_aligned(2*sizeof(int),num_cores,(void**)&helper_queue_idxarray_ptr);
  float** helper_queue_valarray = (float**)malloc(sizeof(float*)*num_cores);
  int** helper_queue_idxarray = (int**)malloc(sizeof(int*)*num_cores);
  //printf("allocate helper queue array, helper_queue_valarray:%d, helper_queue_idxarray:%d\n",helper_queue_valarray,helper_queue_idxarray);
  for(int core_idx=0; core_idx<num_cores; core_idx++){
    float * val_queue_DRAM_ptr; 
    float* val_queue_DRAM = (float*)malloc_cache_aligned(sizeof(float),QUEUE_SIZE_DRAM*3,(void**)&val_queue_DRAM_ptr);
    int * idx_queue_DRAM_ptr; 
    int* idx_queue_DRAM = (int*)malloc_cache_aligned(sizeof(int),QUEUE_SIZE_DRAM*3,(void**)&idx_queue_DRAM_ptr);
    helper_queue_valarray[core_idx] = val_queue_DRAM;
    helper_queue_idxarray[core_idx] = idx_queue_DRAM;
    //printf("allocate helper queue for core[%d],helper_queue_valarray:%d, helper_queue_idxarray:%d\n",core_idx,val_queue_DRAM,idx_queue_DRAM);
  }
  //printf("come to line 149\n");
  

  // initialize the arguments to send to each device core
  Kern_Args **kern_args = (Kern_Args **)malloc(sizeof(Kern_Args *) * num_cores);

  for (int y = 0; y < cores_y; y++)
  {
    for (int x = 0; x < cores_x; x++)
    {
      int i = x + y * cores_x;
    
      kern_args[i] = construct_args(valA, valB, C_val,C_inter_val_pointer,C_inter_idx_pointer, idxA, ptrA, idxB, ptrB, C_index, C_pointer, mat_m, mat_n, mat_k, i,num_cores,C_valout_tmp,C_idxout_tmp,helper_queue_valarray,helper_queue_idxarray,C_val_extend,C_index_extend);
    }
  }

  /*--------------------------------------------------------------------
  * Run the kernel
  *-------------------------------------------------------------------*/

  printf("Begin kernel on %d cores\n", num_cores);
  printf("Cores x:%d Cores y:%d\n", cores_x, cores_y);
  launch_kernel(pthread_kernel, (void **)kern_args, cores_x, cores_y);

/*--------------------------------------------------------------------
  * Check result and cleanup data
  *-------------------------------------------------------------------*/
  //printf("returned VAL_C pointer is: %d\n",C_kernel);
  int error=0;
  //float *C_val = C_kernel[0];
  //int *C_idx = C_index;
  /*for(int i=0;i<matC_nnz;i++){
    if(C_val[i]!=valC[i]){
      printf("VAL missmatch found in valC[%d]! Expect: %f, returned: %f\n",i,valC[i],C_val[i]);
      error +=1;
    }
  }
  for(int i=0;i<matC_nnz;i++){
    if(C_index[i]!=idxC[i]){
      printf("IDX missmatch found in idxC[%d]! Expect: %d, returned: %d\n",i,idxC[i],C_index[i]);
      error +=1;
    }
  }*/
  for(int i=1;i<mat_m+1;i++){
    if(C_pointer[i]!=ptrC[i]){
      printf("Pointer missmatch found in ptrC[%d]! Expect: %d, returned: %d\n",i,ptrC[i],C_pointer[i]);
      error +=1;
    }
  }
  if (error >0){
    printf("total error detected: %d\n",error);
    return 1;
  }

  printf("[[SUCCESS]]\n");

  
  
  //free(C_pointer_ptr);
  //free(C_inter_val_ptr);
  //free(C_inter_idx_ptr);
  

  return 0;
}
