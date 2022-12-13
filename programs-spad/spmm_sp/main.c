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


// #define PRINT_OUT

// #define TRANSPOSE

/*void read_file_float(char* filename, float* buffer, char* mat_sz_m,char* mat_sz_n,char* mat_sz_k, char* density_str) {
  FILE* file = fopen(filename, "r");
  char* line_buf;
  size_t line_buf_size = 0;
  ssize_t line_size;
  int line_cnt = 0;
  // check whether the file canbe open
  if (!file) {
    printf("Can't open file: '%s'", filename);
  }
  // get the first line 
  line_size = getline(&line_buf, &line_buf_size, file);
  while (line_size >= 0) {
    if (line_cnt == 0) {
      for (int i = 0; i < PARA_SZ; i++) {
        mat_sz_m[i] = line_buf[i];
      }
    }
    else if (line_cnt == 1) {
      for (int i = 0; i < PARA_SZ; i++) {
        mat_sz_n[i] = line_buf[i];
      }
    }
    else if (line_cnt == 2) {
      for (int i = 0; i < PARA_SZ; i++) {
        mat_sz_k[i] = line_buf[i];
      }
    }
    else if (line_cnt == 3) {
      for (int i = 0; i < PARA_SZ; i++) {
        density_str[i] = line_buf[i];
        //printf("current lincnt is %d, while line buf is: %s",line_cnt,line_buf);
      }
    }
    buffer[line_cnt] = atof(line_buf);
    line_cnt++;
    // Show the line details 
    //printf("line[%06d]: chars=%06zd, buf size=%06zu, contents: %s", line_cnt, line_size, line_buf_size, line_buf);

    line_size = getline(&line_buf, &line_buf_size, file);
  }

  free(line_buf);
  line_buf = NULL;  
  // close the file
  fclose(file);
}*/

/*void read_file_int(char* filename, int* buffer) {
  FILE* file = fopen(filename, "r");
  char* line_buf;
  size_t line_buf_size = 0;
  ssize_t line_size;
  int line_cnt = 0;
  // check whether the file canbe open
  if (!file) {
    printf("Can't open file: '%s'", filename);
  }
  // get the first line 
  line_size = getline(&line_buf, &line_buf_size, file);
  while (line_size >= 0) {
    buffer[line_cnt] = atoi(line_buf);
    line_cnt++;
    // Show the line details 
    //printf("line[%06d]: chars=%06zd, buf size=%06zu, contents: %s", line_cnt, line_size, line_buf_size, line_buf);

    // Get the next line 
    line_size = getline(&line_buf, &line_buf_size, file);
  }
  
  free(line_buf);
  line_buf = NULL;  
  // close the file
  fclose(file);
}

void read_value_float(char* filename, float* buffer) {
  FILE* file = fopen(filename, "r");
  char* line_buf;
  size_t line_buf_size = 0;
  ssize_t line_size;
  int line_cnt = 0;
  // check whether the file canbe open
  if (!file) {
    printf("Can't open file: '%s'", filename);
  }
  // get the first line 
  line_size = getline(&line_buf, &line_buf_size, file);
  while (line_size >= 0) {
    buffer[line_cnt] = atof(line_buf);
    line_cnt++;
    // Show the line details 
    if(line_cnt % 1000 == 0)
      printf("line[%06d]: chars=%06zd, buf size=%06zu, contents: %s", line_cnt, line_size, line_buf_size, line_buf);

    // Get the next line 
    line_size = getline(&line_buf, &line_buf_size, file);
  }
  
  free(line_buf);
  line_buf = NULL;  
  // close the file
  fclose(file);
}*/

atomic_int workq;


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
  int wr_complete[64];
  for(int i=0; i<64; i++){
    wr_complete[i] = 0;
  }
  /*--------------------------------------------------------------------
  * Data initialization
  *-------------------------------------------------------------------*/
  /*int debug = 0;
  int debug_2 =0;
  int synthetic_mat = 1; 
  int m,n,k,nnz;
  float den;
  char* filename_base ;
  if(synthetic_mat) 
      filename_base = "./data/synthetic/";
  else
      filename_base = "./data/cora/";
  char filename_info[FILENAME_SZ];
  // concat the string
  strcpy(filename_info, filename_base);
  strcat(filename_info, "spm_tb_info.dat");
  FILE* file = fopen(filename_info, "r");
  char* line_buf;
  size_t line_buf_size = 0;
  ssize_t line_size;
  int line_cnt = 0;
  // check whether the file canbe open
  if (!file) {
    printf("Can't open file: '%s'", filename_info);
  }
  // get the first line 
  line_size = getline(&line_buf, &line_buf_size, file);
  while (line_size >= 0) {
    //printf("current lincnt is %d, while line buf is: %s",line_cnt,line_buf);
    if(line_cnt == 0){
      m = atoi(line_buf);
    }
    if(line_cnt == 1){
      n = atoi(line_buf);
    }
    if(line_cnt == 2){
      k = atoi(line_buf);
    }
    if(line_cnt == 3){
      den = atof(line_buf);
    }
    if(line_cnt == 4){
      nnz = atoi(line_buf);
    }

    line_cnt++;
    // Show the line details 
    //printf("line[%06d]: chars=%06zd, buf size=%06zu, contents: %s", line_cnt, line_size, line_buf_size, line_buf);

    // Get the next line 
    line_size = getline(&line_buf, &line_buf_size, file);
  }

  free(line_buf);
  line_buf = NULL;  
  // close the file
  fclose(file);
  //printf("parameter from file:[%d,%d,%d,%d,%f]\n",m,n,k,nnz,den);

  
  
    size_t A_len  = nnz;
    size_t B_len  = n * k;
    size_t C_len  = m * k;
    size_t size_ptr = m+1;


    float *A_val_ptr, *B_val_ptr, *C_val_ptr, *C_cmp_ptr;
    int *A_idx_ptr, *A_ptr_ptr;
        
    // allocate memory for sparse matrix A
    float* A_val = (float*)malloc_cache_aligned(sizeof(float),A_len,(void**)&A_val_ptr);
    float* B_val = (float*)malloc_cache_aligned(sizeof(float),B_len,(void**)&B_val_ptr);
    float* C_val = (float*)malloc_cache_aligned(sizeof(float),C_len,(void**)&C_val_ptr);
    float* C_cmp = (float*)malloc_cache_aligned(sizeof(float),C_len,(void**)&C_cmp_ptr);

    int* A_idx = (int*)malloc_cache_aligned(sizeof(int),A_len,(void**)&A_idx_ptr);
    int* A_ptr = (int*)malloc_cache_aligned(sizeof(int),size_ptr,(void**)&A_ptr_ptr);

    char filename_b[FILENAME_SZ];
    strcpy(filename_b, filename_base);
    if(synthetic_mat)
      strcat(filename_b, "spm16x100x2density0.1");
    else
      strcat(filename_b, "cora_gcn");

    printf("Test benchmark: %s", filename_b);
    printf("Reading A_val...\n");
    // matrix A - val
    char f_A_val[FILENAME_SZ];
    strcpy(f_A_val, filename_b);
    strcat(f_A_val, "_A_val.dat");
    read_value_float(f_A_val, A_val);

    printf("Reading A_idx...\n");
    // matrix A - val
    // matrix A - idx
    char f_A_idx[FILENAME_SZ];
    strcpy(f_A_idx, filename_b);
    strcat(f_A_idx, "_A_idx.dat");
    read_file_int(f_A_idx, A_idx);

    printf("Reading A_ptr...\n");
    // matrix A - ptr
    char f_A_ptr[FILENAME_SZ];
    strcpy(f_A_ptr, filename_b);
    strcat(f_A_ptr, "_A_ptr.dat");
    read_file_int(f_A_ptr, A_ptr);

    printf("Reading B_val...\n");
    // matrix A - ptr
    char f_B_val[FILENAME_SZ];
    strcpy(f_B_val, filename_b);
    strcat(f_B_val, "_B_val.dat");
    read_value_float(f_B_val, B_val);

    printf("Reading C_val...\n");
    // matrix A - ptr
    char f_C_val[FILENAME_SZ];
    strcpy(f_C_val, filename_b);
    strcat(f_C_val, "_C_val.dat");
    read_value_float(f_C_val, C_cmp);

*/
  

  //warm up l2
//stats_on();

/*#ifdef CACHE_WARM

float tempB, tempC, tempA;
int tempAi, tempAp;
  for(int i=0; i<mat_k*mat_n; i++){
    if (valB[i] > tempB)
      tempB = valB[i]; 
  }

  for(int i=0; i<mat_k*mat_m; i++){
    if (valC[i] > tempC)
      tempC = valC[i];
  }

  for(int i=0; i<mat_nnz; i++){
    if (valA[i] > tempA){
      tempA = valA[i];
      tempAi = idxA[i]; 
    }
  }

  for(int i=0; i<mat_m+1; i++){
    if (ptrA[i] > tempAp)
      tempAp = ptrA[i];
  }
  printf("tempA: %f, tempB: %f, tempAi: %d, tempAp: %d", tempA, tempB, tempAi, tempAp);

  #endif*/



  
  /*--------------------------------------------------------------------
  * Pack argument for kernel
  *-------------------------------------------------------------------*/

  float *C_val_ptr;

  float* C_kernel = (float*)malloc_cache_aligned(sizeof(float),mat_m*mat_k,(void**)&C_val_ptr);

  // initialize the arguments to send to each device core
  Kern_Args **kern_args = (Kern_Args **)malloc(sizeof(Kern_Args *) * num_cores);

  for (int y = 0; y < cores_y; y++)
  {
    for (int x = 0; x < cores_x; x++)
    {
      int i = x + y * cores_x;
    
      kern_args[i] = construct_args(valA, valB, C_kernel, idxA, ptrA, mat_m, mat_n, mat_k, mat_nnz, i,num_cores,wr_complete);
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
  int error=0;
  for(int i=0; i<mat_m; i++){
    for(int j=0; j<mat_k; j++){
      if (polybenchCompare(C_kernel[i * mat_k + j], valC[i * mat_k + j]))
      {
        printf("Kernel result: %f, expected: %f at [i:%d, j:%d]\n",C_kernel[i * mat_k + j],valC[i * mat_k + j], i,j);
        error +=1;
      }
    }
  }
  if (error >0){
    printf("total error detected: %d\n",error);
    return 1;
  }

  printf("The workq result is %d\n ",workq);
  printf("[[SUCCESS]]\n");

  
  free(C_val_ptr);

  return 0;
}
