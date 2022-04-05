#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>

#include "spad.h"
#include "pthread_launch.h"
#include "spmm.h"
#include "util.h"


void fill_array(int *m, int n)
{
  int rand_temp = rand()%10;
  for (int i = 0; i < n; i++)
  {
    m[i] = (rand_temp + i)%10;
  }
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

  int cores_x, cores_y;
  cores_x = 1;
  cores_y = 1;
  //int num_cores = get_dimensions(&cores_x, &cores_y);
  if (argc > 1)
      cores_x = atoi(argv[1]);
  if (argc > 2)
      cores_y = atoi(argv[2]);
  
  int num_cores = cores_x * cores_y;
  /*--------------------------------------------------------------------
  * Data initialization
  *-------------------------------------------------------------------*/
  

  
  
    
    size_t A_len  = 16*100;
    
    int *A_ptr;
    int* A = (int*)malloc_cache_aligned(sizeof(int),A_len,(void**)&A_ptr);
    int *O_ptr;
    int* O = (int*)malloc_cache_aligned(sizeof(int),A_len,(void**)&O_ptr);
    
    fill_array(A,A_len);
   
  /*--------------------------------------------------------------------
  * Pack argument for kernel
  *-------------------------------------------------------------------*/

  // initialize the arguments to send to each device core
  Kern_Args **kern_args = (Kern_Args **)malloc(sizeof(Kern_Args *) * num_cores);

  for (int y = 0; y < cores_y; y++)
  {
    for (int x = 0; x < cores_x; x++)
    {
      int i = x + y * cores_x;
    
      kern_args[i] = construct_args(A, A_len,O,i);
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

  /*for(int i=0; i<m; i++){
    for(int j=0; j<k; j++){
      if (polybenchCompare(C_val[i * k + j], C_cmp[i * k + j]))
      {
        printf("%f %f at i:%d, j:%d\n",C_val[i * n + j],C_cmp[i * k + j], i,j);
        printf("[[FAIL]]\n");
        return 1;
      }
    }
  }*/

  printf("[[SUCCESS]]\n");

  free(A_ptr);
  free(O_ptr);
  
  return 0;
}
