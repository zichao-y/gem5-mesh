#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>

#include "pthread_launch.h"
#include "spad.h"
#include "bind_defs.h"
#include "group_templates.h"
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

  int matrixlen = 100*16*10;
  //initialize matrix
  int * matrix = (int*)malloc(sizeof(int)*matrixlen);  
  fill_array(matrix,matrixlen);

  int temp_c =0;
  int check_row = 5;

  stats_on();
  for(int i=0; i<check_row; i++){
    if (matrix[i*16] > temp_c){
      temp_c = matrix[i*16];
    }
  }
  stats_off();

  printf("result is %d ! \n",temp_c);
    

  free(matrix);
  
  return 0;
}
