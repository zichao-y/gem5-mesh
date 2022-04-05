#include <stdio.h>

#include "pthread_launch.h"
#include "spmm.h"
#include "spad.h"
#include "bind_defs.h"
#include "group_templates.h"
#include "util.h"




void __attribute__((optimize("-fno-inline")))
spmm_manycore(int *A, int len, int *O,int ptid)
{

  /*if (ptid == 0 )
  {
    stats_on();
  }*/

  DTYPE *sp_c = (DTYPE *)getSpAddr(ptid, 0);
  //int sp_c[100];
  printf("in kernel!");
  //TEST loading from L2
  //pesudo CODE that load B to L2
  float b_cmp = 0;
  for(int i=0; i<LOAD_SZ; i ++){
    float b_val = A[i*16+ptid];
    if (b_val > b_cmp){
      b_cmp = b_val;
    }
  }
  FSTORE_NOACK(b_cmp, O + ptid, 0);

  if (ptid == 0 )
  {
    stats_on();
  }

    //code loading data from L2 to scratchpad
  //if (ptid ==0){
    for(int i=0; i<LOAD_SZ; i++){
      sp_c[i] = A[i*16+ptid];
    }
  //}
  if (ptid == 0)
  {
    stats_off();
  }
  

  //code to prevent from being optimized
  for(int i=0; i<LOAD_SZ; i++){
    b_cmp += sp_c[i];
  }
  FSTORE_NOACK(b_cmp, O + 128 + ptid, 0);

  /*if (ptid == 0)
  {
    stats_off();
  }*/
  
}