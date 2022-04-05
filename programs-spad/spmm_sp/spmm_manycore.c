#include <stdio.h>

#include "pthread_launch.h"
#include "spmm.h"
#include "spad.h"
#include "bind_defs.h"
#include "group_templates.h"
#include "util.h"


void __attribute__((optimize("-fno-inline")))
spmm_manycore(float *A_val, float *B_val, float *C_val, int *A_idx, int *A_ptr, int m, int k, int ptid)
{

  DTYPE *sp_c = (DTYPE *)getSpAddr(ptid, 0);

  //row-wise product SpMM
  for(int i=ptid; i<m; i += _N_SPS){
    for(int j=A_ptr[i]; j<A_ptr[i+1]; j++){
      float a_val = A_val[j];
      int a_idx = A_idx[j];
      for(int c_idx=0; c_idx<k; c_idx ++){
        float b_val = B_val[a_idx * k + c_idx];
        sp_c[c_idx] += a_val * b_val;
      }
    }

    for(int c_idx=0; c_idx<k; c_idx++){
      FSTORE_NOACK(sp_c[c_idx], C_val + i*k + c_idx, 0);
      sp_c[c_idx] = 0; 
    }
  }
}