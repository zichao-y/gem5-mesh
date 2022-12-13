#include <stdio.h>
#include <stdatomic.h>

#include "pthread_launch.h"
#include "spmm.h"
#include "spad.h"
#include "data.h"
#include "bind_defs.h"
#include "group_templates.h"
#include "util.h"



void __attribute__((optimize("-fno-inline")))
spmm_manycore(const float *A_val, const float *B_val, float *C_val, const int *A_idx, const int *A_ptr, int m, int n,int k, int ptid, int num_cores, int *wr_mask)
{

  DTYPE *sp_base = (DTYPE *)getSpAddr(ptid, 0);
  DTYPE *sp_b = sp_base;
  DTYPE *sp_c = sp_base + 64;
  int sp_b_offset = 0;

  //row-wise product SpMM
  for(int i=atomic_fetch_add_explicit(&workq, 1, memory_order_relaxed); i<m; i =atomic_fetch_add_explicit(&workq, 1, memory_order_relaxed) ){
    int start=A_ptr[i];
    int end=A_ptr[i+1];
    for(int j=start; j<end; j++){
      float a_val = A_val[j];
      int a_idx = A_idx[j];
      /*if(ptid==0){
          printf("k is %d \n",k);
          printf("B address is %f: \n",B_val);
      }*/
      //FRAME_START(16);
      VPREFETCH_LR(sp_b_offset, B_val+a_idx*k, 0, 16,TO_SELF);
      //VPREFETCH_LR(sp_b_offset+16, B_val+a_idx*k+16, 0, 3,TO_SELF);
      //VPREFETCH_LR(sp_b_offset+32, B_val+a_idx*k+32, 0, 16,TO_SELF);
      //VPREFETCH_LR(sp_b_offset+48, B_val+a_idx*k+48, 0, 16,TO_SELF);
      FRAME_START(16);
      /*if(i==624){
        for(int l=0; l<k; l++){
          printf("spb[%d] is %f: \n",l,sp_base[l]);
          printf("B[%d] is %f: \n",l,B_val[a_idx*k+l]);
        }
      }*/
      #pragma GCC unroll(16)
      for(int c_idx=0; c_idx<k; c_idx ++){
        sp_c[c_idx] += a_val * sp_b[c_idx];
      }
      REMEM();
    }

    #pragma GCC unroll(16)
    for(int c_idx=0; c_idx<k; c_idx++){
      FSTORE_NOACK(sp_c[c_idx], C_val + i*k + c_idx, 0);
      sp_c[c_idx] = 0; 
    }
    //atomic_fetch_add_explicit(&workq, 1, memory_order_relaxed);
  }

}