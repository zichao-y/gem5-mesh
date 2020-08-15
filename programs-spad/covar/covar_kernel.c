#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "pthread_launch.h"
#include "covar.h"
#include "spad.h"
#include "bind_defs.h"
#include "group_templates.h"
#include "util.h"

// #define SCALAR_CORE
// #define VECTOR_CORE

/*-----------------------------------------------------------------------------------
 * Vector versions of the kernels.
 *---------------------------------------------------------------------------------*/

#ifdef USE_VEC
// compute each mean across each vector (single dimension)
void tril_mean(int mask, DTYPE *mean, DTYPE *data, int N, int M, 
    int ptid, int groupId, int numGroups, int vtid) {

  #ifdef SCALAR_CORE
  VECTOR_EPOCH(mask);

  int start = ((groupId + 0) * N) / numGroups;
  int end   = ((groupId + 1) * N) / numGroups;

  // make it a factor of vector group mapping size
  start = roundUp(start, VECTOR_LEN);
  end   = roundUp(end  , VECTOR_LEN);

  int startOffset = min(INIT_MEAN_OFFSET, N);

  ISSUE_VINST(init_label);
  #endif

  #ifdef VECTOR_CORE
  asm("trillium vissue_delim until_next vector_init");
  int start = ((groupId + 0) * N) / numGroups;
  start = roundUp(start, VECTOR_LEN);
  int i = start + vtid;
  int j = 0;
  DTYPE mean_i = 0.0f;
  int sp = 0;
  DTYPE *sp_ptr = (DTYPE*)getSpAddr(ptid, 0);
  #endif

  #ifdef SCALAR_CORE

  int sp  = 0;

  for (int i = start; i < end; i+=VECTOR_LEN) {

    ISSUE_VINST(vec_body_init_label);

    // initial round
    for (int j = 0; j < startOffset; j+=MEAN_UNROLL_LEN) {
      // printf("bpf %d %d\n", i, j);
      prefetch_mean_frame(data, i, j, &sp, M);
    }

    // steady state
    for (int j = startOffset; j < N; j+=MEAN_UNROLL_LEN) {
      // printf("mpf %d %d\n", i, j);
      prefetch_mean_frame(data, i, j, &sp, M);
      ISSUE_VINST(mean_body_label);
    }

    //cooldown
    for (int j = N - startOffset; j < N; j+=MEAN_UNROLL_LEN) {
      // printf("epf %d %d\n", i, j);
      ISSUE_VINST(mean_body_label);
    }

    ISSUE_VINST(center_begin_label);


    // initial round
    for (int j = 0; j < startOffset; j+=MEAN_UNROLL_LEN) {
      prefetch_mean_frame(data, i, j, &sp, M);
    }

    // steady state
    for (int j = startOffset; j < N; j+=MEAN_UNROLL_LEN) {
      prefetch_mean_frame(data, i, j, &sp, M);
      ISSUE_VINST(center_body_label);
    }

    //cooldown
    for (int j = N - startOffset; j < N; j+=MEAN_UNROLL_LEN) {
      ISSUE_VINST(center_body_label);
    }

    ISSUE_VINST(vec_body_end_label);
  }
  #endif

  #ifdef VECTOR_CORE
  volatile int BH;
  volatile int BHO;
  do {

    asm("trillium vissue_delim until_next vec_body_init");

    do {
      asm("trillium vissue_delim if_begin mean_body");
      START_FRAME();
      #pragma GCC unroll(16)
      for (int u = 0; u < MEAN_UNROLL_LEN; u++) {
        mean_i += sp_ptr[sp + u];
      }
      END_FRAME();
      sp+=MEAN_FRAME_SIZE;
      sp = sp % POST_FRAME_WORD;
      // #if VECTOR_LEN==16
      // #pragma GCC unroll(16)
      // for (int n = 0; n < 3; n++) {
      //   asm volatile("nop\n\t");
      // }
      // #endif
      asm("trillium vissue_delim end at_jump");
    } while(BH);

    asm("trillium vissue_delim until_next center_begin");
    mean_i /= (DTYPE)FLOAT_N;
    STORE_NOACK(mean_i, &mean[i], 0);

    do {
      asm("trillium vissue_delim if_begin center_body");
      START_FRAME();
      #pragma GCC unroll(16)
      for (int u = 0; u < MEAN_UNROLL_LEN; u++) {
        DTYPE dat = sp_ptr[sp + u] - mean_i;
        STORE_NOACK(dat, &data[i * N + j + u], 0);
      }
      END_FRAME();
      j+=MEAN_UNROLL_LEN;
      sp += MEAN_FRAME_SIZE;
      sp = sp % POST_FRAME_WORD;
      asm("trillium vissue_delim end at_jump");
    } while(BH);

    asm("trillium vissue_delim if_begin vec_body_end");
    i+=VECTOR_LEN;
    mean_i = 0.0f;
    j = 0;
    asm("trillium vissue_delim end at_jump");

  } while (BHO);
  #endif


  // Clean up on the vector cores.
#ifdef SCALAR_CORE
  ISSUE_VINST(vector_return_label);
#elif defined VECTOR_CORE
  asm("trillium vissue_delim return vector_return");
  return;
#endif

#ifdef SCALAR_CORE
  // devec with unique tag
  DEVEC(devec_0);

  // we are doing lazy store acks, so use this to make sure all stores have commited to memory
  asm volatile("fence\n\t");
  asm("trillium vissue_delim return scalar_return");  // XXX is this real???
  return;
#endif

  // Glue points!
#ifdef SCALAR_CORE
init_label:
  asm("trillium glue_point vector_init");
vec_body_init_label:
  asm("trillium glue_point vec_body_init");
mean_body_label:
  asm("trillium glue_point mean_body");
center_begin_label:
  asm("trillium glue_point center_begin");
center_body_label:
  asm("trillium glue_point center_body");
vec_body_end_label:
  asm("trillium glue_point vec_body_end");
vector_return_label:
  asm("trillium glue_point vector_return");
#endif
  // }
  // else {
  // DTYPE *sp_ptr = (DTYPE*)getSpAddr(ptid, 0);
  // for (int j = start + vtid; j < end; j+=VECTOR_LEN) {
  //   DTYPE mean_j = 0.0f;
  //   for (int i = 1; i < (N+1); i++) {
  //     FRAME_START(MEAN_FRAME_SIZE);
  //     mean_j += sp_ptr[sp];
  //     REMEM(MEAN_FRAME_SIZE);
  //     sp++;
  //     if (sp == POST_FRAME_WORD) sp = 0;
  //   }
  //   mean_j /= (DTYPE)FLOAT_N;
  //   mean[j] = mean_j;
  // }

  // }
}

// compute the covariance matrix
void tril_covar(int mask, DTYPE *symmat, DTYPE *data, int N, int M, 
    int ptid, int groupId, int numGroups, int vtid) {

  #ifdef SCALAR_CORE
  VECTOR_EPOCH(mask);

  int start = groupId; // * VECTOR_LEN;
  int stride = numGroups; // * VECTOR_LEN;
  int end = N;

  int startOffset = min(INIT_COVAR_OFFSET, N);

  ISSUE_VINST(init_label);
  #endif

  #ifdef VECTOR_CORE
  asm("trillium vissue_delim until_next vector_init");
  int start = groupId; // * VECTOR_LEN;
  // int j2 = j1;
  int i2;
  int stride = numGroups;// * VECTOR_LEN;
  int i1 = start - stride;
  int sp = 0;
  DTYPE* sp_ptr = (DTYPE*)getSpAddr(ptid, 0);
  #endif

  #ifdef SCALAR_CORE
  int sp  = 0;

  for (int i1 = start; i1 < end; i1+=stride) {

    ISSUE_VINST(j2_begin_label);

    for (int i2 = i1; i2 < N; i2+=VECTOR_LEN) {

      ISSUE_VINST(vec_body_init_label);

      // initial round
      for (int j = 0; j < startOffset; j+=COVAR_UNROLL_LEN) {
        prefetch_covar_frame(data, i1, i2, j, &sp, M);
      }

       // steady state
      for (int j = startOffset; j < M; j+=COVAR_UNROLL_LEN) {
        prefetch_covar_frame(data, i1, i2, j, &sp, M);
        ISSUE_VINST(vec_body_label);
      }

      // cooldown
      for (int j = N - startOffset; j < N; j+=COVAR_UNROLL_LEN) {
        ISSUE_VINST(vec_body_label);
      }

      ISSUE_VINST(vec_body_end_label);
    }

    ISSUE_VINST(j2_end_label);
  }
  #endif

  #ifdef VECTOR_CORE
  volatile int BH;
  volatile int BHO;
  volatile int BHOO;
  do {
    asm("trillium vissue_delim until_next j2_begin");
    i1+=stride;
    i2 = i1 + vtid;

  do {

    asm("trillium vissue_delim until_next vec_body_init");
    DTYPE symmat_idx = 0.0f;

    do {
      asm("trillium vissue_delim if_begin vec_body");
      FRAME_START(COVAR_FRAME_SIZE);
      #pragma GCC unroll(8)
      for (int u = 0; u < COVAR_UNROLL_LEN; u++) {
        symmat_idx += sp_ptr[sp + u] * sp_ptr[sp + COVAR_UNROLL_LEN + u];
      }
      REMEM(COVAR_FRAME_SIZE);
      sp+=COVAR_FRAME_SIZE;
      sp = sp % POST_FRAME_WORD;
      // #if VECTOR_LEN==16
      // #pragma GCC unroll(16)
      // for (int n = 0; n < 1; n++) {
      //   asm volatile("nop\n\t");
      // }
      // #endif
      asm("trillium vissue_delim end at_jump");
    } while(BH);


    asm("trillium vissue_delim if_begin vec_body_end");
    int gt = (i2 >= N);
    PRED_EQ(gt, 0);
    STORE_NOACK(symmat_idx, &symmat[i2 * M + i1], 0);
    STORE_NOACK(symmat_idx, &symmat[i1 * M + i2], 0);
    // symmat[j2 * (M+1) + j1] = symmat_idx;
    // symmat[j1 * (M+1) + j2] = symmat_idx;
    PRED_EQ(i2, i2);
    i2+=VECTOR_LEN;
    asm("trillium vissue_delim end at_jump");

  } while (BHO);

    asm("trillium vissue_delim if_begin j2_end");

    asm("trillium vissue_delim end at_jump");

  } while(BHOO);
  #endif


  // Clean up on the vector cores.
#ifdef SCALAR_CORE
  ISSUE_VINST(vector_return_label);
#elif defined VECTOR_CORE
  asm("trillium vissue_delim return vector_return");
  return;
#endif

#ifdef SCALAR_CORE
  // devec with unique tag
  DEVEC(devec_0);

  // we are doing lazy store acks, so use this to make sure all stores have commited to memory
  asm volatile("fence\n\t");
  asm("trillium vissue_delim return scalar_return");  // XXX is this real???
  return;
#endif

  // Glue points!
#ifdef SCALAR_CORE
init_label:
  asm("trillium glue_point vector_init");
  exit(1);
j2_begin_label:
  asm("trillium glue_point j2_begin");
  exit(1);
vec_body_init_label:
  asm("trillium glue_point vec_body_init");
  exit(1);
vec_body_label:
  asm("trillium glue_point vec_body");
  exit(1);
vec_body_end_label:
  asm("trillium glue_point vec_body_end");
  exit(1);
j2_end_label:
  asm("trillium glue_point j2_end");
  exit(1);
vector_return_label:
  asm("trillium glue_point vector_return");
  exit(1);
#endif



  // if (ptid == 0) {
  
  // ISSUE_VINST()

  // // initial round
  // for (int i = 1; i < 1 + INIT_COVAR_OFFSET; i++) {
  //   prefetch_covar_frame(data, i, start, start, &sp, M);
  // }

  // // first row
  // for (int i = 1 + INIT_COVAR_OFFSET; i < (N+1); i++) {
  //   prefetch_covar_frame(data, i, start, start, &sp, M);
  //   // ISSUE_VINST()
  // }

  // // steady state
  // for (int j1 = start; j1 < end; j1++) {
  //   int startJ2 = j1;
  //   if (j1 == start) startJ2 += VECTOR_LEN;
  //   for (int j2 = startJ2; j2 < (M+1); j2+=VECTOR_LEN) {
  //     for (int i = 1; i < (N+1); i++) {
  //       prefetch_covar_frame(data, i, j1, j2, &sp, M);
  //       // ISSUE_VINST()
  //     }
  //   }
  // }

  // // cooldown
  // for (int i = N - INIT_MEAN_OFFSET; i < (N+1); i++) {
  //   // ISSUE_VINST()
  // }

  // }
  // else {
  // DTYPE *sp_ptr = (DTYPE*)getSpAddr(ptid, 0);
  // for (int j1 = start; j1 < end; j1++) {
  //   // for (int j2 = j1 + vtid; j2 < (M+1); j2+=VECTOR_LEN) { // TODO needs predication on this loop
  //   for (int j2 = j1; j2 < (M+1); j2+=VECTOR_LEN) {
  //     int j2_idx = j2 + vtid;
  //     DTYPE symmat_idx = 0.0f;
  //     for (int i = 1; i < (N+1); i++) {
  //       FRAME_START(COVAR_FRAME_SIZE);
  //       // printf("j1 %d j2 %d i %d vtid %d %f ?= %f | %f ?= %f\n", j1, j2, i, vtid, sp_ptr[sp+0], data[i *(M+1) + j1], sp_ptr[sp+1], data[i *(M+1) + j2]);
  //       symmat_idx += sp_ptr[sp + 0] * sp_ptr[sp + 1]; // not prefetching the right stuff here
  //       REMEM(COVAR_FRAME_SIZE);
  //       sp+=2;
  //       if (sp == POST_FRAME_WORD) sp = 0;
  //     }

  //     if (j2_idx < (M+1)) {
  //       symmat[j2_idx * (M+1) + j1] = symmat_idx;
  //       symmat[j1 * (M+1) + j2_idx] = symmat_idx;
  //     }
  //   }
  // }

  // }

  // for (int j1 = start + 1; j1 < (end+1); j1++) {
  //   for (int j2 = j1; j2 < (M+1); j2++) {
  //     DTYPE symmat_idx = 0.0f;
  //     for (int i = 1; i < (N+1); i++) {
  //       symmat_idx += data[i *(M+1) + j1] * data[i *(M+1) + j2];
  //     }
  //     symmat[j2 * (M+1) + j1] = symmat_idx;
  //     symmat[j1 * (M+1) + j2] = symmat_idx;
  //   }
  // }
}
#endif