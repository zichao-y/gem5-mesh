#ifndef __SYRK_H__
#define __SYRK_H__

#include "bind_defs.h"

// data type to do computation with
#define DTYPE float

/* Declared constant values for alpha and beta specfically for SYRK bench */
#define alpha 12435
#define beta 4546

// one of these should be defined to dictate config
// #define NO_VEC 1
// #define VEC_4_SIMD 1
// #define VEC_16_SIMD 1
// #define MANYCORE_PREFETCH

// vvadd_execute config directives
#if !defined(NO_VEC) && !defined(MANYCORE_PREFETCH)
#define USE_VEC 1
#endif

// vector grouping directives
#if defined(VEC_4_SIMD)
#define VECTOR_LEN 4
#endif
#if defined(VEC_16_SIMD)
#define VECTOR_LEN 16
#endif
#if defined(MANYCORE_PREFETCH)
#define VECTOR_LEN 1
#endif

// prefetch sizing
#if defined(USE_VEC) || defined(MANYCORE_PREFETCH)
// dedicate a quarter of scratchpad to frames
#define POST_FRAME_WORD 256

// number of frames to get ahead
#define INIT_FRAMES 1

// prefetch config for inner kernel
#define INNER_PREFETCH_LEN 16
#define INNER_FRAME_SIZE (2*INNER_PREFETCH_LEN)
#define NUM_FRAMES (POST_FRAME_WORD / INNER_FRAME_SIZE)
#define INIT_OFFSET (INIT_FRAMES * INNER_PREFETCH_LEN)

// frame size to get the c to accumulate on
#define OUTER_FRAME_SIZE INNER_FRAME_SIZE
#define OUTER_PREFETCH_LEN INNER_PREFETCH_LEN

// prefetch c
// pad out to the frame size (1->2 currently)
// maybe don't have to prefetch this
inline void prefetch_outer_frame(DTYPE *c, int i, int j, int *sp, int N) {
  for (int core = 0; core < VECTOR_LEN; core++) {
    VPREFETCH_LR(*sp + 0, &c[i * N + j + core], core, OUTER_PREFETCH_LEN, VERTICAL);

    // pad out
    VPREFETCH_LR(*sp + OUTER_PREFETCH_LEN, &c[i * N + j + core], core, OUTER_PREFETCH_LEN, VERTICAL);
  }

  *sp = *sp + OUTER_FRAME_SIZE;
  if (*sp == POST_FRAME_WORD) *sp = 0;
}

// prefetch a
inline void prefetch_inner_frame(DTYPE *a, int i, int j, int k, int *sp, int M) {
  for (int core = 0; core < VECTOR_LEN; core++) {
    // TODO redundant across cores
    VPREFETCH_L(*sp + 0, &a[i * M + k], core, INNER_PREFETCH_LEN, VERTICAL);

    // TODO this can be horizontal?
    VPREFETCH_L(*sp + INNER_PREFETCH_LEN, &a[(j + core) * M + k], core, INNER_PREFETCH_LEN, VERTICAL);
  }

  // will be done manually for manycore prefetch
  #ifndef MANYCORE_PREFETCH
  *sp = *sp + INNER_FRAME_SIZE;
  if (*sp == POST_FRAME_WORD) *sp = 0;
  #endif
}

#endif

// grid dim xy assuming always a square
#if _N_SPS==16
#define GRID_XDIM 4
#define GRID_YDIM 4
#elif _N_SPS==64
#define GRID_XDIM 8
#define GRID_YDIM 8
#endif

// pthread argument for the kernel
typedef struct Kern_Args {
  DTYPE *a, *c;
  int N, M;
  int tid_x, tid_y;
  int dim_x, dim_y;
} Kern_Args;

// helper to pack args
Kern_Args *construct_args(
    DTYPE *a, DTYPE *c,
    int N, int M,
    int tid_x, int tid_y, int dim_x, int dim_y
  );

// pthread call
void *pthread_kernel(void *args);

// kernel
void kernel(
    DTYPE *a, DTYPE *c,
    int N, int M,
    int tid_x, int tid_y, int dim_x, int dim_y
  );

#endif
