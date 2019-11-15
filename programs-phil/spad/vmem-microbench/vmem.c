#include <stdlib.h>
#include <stdio.h>

#include "pthread_launch.h"
#include "vmem.h"
#include "spad.h"
#include "../../common/bind_defs.h"

int data[4] = { 1, 2, 3, 4 };

// actual kernel
void kernel(
    int *a, int *b, int *c, int n,
    int tid_x, int tid_y, int dim_x, int dim_y) {
  
  // start recording all stats (all cores)
  if (tid_x == 0 && tid_y == 0) {
    stats_on();
  }
  
  // linearize tid and dim
  int tid = tid_x + tid_y * dim_x;
  int dim = dim_x * dim_y;
  
  // figure out which work this thread should do
  int start = tid * (n / dim);  

  // get end with remainders
  int chunk = (int)(n / dim);
  if (tid_x == dim - 1) {
    chunk += n % dim;
  }
  int end = start + chunk;
  
  //printf("iterations %d->%d\n", start, end);
  
  int mask = ALL_NORM;
  
  // upper left corner is the master
  if (tid_x == 0 && tid_y == 0) {
    mask = FET_O_INST_DOWN_SEND | FET_O_INST_RIGHT_SEND;
  }
  
  // right edge does not send to anyone
  else if (tid_x == dim_x - 1) {
    mask = FET_I_INST_LEFT;
  }
  
  // bottom left corner just sends to the right
  else if (tid_x == 0 && tid_y == dim_y - 1) {
    mask = FET_I_INST_UP | FET_O_INST_RIGHT_SEND;
  }
  
  // the left edge (besides corners) sends down and to the right
  else if (tid_x == 0) {
    mask = FET_I_INST_UP | FET_O_INST_DOWN_SEND | FET_O_INST_RIGHT_SEND;
  }
  
  // otherwise we're just forwarding to the right in the middle area
  else {
    mask = FET_I_INST_LEFT | FET_O_INST_RIGHT_SEND;
  }
  
  // specify the vlen
  int vlenX = 2;//2;
  int vlenY = 2;//2;
  mask |= (vlenX << FET_XLEN_SHAMT) | (vlenY << FET_YLEN_SHAMT);
  
  // NOTE potential optimization to avoid 64bit pointer store
  // b/c spad addresses are always 32bits in this setup
  int *spAddr = &(((int*)getSpAddr(tid, 0))[4]);
  int *dataAddr = &(data[tid]);
  printf("%d %p %p\n", tid, spAddr, dataAddr);
  
  VECTOR_EPOCH(mask);
  
  // do a memory load (prob need to do static analysis to know this will be consecutive iterations?)
  int val = -1; // for some reason asm volatile doesn't work without this
  /*asm volatile (
    ".insn sb 0x23, 0x2, %[st], 0(%[mem])\n\t" :: [st] "r" (spAddr), [mem] "r" (dataAddr)
  );*/
  //".insn sb 0x23, 0x2, %[st], 0(%[mem])\n\t" :: [st] "r" (spAddr), [mem] "r" (dataAddr)
  //"sw %[st], 0(%[mem])\n\t" :: [st] "r" (spAddr), [mem] "r" (dataAddr)
  //"lw %[st], 0(%[mem])\n\t" :: [st] "r" (val), [mem] "r" (&(data[tid]))
  VPREFETCH(spAddr, dataAddr, 0);
  LWSPEC_RESET(val, spAddr, 0);
  //val = spAddr[0];
  //val = data[tid];
  
  VECTOR_EPOCH(ALL_NORM);
  
  c[tid] = val;
  
  if (tid_x == 0 && tid_y == 0) {
    stats_off();
  }
  
  
  
}


// helper functions
Kern_Args *construct_args(int *a, int *b, int *c, int n,
  int tid_x, int tid_y, int dim_x, int dim_y) {
      
  Kern_Args *args = (Kern_Args*)malloc(sizeof(Kern_Args));
  
  args->a = a;
  args->b = b;
  args->c = c;
  args->n = n;
  args->tid_x = tid_x;
  args->tid_y = tid_y;
  args->dim_x = dim_x;
  args->dim_y = dim_y;
  
  return args;
      
}

void *pthread_kernel(void *args) {
  // guarentee one thread goes to each core, by preventing any threads
  // from finishing early
  pthread_barrier_wait(&start_barrier);
  
  // call the spmd kernel
  Kern_Args *a = (Kern_Args*)args;
  
  kernel(a->a, a->b, a->c, a->n,
      a->tid_x, a->tid_y, a->dim_x, a->dim_y);
      
  return NULL;
}
