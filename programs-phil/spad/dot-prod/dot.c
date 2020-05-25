#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "pthread_launch.h"
#include "dot.h"
#include "spad.h"
#include "../../common/bind_defs.h"

/*
  Dot product. Shows implementation of a dot product on our architecture

  The general flavor is to do local accumulation within a group/core.
  To accumulate across groups we want to go back to manycore mode
*/

inline int min(int a, int b) {
  if (a > b) {
    return b;
  }
  else {
    return a;
  }
}

int local_dot_manycore(DTYPE *a, DTYPE *b, int start, int end) {
  // accumulate a partial sum locally
  DTYPE partialSum = 0;
  for (int i = start; i < end; i++) {
    partialSum += a[i] * b[i];
  }
  return partialSum;
}

// // a shared buffer is just written once and then done
// typedef struct shared_buffer {
//   void *basePtr;
//   int size;
// } shared_buffer_t;

// // want to have token queue helpers to facilitate memory transfer for manycore code
// void init_shared_buffer_consumer(int consumerCore, int offset, int numWords, shared_buffer_t *buf) {
//   buf->basePtr = getSpAddr(consumerCore, offset);
//   buf->size = numWords;
//   int *doneFlagPtr = (int*)buf->basePtr + buf->size;
//   *doneFlagPtr = 0; // set done flag to 0
// }

// void init_shared_buffer_producer(int consumerCore, int offset, int numWords, shared_buffer_t *buf) {
//   buf->basePtr = getSpAddr(consumerCore, offset);
//   buf->size = numWords;
// }

// // consumer tries to read token queue, only return ptr when data is ready
// void *consume_shared_buffer(shared_buffer_t *buf) {
//   int *doneFlagPtr = (int*)buf->basePtr + buf->size;
//   while (*doneFlagPtr != 1) {}

//   return buf->basePtr;
// }

// // producer pushes to consumer once ready
// void produce_shared_buffer(shared_buffer_t *buf, void *mem, int numWords) {
//   for (int i = 0; i < numWords; i++) {
//     ((int*)(buf->basePtr))[i] = ((int*)(mem))[i];
//   }
//   int *doneFlagPtr = (int*)buf->basePtr + buf->size;
//   *doneFlagPtr = 1;
// }

// to support a circular queue instead, would need to have a head and tail pointer for both producer and consumer along with data
// producer updates where head pointer is and consumer updates wehere tail pointer is (would need to inform both producer and consumer of each pointer change)

typedef struct token_queue {
  int basePtr;
  // word offsets in the producer and consumers scratchpad
  // int headPtrOffset;
  // int tailPtrOffset;
  // offset in scratchpad where data is stored on the consumer
  // int dataPtrOffset;
  // the max size of the queue
  int size;
  // which core we want to establish the link with
  // int otherCoreIdx;
} token_queue_t;

inline int* get_head_ptr(token_queue_t *tq, int coreId) {
  return (int*)getSpAddr(coreId, tq->basePtr + 0);
}

inline int* get_tail_ptr(token_queue_t *tq, int coreId) {
  return (int*)getSpAddr(coreId, tq->basePtr + 1);
}

inline volatile int get_head(token_queue_t *tq, int coreId) {
  return *get_head_ptr(tq, coreId);
}

inline volatile int get_tail(token_queue_t *tq, int coreId) {
  return *get_tail_ptr(tq, coreId);
}

inline int* get_pair_tid_ptr(token_queue_t *tq, int coreId) {
  return (int*)getSpAddr(coreId, tq->basePtr + 2);
}

inline int* get_pair_base_ptr(token_queue_t *tq, int coreId) {
  return (int*)getSpAddr(coreId, tq->basePtr + 3);
}

inline int* get_data_ptr(token_queue_t *tq, int coreId) {
  return (int*)getSpAddr(coreId, tq->basePtr + 4);
}

inline int* get_other_head_ptr(token_queue_t *tq, int coreId) {
  return get_pair_base_ptr(tq, coreId) + 0;
}

inline int* get_other_tail_ptr(token_queue_t *tq, int coreId) {
  return get_pair_base_ptr(tq, coreId) + 1;
}

inline int* get_other_pair_tid_ptr(token_queue_t *tq, int coreId) {
  return get_pair_base_ptr(tq, coreId) + 2;
}

inline int* get_other_pair_base_ptr(token_queue_t *tq, int coreId) {
  return get_pair_base_ptr(tq, coreId) + 3; 
}

inline int* get_other_data_ptr(token_queue_t *tq, int coreId) {
  return get_pair_base_ptr(tq, coreId) + 4;
}

// want to declare token queue on the stack, so don't malloc
void init_token_queue_consumer(int spadOffset, int size, int thisCoreIdx, token_queue_t *tq) {
  tq->basePtr = spadOffset;
  tq->size = size;
  // tq->otherCoreIdx = otherCoreIdx; // confirm partner lazily, useful for vector were annoying to figure out partner

  // set head and tail offsets
  *get_head_ptr(tq, thisCoreIdx) = 0; // meaning offset 0 from base data ptr
  *get_tail_ptr(tq, thisCoreIdx) = 0;
}

void init_token_queue_producer(int spadOffset, int consumerOffset, int size, int thisCoreIdx, int otherCoreIdx, token_queue_t *tq) {
  tq->basePtr = spadOffset;
  tq->size = size;

  *get_pair_base_ptr(tq, thisCoreIdx) = consumerOffset;
  *get_other_pair_base_ptr(tq, otherCoreIdx) = spadOffset;

  // tq->otherCoreIdx = otherCoreIdx;
  *get_pair_tid_ptr(tq, thisCoreIdx) = otherCoreIdx;
  *get_other_pair_tid_ptr(tq, otherCoreIdx) = thisCoreIdx;

  // inform other core that this will be producing for it

  // set head and tail offsets
  *get_head_ptr(tq, thisCoreIdx) = 0; // meaning offset 0 from base data ptr
  *get_tail_ptr(tq, thisCoreIdx) = 0;
}
// don't use the fancy sleepy wait that's in hammerblade but w/e
// return base offset of first to read
int wait_tokens_consumer(token_queue_t *tq, int numTokens, int coreId) {
  int numTokensAvail;
  volatile int head;
  volatile int tail;
  do {
    head = get_head(tq, coreId);
    tail = get_tail(tq, coreId);
    if (tail > head) {
      numTokensAvail = tail - head;
    }
    else if (head < tail) {
      numTokensAvail = tq->size - (head - tail);
    }
    else { // ==
      numTokensAvail = 0;
    }
    printf("tid %d head %d tail %d tokensAvail %d tailAddr %p\n", coreId, head, tail, numTokensAvail, get_tail_ptr(tq, coreId));

  } while(numTokensAvail < numTokens);

  return head;
}

// wait for slots to be able to write to
// return base offset of first place to write to
int wait_tokens_producer(token_queue_t *tq, int numTokens, int coreId) {
  int openSpots;
  int head;
  int tail;
  do {
    int numTokensAvail;
    head = *get_head_ptr(tq, coreId);
    tail = *get_tail_ptr(tq, coreId);
    if (tail > head) {
      numTokensAvail = tail - head;
    }
    else if (head < tail) {
      numTokensAvail = tq->size - (head - tail);
    }
    else { // ==
      numTokensAvail = 0;
    }

    openSpots = tq->size - numTokensAvail;

  // } while(head + numTokens >= tail);
  } while (openSpots < numTokens);

  return tail;
}

// resolve circular offset in token queue
inline int get_circular_offset(token_queue_t *tq, int baseOffset, int tokenOffset, int bufSize) {
  int dataOffset = 0;
  int overShoot = baseOffset + tokenOffset - bufSize;
  if (overShoot > 0) {
    dataOffset = overShoot;
  }
  else {
    dataOffset = baseOffset + tokenOffset;
  }
  return dataOffset;
}

// consumer gets token
inline void *get_token(token_queue_t *tq, int tokenIdx, int coreId) {
  int head = *get_head_ptr(tq, coreId);
  int bufSize = tq->size;
  int dataOffset = get_circular_offset(tq, head, tokenIdx, bufSize);
  return (void*)(get_data_ptr(tq, coreId) + dataOffset);
}

// producer sets token
inline void set_token(token_queue_t *tq, int data, int tokenIdx, int coreId) {
  int tail = *get_other_tail_ptr(tq, *get_pair_tid_ptr(tq, coreId));
  int bufSize = tq->size;
  int dataOffset = get_circular_offset(tq, tail, tokenIdx, bufSize);
  get_other_data_ptr(tq, *get_pair_tid_ptr(tq, coreId))[dataOffset] = data;
}

// consumer consumes tokens by modifying tail pointer in both itself and producer core
void consume_tokens(token_queue_t *tq, int numTokens, int coreId) {
  int tail = *get_tail_ptr(tq, coreId);
  int bufSize = tq->size;
  int offset = get_circular_offset(tq, tail, numTokens, bufSize);

  // update the pointer
  int newHeadPtr = offset;
  *get_head_ptr(tq, coreId) = newHeadPtr;
  *get_other_head_ptr(tq, *get_pair_tid_ptr(tq, coreId)) = newHeadPtr; // maybe lazy ack?
}

// produce tokens by modifying head pointer both iteself and consumer core
void produce_tokens(token_queue_t *tq, int numTokens, int coreId) {
  int head = *get_head_ptr(tq, coreId);
  int bufSize = tq->size;
  int offset = get_circular_offset(tq, head, numTokens, bufSize);

  // update the pointer
  int newTailPtr = offset;
  *get_tail_ptr(tq, coreId) = newTailPtr;
  *get_other_tail_ptr(tq, *get_pair_tid_ptr(tq, coreId)) = newTailPtr;

  printf("tid %d produce tokens %d for tid %d new tail ptr %d == %d addr %p\n", 
    coreId, numTokens, *get_pair_tid_ptr(tq, coreId), *get_tail_ptr(tq, coreId), *get_other_tail_ptr(tq, *get_pair_tid_ptr(tq, coreId)), 
    get_other_tail_ptr(tq, *get_pair_tid_ptr(tq, coreId)));
}

// maybe add a "set producer option" to the token queue so can come from anywhere


// if did remote load would need flag to denote rdy and then also flag to denote done reading


// don't do a parallel reduction tree, instead just have one core do the summation of 64 values
// instead of having to wait for remote store from each core, can just load from them manually
// do a reduction, which cores to accumulate in? maybe for now just accumulate in a single core
// it won't matter for the grid size we're doing most likely (64 threads, might matter if have 1000s like in a GPU)
void reduce_manycore(int partialSum, DTYPE *c, int tid, int dim, token_queue_t *cons0, token_queue_t *cons1, token_queue_t *prod) {
  
  // advantage of remote loads is that don't need to have sync buffers
  // disadvantage is that need to figure out which cores have data which is not trivial in vector core case
  // would be nice if template had an easy to get all active vector cores

  // TODO for now just do remote stores
  // setup remote store buffers equal to number of active cores

  // TODO lightweight memory allocator for scratchpad (opt in per core)?
  // want to create a dynamically size array and store on scratchpad

  // also can potentially do thing where try to regularize the data by storing to a core reflecitve of your
  // group id and then its easier to collapse that
 
  // core 0 recvs data and does the actual work

  // this isn't gaurenteed to be synced... need to reset rdy flag. remote loads without any extra sync might be easier
  // if (tid == 0) {
  //   shared_buffer_t *bufs = (shared_buffer_t*)malloc(sizeof(shared_buffer_t) * numPartialSums);
  //   for (int i = 0; i < numPartialSums; i++) {
  //     init_shared_buffer_consumer(0, i, 1, &(bufs[i]));
  //   }
  // }
  // else {
  //   shared_buffer_t buf; 
  //   init_shared_buffer_producer(0, tid, 1, &buf);
  // }


  // can we use frames here again to help facilitate gather? since awkward/not scalable to create a token queue between one to everyone
  // would need to sync and change frame size after completing first part of kernel. seems kind of nice now to be able to change frame size on the fly
  // potentially could allow change to happen after start receiving counts. especially if doing token based

  // need to show that if the epoch value was above or below and received packets, still will be fine once change
  // so below is fine, although unclear what to do about the secondary counts
  //    cntr0 -> cntr0
  //    cntr1 -> ?
  // can avoid if always change by a factor of 2 (or how many counters we have)

  // also currently having an issue with barrier overflow. i guess not doing the sync neil suggests

  // PREFETCH_EPOCH()
  // pthread_barrier_wait();

  // get in the reduction

  // if (tid == 0) {
  //   DTYPE sum = partialSum;
  //   DTYPE *sp = (DTYPE*)getSpAddr(tid, 0);

  //   FRAME_START(numPartialSums);

  //   for (int i = 0; i < numPartialSums; i++) {
  //     sum += sp[i];
  //   }

  //   REMEM(numPartialSums);

  //   c[0] = sum;
  // }
  // else {
  //   DTYPE *targAddr = (DTYPE*)getSpAddr(0, tid);
  //   targAddr[0] = partialSum;
  // }

  int sum = partialSum;

  // the normal gpu way requires there to be different senders each time
  // // do a horizontal reduction
  // for (int bound = dim_x / 2; bound > 0; bound /= 2) {
  //   int producerBound = bound * 2;
  //   // going to recv from sender
  //   if (tid_x < bound) {

  //   }
  //   // going to recv from producer
  //   else if (tid_x < producerBound) {

  //   }
  // }
  // // do the vertical reduction

  // // do horizontal reduction
  // int bound = dim_x;
  // while (bound > 0) {
  //   int nextBound = bound / 2;
  //   int nextnextBound = nextBound / 2;

  //   // consumer section
  //   if (tid_x >= nextnextBound && tid_x < nextBound) {
  //     int t0 = wait_tokens_consumer(cons0, 1, tid);
  //     int t1 = wait_tokens_consumer(cons1, 1, tid);
  //     int *data0 = (int*)get_token(cons0, t0, tid);
  //     int *data1 = (int*)get_token(cons1, t1, tid);
  //     sum += data0[0] + data1[0];
  //     consume_tokens(cons0, 1, tid);
  //     consume_tokens(cons1, 1, tid);
  //   }

  //   // producer section
  //   if (tid_x >= nextBound && tid_x < bound) {
  //     int tokenOffset = wait_tokens_producer(prod, 1, tid);
  //     set_token(prod, sum, tokenOffset, tid);
  //     produce_tokens(prod, 1, tid);
  //   }

  //   bound = nextBound;
  // }

  // // do vertical reduction
  // bound = dim_y;
  // while (bound > 0) {

  // }

  // lower half consumes
  if (tid < dim / 2) {
    int t0 = wait_tokens_consumer(cons0, 1, tid);
    int t1 = wait_tokens_consumer(cons1, 1, tid);
    int *data0 = (int*)get_token(cons0, t0, tid);
    int *data1 = (int*)get_token(cons1, t1, tid);
    printf("tid %d get tokens %d %d\n", tid, data0[0], data1[1]);
    sum += data0[0] + data1[0];
    consume_tokens(cons0, 1, tid);
    consume_tokens(cons1, 1, tid);
  }

  // everyone produces, except for tid0 who does the writeback
  if (tid == 0) {
    *c = sum;
  }
  else {
    int tokenOffset = wait_tokens_producer(prod, 1, tid);
    set_token(prod, sum, tokenOffset, tid);
    produce_tokens(prod, 1, tid);
    printf("tid %d produce tokens %d\n", tid, sum);
  }
  

  printf("tid %d finish\n", tid);

}

// // have each vector core do a remote store into an easy to access core (i.e. there's now an easy to calculate patten of data)
// // now can do remote loads easily from the reduction core (cores)
// void organize_vector_results(int group_id, int data) {
//   int *spPtr = (int*)getSpAddr(group_id, 0);
//   spPtr[0] = data;
// }

// based on id (in manycore its the ptid, in vector its the group id + vtid)
// figure out where to send your data to be reduced by another core
int get_reduction_dest(int src_id) {
  // pattern is to half your id and send to that id

  // int dest_x, dest_y;

  // if (src_x != 0) {
  //   dest_x = src_x / 2;
  //   dest_y = src_y;
  // }
  // else {
  //   dest_x = src_x;
  //   dest_y = src_y / 2;
  // }

  // return dest_y * out_dim_x + dest_x;

  return src_id / 2;

}


void __attribute__((optimize("-freorder-blocks-algorithm=simple"))) kernel(
    DTYPE *a, DTYPE *b, DTYPE *c, int len,
    int tid_x, int tid_y, int dim_x, int dim_y) {
  
  // start recording all stats (all cores)
  if (tid_x == 0 && tid_y == 0) {
    stats_on();
  }

  // linearize tid and dim
  int tid = tid_x + tid_y * dim_x;
  int dim = dim_x * dim_y;

  // split into physical and virtual tids + dim
  int ptid_x = tid_x;
  int ptid_y = tid_y;
  int ptid   = tid;
  int pdim_x = dim_x;
  int pdim_y = dim_y;
  int pdim   = dim;
  int vtid_x = 0;
  int vtid_y = 0;
  int vtid   = 0;
  int vdim_x = 0;
  int vdim_y = 0;
  int vdim   = 0;
  int start  = 0;
  int end    = 0;
  int orig_x = 0;
  int orig_y = 0;
  int is_da  = 0;
  int master_x = 0;
  int master_y = 0;
  int unique_id = 0;
  int total_groups = 0;
  
  // // number of partial sums to expect
  // int num_partial_sums = 0;

  // group construction
  #if VECTOR_LEN==4
  // virtual group dimension
  vdim_x = 2;
  vdim_y = 2;

  int used = vector_group_template_4(ptid_x, ptid_y, pdim_x, pdim_y, 
    &vtid, &vtid_x, &vtid_y, &is_da, &orig_x, &orig_y, &master_x, &master_y, &unique_id, &total_groups);

  // TODO should use alignment
  if (used) {
    start = ( (unique_id + 0) * effRows ) / total_groups;
    end   = ( (unique_id + 1) * effRows ) / total_groups;
  }

  // num_partial_sums = total_groups * VECTOR_LEN;

  // printf("ptid %d(%d,%d) vtid %d(%d,%d) dim %d(%d,%d) %d->%d used? %d\n", ptid, ptid_x, ptid_y, vtid, vtid_x, vtid_y, 4, vdim_x, vdim_y, start, end, used); 

  #elif VECTOR_LEN==16

  vdim_x = 4;
  vdim_y = 4;

  int used = vector_group_template_16(ptid_x, ptid_y, pdim_x, pdim_y, 
    &vtid, &vtid_x, &vtid_y, &is_da, &orig_x, &orig_y, &master_x, &master_y, &unique_id, &total_groups);

  if (used) {
    start = ( (unique_id + 0) * effRows ) / total_groups;
    end   = ( (unique_id + 1) * effRows ) / total_groups;
  }

  // num_partial_sums = total_groups * VECTOR_LEN;

  // printf("ptid %d(%d,%d) vtid %d(%d,%d) dim %d(%d,%d) %d->%d used? %d\n", ptid, ptid_x, ptid_y, vtid, vtid_x, vtid_y, 16, vdim_x, vdim_y, start, end, used); 

  #elif !defined(USE_VEC)

  vdim_x = 1;
  vdim_y = 1;
  vtid_x = 0;
  vtid_y = 0;
  vtid   = 0;
  start  = ( ( ptid + 0 ) * len ) / pdim;
  end    = ( ( ptid + 1 ) * len ) / pdim;

  // num_partial_sums = pdim;

  // printf("%d->%d\n", start, end); 
  

  #endif

  // linearize some fields
  vdim = vdim_x * vdim_y;
  int orig = orig_x + orig_y * dim_x;

  #ifdef USE_VEC
  // volatile so dont reorder this function call
  int mask = getSIMDMask(master_x, master_y, orig_x, orig_y, vtid_x, vtid_y, vdim_x, vdim_y, is_da);
  #endif

  // printf("ptid %d(%d,%d) vtid %d(%d,%d) dim %d(%d,%d) %d->%d\n", ptid, ptid_x, ptid_y, vtid, vtid_x, vtid_y, vdim, vdim_x, vdim_y, start, end); 

  #ifdef NUM_REGIONS
  int prefetchMask = (NUM_REGIONS << PREFETCH_NUM_REGION_SHAMT) | (REGION_SIZE << PREFETCH_REGION_SIZE_SHAMT);
  PREFETCH_EPOCH(prefetchMask);

  // make sure all cores have done this before begin kernel section --> do thread barrier for now
  // TODO hoping for a cleaner way to do this
  pthread_barrier_wait(&start_barrier);
  #endif

  // setup token queues
  // TODO lightweight scratchpad memory allocator
  int spmOffset = 100;
  int bufSize = 10;
  int tqWords = bufSize + 4 + 2 + 2; // +2 extra just to be safe

  // each spm gets two consumer queues and one producer queue for a reduction
  token_queue_t consumer0;
  token_queue_t consumer1;
  token_queue_t producer;
  int pairTid = get_reduction_dest(ptid); 
  init_token_queue_consumer(spmOffset + tqWords * 0, bufSize, ptid, &consumer0);
  init_token_queue_consumer(spmOffset + tqWords * 1, bufSize, ptid, &consumer1);

  int pairOffset;
  if (ptid % 2 == 0) {
    pairOffset = spmOffset + tqWords * 0;
  }
  else {
    pairOffset = spmOffset + tqWords * 1;
  }
  init_token_queue_producer(spmOffset + tqWords * 2, pairOffset, bufSize, ptid, pairTid, &producer);

  pthread_barrier_wait(&start_barrier);

  // each vector group size is rated to do a certain problem size and multiples of that problem size
  // for the mod of this we need to do the rest on the flexible manycore version
  int rated_size = 0;
  #ifdef REUSE
  rated_size = ( VECTOR_LEN * FILTER_DIM - (FILTER_DIM - 1) );
  #elif defined(VERTICAL_LOADS)
  rated_size = ( VECTOR_LEN * CORE_STEP );
  #elif defined(VECTOR_LEN)
  rated_size = ( VECTOR_LEN * FILTER_DIM );
  #else
  rated_size = 1;
  #endif

  // mapped len is schedule on main config, unmapped will be scheduled on base manycore
  int unmapped_len = len % rated_size;
  int mapped_len = len - unmapped_len;

  // if (ptid == 0)
  //   printf("size %d rated size %d mapped %d unmapped %d\n", eff_len, rated_size, mapped_len, unmapped_len);

  // only let certain tids continue
  #if defined(USE_VEC)
  if (used == 0) return;
  #endif

  // save the stack pointer to top of spad and change the stack pointer to point into the scratchpad
  // reset after the kernel is done
  // do before the function call so the arg stack frame is on the spad
  // store the the current spAddr to restore later 
  unsigned long long *spTop = getSpTop(ptid);
  // guess the remaining of the part of the frame that might be needed??
  spTop -= 12;

  unsigned long long stackLoc;
  unsigned long long temp;
  for(int i=0;i<12;i++){
    asm volatile("ld t0, %[id](sp)\n\t"
                "sd t0, %[id](%[spad])\n\t"
                : "=r"(temp)
                : [id] "i"(i*8), [spad] "r"(spTop));
  }
  asm volatile (// save the stack ptr
      "addi %[dest], sp, 0\n\t"
      // overwrite stack ptr
      "addi sp, %[spad], 0\n\t"
      : [ dest ] "=r"(stackLoc)
      : [ spad ] "r"(spTop));

  // accumulate partial sums locally
  int partialSum = local_dot_manycore(a, b, start, end);
  

  printf("tid %d psum %d red_tid %d\n", ptid, partialSum, pairTid);

  // the core who does the reduction doesn't need to wait for iteself
  // num_partial_sums--;

  // // setup syncronizations
  // // TODO can we somehow manage without a barrier here?
  // int prefetchMask = (1 << PREFETCH_NUM_REGION_SHAMT) | (num_partial_sums << PREFETCH_REGION_SIZE_SHAMT);
  // PREFETCH_EPOCH(prefetchMask);

  // // make sure all cores have done this before begin kernel section --> do thread barrier for now
  // // TODO hoping for a cleaner way to do this
  // pthread_barrier_wait(&start_barrier);

  // // do reduction across cores (currently just send all to a single core rather than reduction tree)
  reduce_manycore(partialSum, c, ptid, pdim, &consumer0, &consumer1, &producer);


  // restore stack pointer
  asm volatile (
    "addi sp, %[stackTop], 0\n\t" :: [stackTop] "r" (stackLoc)
  );

}


// helper functions
Kern_Args *construct_args(DTYPE *a, DTYPE *b, DTYPE *c, int len,
  int tid_x, int tid_y, int dim_x, int dim_y) {

  Kern_Args *args = (Kern_Args*)malloc(sizeof(Kern_Args));
  
  args->a = a;
  args->b = b;
  args->c = c;
  args->len = len;
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
  
  kernel(a->a, a->b, a->c, a->len,
      a->tid_x, a->tid_y, a->dim_x, a->dim_y);

  pthread_barrier_wait(&start_barrier);

  if (a->tid_x == 0 && a->tid_y == 0) {
    stats_off();
  }

  return NULL;
}
