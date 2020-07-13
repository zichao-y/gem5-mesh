#include "if_test_kernel.h"

void tril_if_delim_test(int mask, DTYPE *outbuf) {
#ifdef SCALAR_CORE
  VECTOR_EPOCH(mask);
#endif

  // The vector block for initialization.
#ifdef SCALAR_CORE
  ISSUE_VINST(init_label);
#elif defined VECTOR_CORE
  asm("trillium vissue_delim until_next vector_init");
#endif

  // A loop that issues vector blocks. We use a "black hole" loop to replace
  // the proper loop on the vector cores.
#ifdef SCALAR_CORE
  for (int i = 0; i < 100; ++i) {
#elif defined VECTOR_CORE
  volatile int BH;
  int vector_i = 0;
  do {
#endif
#ifdef SCALAR_CORE
    ISSUE_VINST(if_block_label);
#elif defined VECTOR_CORE
    asm("trillium vissue_delim if_begin if_block");
    if (outbuf[vector_i] == 7) {
      outbuf[vector_i] = 42;
    }
    ++vector_i;
    asm("trillium vissue_delim end at_jump");
#endif
#ifdef SCALAR_CORE
  }
#elif defined VECTOR_CORE
  } while (BH);
#endif

  // Clean up on the vector cores.
#ifdef SCALAR_CORE
  ISSUE_VINST(vector_return_label);
#elif defined VECTOR_CORE
  asm("trillium vissue_delim return vector_return");
  return;
#endif

  // Disband the vector group.
#ifdef SCALAR_CORE
  DEVEC(devec_0);
  asm volatile("fence\n\t");
  asm("trillium vissue_delim return scalar_return");  // XXX is this real???
  return;
#endif

  // Glue points!
#ifdef SCALAR_CORE
init_label:
  asm("trillium glue_point vector_init");
if_block_label:
  asm("trillium glue_point if_block");
vector_return_label:
  asm("trillium glue_point vector_return");
#endif
}
