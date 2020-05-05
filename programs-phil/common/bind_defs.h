#ifndef __BIND_DEFS_H__
#define __BIND_DEFS_H__

// include spec from gem5
#include "../../src/custom/bind_spec.hh"

// #if !defined(__x86_64__) && !defined(__i386__)

// 20 bit / 5 hex
#define ALL_NORM  0x00000

// if you want to include a comma in a macro, need to indirectly do like
// so, otherwise the pre-processor will assume its a delimiter for the
// macro args
#define COMMA ,

// https://forums.sifive.com/t/confusion-regarding-freedom-e-sdk-inline-asm/383
// # is stringify, 'reg' must be explictliy written out
// 'val' must be defined at compile time
// in c this means it MUST BE a define or value
// in c++ it can be define, value, or const int
#define WRITE_CSR(reg, val) \
  asm volatile ("csrrwi x0, " #reg ", %[x]\n\t" :: [x] "i" (val))

// 0x400 is the csr specified in gem5 in src/arch/riscv/register.hh
#define WRITE_MESH_CSR(val) \
  WRITE_CSR(0x400, val)

// #define BIND_EXE(val) \
//   asm volatile (".insn u 0x6b, x0, %[x]\n\t" :: [x] "i" (val))

// #define BIND_FET(val) \
//   asm volatile (".insn u 0x77, x0, %[x]\n\t" :: [x] "i" (val))
  
// 0x401 is MISCREG_FET
#define VECTOR_EPOCH(val) \
  asm volatile (".insn i 0x77, 0, x0, %[x], 0x401\n\t" :: [x] "r" (val) : "memory")

// revec instruction with unique hash id
#define REVEC(hash)                                                           \
  asm volatile (".insn u 0x7b, x0, %[id]\n\t" :: [id] "i" (hash))
  
// remem instruction with unique hash id (mem barrier instead of control barrier)
#define REMEM(count)                                                           \
  asm volatile (".insn i 0x1b, 0x2, x0, %[src0], 0\n\t":: [src0] "r" (count) : "memory")

#define FRAME_START(count)                                                     \
  asm volatile (".insn i 0x1b, 0x3, x0, %[src0], 0\n\t":: [src0] "r" (count) : "memory")

#define ISSUE_VINST(label)                                                    \
  asm volatile goto (".insn uj 0x6b, x0, %l[" #label "]\n\t"                  \
    :                                                                         \
    :                                                                         \
    :                                                                         \
    : label                                                                   \
  )

#define DEVEC(devec_id)                                                       \
  devec_id:                                                                   \
  asm volatile goto (".insn uj 0x2b, x0, %l[" #devec_id "]\n\t"               \
    :                                                                         \
    :                                                                         \
    :                                                                         \
    : devec_id                                                                \
  )

#define PREFETCH_EPOCH(val) \
  asm volatile ("csrw 0x402, %[x]\n\t" :: [x] "r" (val))
  
#define BROADCAST(dest_reg, val, imm) \
  asm volatile (".insn i 0x1b, 0x6, " #dest_reg ", %[src_reg], %[imm_val]\n\t" \
    :: [src_reg] "r" (val), [imm_val] "i" (imm)                                \
  )

// allow following instructions to proceed if registers equal
#define PRED_EQ(reg0, reg1) \
  asm volatile (".insn r 0x33, 0x7, 0x5, x0, %[rs1], %[rs2]\n\t" \
  :: [rs1] "r" (reg0), [rs2] "r" (reg1) : "memory")

// allow following instructions to proceed if registers not equal
#define PRED_NEQ(reg0, reg1) \
  asm volatile (".insn r 0x33, 0x7, 0x6, x0, %[rs1], %[rs2]\n\t" \
  :: [rs1] "r" (reg0), [rs2] "r" (reg1) : "memory")

#define TERMINATE_BLOCK() \
  asm volatile(".insn i 0x1b, 0x7, x0, x0, 0\n\t")


  // revec instruction with unique hash id
/*#define REVEC(hash)                                                           \
  asm volatile ("" ::: "memory");                                             \
  asm volatile (".insn u 0x7b, x0, %[id]\n\t" :: [id] "i" (hash) : "memory"); \
  asm volatile ("" ::: "memory");
*/
// 0x0f << 2 & 0x3 = 0x3f
// actually use, unused sw funct3
// if don't do this then compiler thinks its a 64bit instructions which
// messes up gem5
// #define VPREFETCH(spadAddr, memAddr, group_start, group_end) \
//   asm volatile (".insn sb 0x23, 0x4, %[spad], %[off](%[mem])\n\t" :: \
//     [spad] "r" (spadAddr), [mem] "r" (memAddr), [off] "i" ((group_start << 6) | (group_end - group_start)))

#define VPREFETCH_L(spadOffset, memAddr, coreOffset, count, config)   \
  asm volatile (".insn sb 0x23, 0x6, %[spad], %[off](%[mem])\n\t" ::  \
    [spad] "r" ((coreOffset << 12) | spadOffset),                     \
    [mem] "r" (memAddr),                                              \
    [off] "i" ((count << 2) | config))

#define VPREFETCH_R(spadOffset, memAddr, coreOffset, count, config)   \
  asm volatile (".insn sb 0x23, 0x7, %[spad], %[off](%[mem])\n\t" ::  \
    [spad] "r" ((coreOffset << 12) | spadOffset),                     \
    [mem] "r" (memAddr),                                              \
    [off] "i" ((count << 2) | config))

#define LWSPEC(dest, spadAddr, offset)                    \
  asm volatile (                                          \
    ".insn s 0x03, 0x7, %[destreg], %[off](%[mem])\n\t"   \
    : [destreg] "=r" (dest)                               \
    : [mem] "r" (spadAddr), [off] "i" (offset))         

#define STORE_NOACK(data, memAddr, offset) \
  asm volatile (".insn sb 0x23, 0x5, %[dataReg], %[off](%[mem])\n\t" :: \
    [dataReg] "r" (data), [mem] "r" (memAddr), [off] "i" (offset))     

  
static inline void stats_on()
{
#if !defined(__x86_64__) && !defined(__i386__)
  int on = 1;
 __asm__ volatile ("csrw 0x7C1, %0;"
                    :
                    : "r" (on)
                    :);
#endif
}

static inline void stats_off()
{
#if !defined(__x86_64__) && !defined(__i386__)
  int off = 10; // can't use 0, but anything other than 1
 __asm__ volatile ("csrw 0x7C1, %0;"
                    :
                    : "r" (off)
                    :);
#endif
}

static int getVecMask(int origin_x, int origin_y, int tid_x, int tid_y, int dim_x, int dim_y) {
  int mask = ALL_NORM;
  
  #ifndef _VEC
  return mask;
  #else
  
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
  int vlenX = dim_x;
  int vlenY = dim_y;
  mask |= (origin_x << FET_XORIGIN_SHAMT) | (origin_y << FET_YORIGIN_SHAMT) | (vlenX << FET_XLEN_SHAMT) | (vlenY << FET_YLEN_SHAMT);

  // specify each core is an execute core
  mask |= (0 << FET_DAE_SHAMT);

  return mask;
  #endif
}

// mask that guarentees a linear chain with no fanout
// implements a snake pattern
// -> -> -> v
// v <- <- <-
// -> -> -> v
// 0 <- <- <-
static int getSerializedMask(int origin_x, int origin_y, int tid_x, int tid_y, int dim_x, int dim_y) {
  int mask = ALL_NORM;
  
  #ifndef _VEC
  return mask;
  #else
  
  // each row alternates between different behavior
  if (tid_y % 2 == 0) {
    // if first column either recv from above or not at all
    if (tid_x == 0) {
      if (tid_y == 0) {
        mask |= ALL_NORM;
      }
      else {
        mask |= FET_I_INST_UP;
      }
    }
    // otherwise recv from the left
    else {
      mask |= FET_I_INST_LEFT;
    }

    // send to the right if not at edge
    if (tid_x < dim_x - 1) {
      mask |= FET_O_INST_RIGHT_SEND;
    }
    // if at the edge send down
    else {
      mask |= FET_O_INST_DOWN_SEND;
    }
  }
  else {
    // input either above if at the right edge or from the right
    if (tid_x == dim_x - 1) {
      mask |= FET_I_INST_UP;
    }
    else {
      mask |= FET_I_INST_RIGHT;
    }

    // output either to the left or down if at left edge
    if (tid_x == 0) {
      mask |= FET_O_INST_DOWN_SEND;
    }
    else {
      mask |= FET_O_INST_LEFT_SEND;
    }
  }
  
  // specify the vlen
  int vlenX = dim_x;
  int vlenY = dim_y;
  mask |= (origin_x << FET_XORIGIN_SHAMT) | (origin_y << FET_YORIGIN_SHAMT) | (vlenX << FET_XLEN_SHAMT) | (vlenY << FET_YLEN_SHAMT);

  // specify each core is an execute core
  mask |= (0 << FET_DAE_SHAMT);

  return mask;
  #endif
}

static int getDAEMask(int origin_x, int origin_y, int tid_x, int tid_y, int dim_x, int dim_y) {
  int mask = (1 << FET_DAE_SHAMT) | 
            (origin_x << FET_XORIGIN_SHAMT) | 
            (origin_y << FET_YORIGIN_SHAMT) | 
            (dim_x << FET_XLEN_SHAMT) | 
            (dim_y << FET_YLEN_SHAMT);
  return mask;
}

typedef struct Vector2_t {
  int x;
  int y;
  // int touched;
  int o;
} Vector2_t;

static int isCoordEqual(Vector2_t a, Vector2_t b) {
  return (a.x == b.x && a.y == b.y);
}

static Vector2_t addVec2(Vector2_t a, Vector2_t b) {
  Vector2_t sum = { .x = a.x + b.x, .y = a.y + b.y };
  return sum;
}

// point sample x,y
// topleft box coord x,y
// dimx > dim to right
// dimy v dim under (but maps to positively increasing)
static int pointIntersectsBox(Vector2_t pt, Vector2_t boxOrig, Vector2_t boxDim) {
  return ((pt.x >= boxOrig.x && pt.x < boxOrig.x + boxDim.x) &&
          (pt.y >= boxOrig.y && pt.y < boxOrig.y + boxDim.y));
}

static int getInputFromOutput(int outputDir) {
  if (outputDir == FET_O_INST_UP_SEND) return FET_I_INST_DOWN;
  if (outputDir == FET_O_INST_DOWN_SEND) return FET_I_INST_UP;
  if (outputDir == FET_O_INST_LEFT_SEND) return FET_I_INST_RIGHT;
  if (outputDir == FET_O_INST_RIGHT_SEND) return FET_I_INST_LEFT;
  else {
    return -1;
  }
}

// vector orientation of group with specific sending pattern
static int getSIMDMaskHoriz(Vector2_t master, Vector2_t origin, Vector2_t tid, Vector2_t dim, Vector2_t virtVectorSrc, Vector2_t vectorSrc) {
  int mask = ALL_NORM;
  // for the rest of the cores, you can determine sending pattern based on location 
  // of this core relative to vector src
  // if +/-y you recv from that respective direction
  // if you are even you recv +/-x
  int yDiff = tid.y - virtVectorSrc.y;
  int xDiff = tid.x - virtVectorSrc.x;
  // printf("vec src (%d,%d) tid (%d,%d) diffs (%d,%d)\n", virtVectorSrc.x, virtVectorSrc.y, tid.x, tid.y, xDiff, yDiff);
  // recv from above and send below unless you are at the bottom
  if (yDiff > 0) {
    mask |= FET_I_INST_UP;
    // printf("in up\n");
    if (tid.y != dim.y - 1) {
      // printf("out down\n");
      mask |= FET_O_INST_DOWN_SEND;
    }
  }
  // recv from below and send above unless you are at the top
  else if (yDiff < 0) {
    mask |= FET_I_INST_DOWN;
    // printf("in down\n");
    if (tid.y != 0) {
      // printf("out up\n");
      mask |= FET_O_INST_UP_SEND;
    }
  }
  // if you are equal then need to look at x direction
  else {
    // recv from left and send to right unless at right edge
    if (xDiff > 0) {
      if (tid.x != 0) {
        mask |= FET_I_INST_LEFT;
        // printf("in left\n");
      }
      if (tid.x != dim.x - 1) {
        // printf("out right\n");
        mask |= FET_O_INST_RIGHT_SEND;
      }
    }
    // recv from right and send to the left unless at left edge
    else if (xDiff < 0) {
      if (tid.x != dim.x - 1) {
        mask |= FET_I_INST_RIGHT;
        // printf("in right\n");
      }
      if (tid.x != 0) {
        // printf("out left\n");
        mask |= FET_O_INST_LEFT_SEND;
      }
    }
    // figure out what to do if you are at the vecSrc
    else {
      // if cores to the right, need to send to the right
      if (vectorSrc.x < origin.x + dim.x - 1) {
        // printf("out right\n");
        mask |= FET_O_INST_RIGHT_SEND;
      }
      // if cores to the left, need to send to the left
      if (vectorSrc.x > origin.x) {
        // printf("out left\n");
        mask |= FET_O_INST_LEFT_SEND;
      }
    }

    // need row at equal height to send up/down
    // if vectorSrc is above bottom of group need to send down
    if (vectorSrc.y < origin.y + dim.y - 1) {
      // printf("out down\n");
      mask |= FET_O_INST_DOWN_SEND;
    }
    // if vectorSrc is below top of group then need to send up
    if (vectorSrc.y > origin.y) {
      // printf("out up\n");
      mask |= FET_O_INST_UP_SEND;
    }
  }

  return mask;
  
}

// configuration for vector-simd group, takes up size dim+1, so be careful about planning
// master x,y --> where the master is
// origin x,y --> where the top-left core is for the trailing cores
// tid    x,y --> thread id within the group, don't care for master
// dim    x,y --> dimension of the trailing core group
// is_master  --> whether this core is the master or not
static int getSIMDMask(int master_x, int master_y, int origin_x, int origin_y, int tid_x, int tid_y, int dim_x, int dim_y, int is_master) {
  // TODO does not handle case where master is above or below due to nesting order?????????

  // pack x,y into coord struct
  Vector2_t master = { .x = master_x, .y = master_y };
  Vector2_t origin = { .x = origin_x, .y = origin_y };
  Vector2_t tid    = { .x = tid_x,    .y = tid_y    };
  Vector2_t dim    = { .x = dim_x,    .y = dim_y    };

  // initialize to no vector mask
  int mask = ALL_NORM;

  // output directions
  Vector2_t directions[4] = { {.x =  1, .y =  0, .o = FET_O_INST_RIGHT_SEND  }, 
                              {.x =  0, .y =  1, .o = FET_O_INST_DOWN_SEND   },
                              {.x = -1, .y =  0, .o = FET_O_INST_LEFT_SEND   },
                              {.x =  0, .y = -1, .o = FET_O_INST_UP_SEND     },
                            };

  // core in vector adjacent to the master core
  Vector2_t vectorSrc;

  // direction master should send
  int masterSendDir = 0;

  // find closest tile in vector group, master will send to that one
  // do this by trying each cardinal diection and seeing if intersect the vector box
  for (int i = 0; i < 4; i++) {
    Vector2_t loc = addVec2(master, directions[i]);
    if (pointIntersectsBox(loc, origin, dim)) {
      vectorSrc = loc;
      masterSendDir = directions[i].o;
    }
  }

  // the master sends to vector src and vectorSrc recvs from master
  if (is_master) {
    // if (masterSendDir == FET_O_INST_UP_SEND) printf("out up\n");
    // if (masterSendDir == FET_O_INST_DOWN_SEND) printf("out down\n");
    // if (masterSendDir == FET_O_INST_LEFT_SEND) printf("out left\n");
    // if (masterSendDir == FET_O_INST_RIGHT_SEND) printf("out right\n");
    mask |= masterSendDir;
  }
  else {
    // make sure vectorSrc is virtualized within the group
    Vector2_t virtVecSrc = { .x = vectorSrc.x - origin.x, .y = vectorSrc.y - origin.y };
    if (isCoordEqual(virtVecSrc, tid)) {
      // if (getInputFromOutput(masterSendDir) == FET_I_INST_UP) printf("in up\n");
      // if (getInputFromOutput(masterSendDir) == FET_I_INST_DOWN) printf("in down\n");
      // if (getInputFromOutput(masterSendDir) == FET_I_INST_LEFT) printf("in left\n");
      // if (getInputFromOutput(masterSendDir) == FET_I_INST_RIGHT) printf("in right\n");
      mask |= getInputFromOutput(masterSendDir);
    }

    // send directions for the non-master vector group
    mask |= getSIMDMaskHoriz(master, origin, tid, dim, virtVecSrc, vectorSrc);
  }

  // specify the vlen
  int vlenX = dim_x;
  int vlenY = dim_y;
  mask |= (origin_x << FET_XORIGIN_SHAMT) | (origin_y << FET_YORIGIN_SHAMT) | (vlenX << FET_XLEN_SHAMT) | (vlenY << FET_YLEN_SHAMT);

  // specify each core is an execute core
  mask |= (is_master << FET_DAE_SHAMT);

  return mask;
}

// https://stackoverflow.com/questions/3407012/c-rounding-up-to-the-nearest-multiple-of-a-number
int roundUp(int numToRound, int multiple) {
  if (multiple == 0) {
    return numToRound;
  }

  int remainder = abs(numToRound) % multiple;
  if (remainder == 0) {
    return numToRound;
  }

  if (numToRound < 0) {
    return -(abs(numToRound) - remainder);
  }
  else {
    return numToRound + multiple - remainder;
  }
}

// design a rectangular vector group with an attached scalar core
inline void rect_vector_group(
    int group_num, int scalar_x, int scalar_y, int vector_start_x, int vector_start_y, int vector_dim_x, int vector_dim_y, int id_x, int id_y, 
    int n, int vGroups, int alignment, int chunk_offset,
    int *vtid_x, int *vtid_y, int *is_scalar, int *orig_x, int *orig_y, int *master_x, int *master_y, int *used, int *start, int *end) {
  
  int vector_end_x = vector_start_x + vector_dim_x;
  int vector_end_y = vector_start_y + vector_dim_y;

  int is_vector_group = id_x >= vector_start_x && id_x < vector_end_x && 
    id_y >= vector_start_y && id_y < vector_end_y;

  int is_scalar_group = id_x == scalar_x && id_y == scalar_y;
  if (is_vector_group) {
    *vtid_x = id_x - vector_start_x;
    *vtid_y = id_y - vector_start_y;
  }
  if (is_scalar_group) {
    *is_scalar = 1;
  }
  if (is_vector_group || is_scalar_group) {
    *start = roundUp((chunk_offset + group_num + 0) * n / vGroups, alignment);
    *end   = roundUp((chunk_offset + group_num + 1) * n / vGroups, alignment); // make sure aligned to cacheline 
    *orig_x = vector_start_x;
    *orig_y = vector_start_y;
    *master_x = scalar_x;
    *master_y = scalar_y;
    *used = 1;
  }
}


// if don't inline then need to copy stack pointer up to addr 88, which too lazy to do atm
// create a template for a vlen=4 config that can copy and paste multiple times on a large mesh
// ret whether core used in template
inline int vector_group_template_4(
    // inputs
    int ptid_x, int ptid_y, int pdim_x, int pdim_y, int n,
    // outputs
    int *vtid, int *vtid_x, int *vtid_y, int *is_scalar, int *orig_x, int *orig_y, int *master_x, int *master_y,
    int *start, int *end
  ) {

  // keep track of which cores will be used in this configuration
  // will want to terminate any cores not apart of a vector group
  int used = 0;

  // virtual group dimension
  int vdim_x = 2;
  int vdim_y = 2;

  // recover trivial fields
  int vdim = vdim_x * vdim_y;
  int ptid = ptid_x + ptid_y * pdim_x;
  int pdim = pdim_x * pdim_y;

  // this is a design for a 4x4 zone
  // potentially there are more than one 4x4 zones in the mesh
  // get ids within the template
  int template_dim_x = 4;
  int template_dim_y = 4;
  int template_dim = template_dim_x * template_dim_y;
  int template_id_x = ptid_x % template_dim_x;
  int template_id_y = ptid_y % template_dim_y;
  int template_id = template_id_x + template_id_y * template_dim_x;

  // which group it belongs to for absolute core coordinates
  int template_group_x = ptid_x / template_dim_x;
  int template_group_y = ptid_y / template_dim_y;
  int template_group_dim_x = pdim_x / template_dim_x;
  int template_group_dim_y = pdim_y / template_dim_y;
  int template_group_dim = template_group_dim_x * template_group_dim_y;
  int template_group = template_group_x + template_group_y * template_group_dim_x;

  // figure out how big chunks of the data should be assigned
  int alignment = 16 * vdim_x * vdim_y;
  int groupSize = vdim + 1; // +scalar core
  int groups_per_template = template_dim / groupSize;
  int vGroups = groups_per_template * template_group_dim;

  int chunk_offset = template_group  * groups_per_template;

  // group 1 top left (master = 0)
  rect_vector_group(0, 0, 0, 1, 0,
    vdim_x, vdim_y, template_id_x, template_id_y, n, vGroups, alignment, chunk_offset,
    vtid_x, vtid_y, is_scalar, orig_x, orig_y, master_x, master_y, &used, start, end);

  // group 2 bot left (master == 4)
  rect_vector_group(1, 0, 1, 0, 2,
    vdim_x, vdim_y, template_id_x, template_id_y, n, vGroups, alignment, chunk_offset,
    vtid_x, vtid_y, is_scalar, orig_x, orig_y, master_x, master_y, &used, start, end);

  // group 3 bottom right (master == 7)
  rect_vector_group(2, 3, 1, 2, 2,
    vdim_x, vdim_y, template_id_x, template_id_y, n, vGroups, alignment, chunk_offset,
    vtid_x, vtid_y, is_scalar, orig_x, orig_y, master_x, master_y, &used, start, end);

  // need to shift the absolute coordinates based on which group this is for
  *orig_x = *orig_x + template_group_x * template_dim_x;
  *orig_y = *orig_y + template_group_y * template_dim_y;
  *master_x = *master_x + template_group_x * template_dim_x;
  *master_y = *master_y + template_group_y * template_dim_y;

  // handle unused cores
  *vtid = *vtid_x + *vtid_y * vdim_x;

  return used;
  
}

inline int vector_group_template_16(
    // inputs
    int ptid_x, int ptid_y, int pdim_x, int pdim_y, int n,
    // outputs
    int *vtid, int *vtid_x, int *vtid_y, int *is_scalar, int *orig_x, int *orig_y, int *master_x, int *master_y,
    int *start, int *end
  ) {

  // keep track of which cores will be used in this configuration
  // will want to terminate any cores not apart of a vector group
  int used = 0;

  // virtual group dimension
  int vdim_x = 4;
  int vdim_y = 4;

  // recover trivial fields
  int vdim = vdim_x * vdim_y;
  int ptid = ptid_x + ptid_y * pdim_x;
  int pdim = pdim_x * pdim_y;

  // this is a design for a 8x8 zone
  // potentially there are more than one 8x8 zones in the mesh
  // get ids within the template
  int template_dim_x = 8;
  int template_dim_y = 8;
  int template_dim = template_dim_x * template_dim_y;
  int template_id_x = ptid_x % template_dim_x;
  int template_id_y = ptid_y % template_dim_y;
  int template_id = template_id_x + template_id_y * template_dim_x;

  // which group it belongs to for absolute core coordinates
  int template_group_x = ptid_x / template_dim_x;
  int template_group_y = ptid_y / template_dim_y;
  int template_group_dim_x = pdim_x / template_dim_x;
  int template_group_dim_y = pdim_y / template_dim_y;
  int template_group_dim = template_group_dim_x * template_group_dim_y;
  int template_group = template_group_x + template_group_y * template_group_dim_x;

  // figure out how big chunks of the data should be assigned
  int alignment = 16 * vdim_x * vdim_y;
  int groupSize = vdim + 1; // +scalar core
  int groups_per_template = template_dim / groupSize;
  int vGroups = groups_per_template * template_group_dim;

  int chunk_offset = template_group  * groups_per_template;

  // group 1 top left (master = 0,4)
  rect_vector_group(0, 0, 4, 0, 0,
    vdim_x, vdim_y, template_id_x, template_id_y, n, vGroups, alignment, chunk_offset,
    vtid_x, vtid_y, is_scalar, orig_x, orig_y, master_x, master_y, &used, start, end);

  // group 2 top right (master = 7, 4)
  rect_vector_group(1, 7, 4, 4, 0,
    vdim_x, vdim_y, template_id_x, template_id_y, n, vGroups, alignment, chunk_offset,
    vtid_x, vtid_y, is_scalar, orig_x, orig_y, master_x, master_y, &used, start, end);

  // group 3 middle (master 1, 4)
  rect_vector_group(2, 1, 4, 2, 4,
    vdim_x, vdim_y, template_id_x, template_id_y, n, vGroups, alignment, chunk_offset,
    vtid_x, vtid_y, is_scalar, orig_x, orig_y, master_x, master_y, &used, start, end);  

  // need to shift the absolute coordinates based on which group this is for
  *orig_x = *orig_x + template_group_x * template_dim_x;
  *orig_y = *orig_y + template_group_y * template_dim_y;
  *master_x = *master_x + template_group_x * template_dim_x;
  *master_y = *master_y + template_group_y * template_dim_y;

  // handle unused cores (51/64 cores used)
  *vtid = *vtid_x + *vtid_y * vdim_x;

  return used;
  
}


#endif
  