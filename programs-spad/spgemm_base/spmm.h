#ifndef __GEMM_H__
#define __GEMM_H__

// #define VECTOR_LEN 16
//#ifdef VECTOR_LEN
//#define _VEC
//#endif


#define FILENAME_SZ 1024
#define PARA_SZ 16
#define BURST_LEN 16
#define QUEUE_SIZE 640


//tile size
#ifndef BLK_DIM
#if defined(PER_CORE_SIMD) || defined(PER_CORE_SIMD)
// match blk dim to simd length for ease of use
// 16 * 16 * 4 = 1kB space so fits in 4kB spad
#define BLK_DIM HARDWARE_VECTOR_LEN
#else
#define BLK_DIM 32
#endif
#endif



#ifndef _N_SPS
#define _N_SPS 64
#endif

#if VECTOR_LEN==4
#define DIM_X 2
#elif VECTOR_LEN==16
#define DIM_X 4
#endif

// #define VEC_MANYCORE_OPT

#if defined VEC_MANYCORE_OPT && defined _VEC
#define MANYCORE_PREFETCH
#endif

#define ALPHA 32412.0f
#define BETA 2123.0f

// #define SHARING
// #define C_PREFETCH 
// #define MANYCORE_PREFETCH

#ifndef INIT_FRAMES
#ifdef PER_CORE_SIMD
#define INIT_FRAMES 1
#else
#define INIT_FRAMES 2
#endif
#endif

#ifdef SHARING
#define REGION_SIZE (BLK_DIM*2)/DIM_X
#define NUM_REGIONS (512 / REGION_SIZE)
#else
#define REGION_SIZE (BLK_DIM * 2)
#define NUM_REGIONS (512 / REGION_SIZE)
#endif

typedef float DTYPE;

void spmm_manycore(const float *A_val, const float *B_val, float *C_val, float **C_inter_val, int **C_inter_idx, const int *A_idx, const int *A_ptr, const int *B_idx, const int *B_ptr, int *C_idx, int *C_ptr,int m, int n, int k, int ptid, int num_cores, float *C_valout_tmp, int *C_idxout_tmp,float** helper_queue_valarray,int** helper_queue_idxarray, float *C_val_extend, int *C_idx_extend);


#if VECTOR_LEN==4 && _N_SPS==64
#define WORK_DIV(m,n) \
  int uid_x,uid_y; \
  int tg_x,tg_y; \
  tg_x = 4; \
  tg_y = 3; \
  uid_x = cinfo.unique_id%tg_x; \
  uid_y = cinfo.unique_id/tg_x; \
  if(cinfo.used) { \
    int alignment = BLK_DIM * vdim_x; \
    m_start = roundUp((uid_y + 0) * m / tg_y, alignment); \
    m_end = roundUp((uid_y + 1) * m / tg_y, alignment); \
    n_start = roundUp((uid_x + 0) * n / tg_x, alignment); \
    n_end = roundUp((uid_x + 1) * n / tg_x, alignment); \
  }

#else

#define WORK_DIV(m,n) \
  int uid_x,uid_y; \
  int tg_x,tg_y; \
  tg_x = 1; \
  tg_y = 3; \
  uid_x = cinfo.unique_id%tg_x; \
  uid_y = cinfo.unique_id/tg_x; \
  if(cinfo.used) { \
    int alignment = BLK_DIM * vdim_x; \
    m_start = roundUp((uid_y + 0) * m / tg_y, alignment); \
    m_end = roundUp((uid_y + 1) * m / tg_y, alignment); \
    n_start = roundUp((uid_x + 0) * n / tg_x, alignment); \
    n_end = roundUp((uid_x + 1) * n / tg_x, alignment); \
  }

#endif

#define PTID_FINDER(ptid)\
  if(ptid==38) ptid_new=0;\
  else if(ptid==40) ptid_new=1;\
  else if(ptid==41) ptid_new=2;\
  else if(ptid==46) ptid_new=3;\
  else if(ptid==47) ptid_new=4;\
  else if(ptid==48) ptid_new=5;\
  else if(ptid==49) ptid_new=6;\
  else if(ptid==54) ptid_new=7;\
  else if(ptid==55) ptid_new=8;\
  else if(ptid==56) ptid_new=9;\
  else if(ptid==57) ptid_new=10;\
  else if(ptid==62) ptid_new=11;\
  else if(ptid==63) ptid_new=12;

#define WORK_DIV_OPT(m,n) \
  int total_groups = 3; \
  int unused = _N_SPS - (total_groups*(VECTOR_LEN+1)); \
  int total_compute_cores = (total_groups*VECTOR_LEN) + unused; \
  int alignment = BLK_DIM * DIM_X; \
  int m_vec = roundUp((total_groups*VECTOR_LEN*m)/total_compute_cores, alignment);\
  m_manycore = m-m_vec; \
  if(cinfo.used) { \
    alignment = BLK_DIM * DIM_X; \
    m_start = roundUp((cinfo.unique_id + 0) * m_vec / total_groups, alignment); \
    m_end = roundUp((cinfo.unique_id + 1) * m_vec / total_groups, alignment); \
    n_start = 0; \
    n_end = n; \
  }\
  else if(ptid_new>0){\
    ptid_new-=1;\
    int tg_x=6;\
    int tg_y=2;\
    int utid_x = ptid_new%tg_x;\
    int utid_y = ptid_new/tg_x;\
    alignment = BLK_DIM; \
    m_start = m_vec + roundUp((utid_y + 0) * m_manycore / tg_y, alignment); \
    m_end = m_vec+ roundUp((utid_y + 1) * m_manycore / tg_y, alignment); \
    n_start = roundUp((utid_x + 0) * n / tg_x, alignment); \
    n_end = roundUp((utid_x + 1) * n / tg_x, alignment); \
  }\
  else\
    m_start=m_end=n_start=n_end=0;

// pthread argument for the kernel
typedef struct Kern_Args
{
  const float *A_val, *B_val; 
  float *C_valout_tmp,*C_val, *C_val_extend;
  float **C_inter_val, **helper_queue_valarray;
  const int  *A_ptr, *A_idx, *B_ptr, *B_idx;
  int *C_ptr,*C_idx,*C_idxout_tmp, *C_idx_extend; 
  int **C_inter_idx, **helper_queue_idxarray;
  int m, n, k;
  int ptid, error;
  int num_cores;
} Kern_Args;

// helper to pack vvadd args
Kern_Args *construct_args(
    const float *A_val, const float *B_val, float *C_val, float **C_inter_val, int **C_inter_idx, const int *A_idx, const int *A_ptr, const int *B_idx, const int *B_ptr, int *C_idx, int *C_ptr,int m, int n,
     int k, int ptid, int num_cores, float *C_valout_tmp, int *C_idxout_tmp,float** helper_queue_valarray,int** helper_queue_idxarray,float *C_val_extend, int *C_idx_extend);

// pthread call
void *pthread_kernel(void *args);

// vvadd kernel
void kernel(
    const float *A_val, const float *B_val, float *C_val, float **C_inter_val, int **C_inter_idx, const int *A_idx, const int *A_ptr, const int *B_idx, const int *B_ptr, int *C_idx, int *C_ptr, int m, int n,
     int k, int ptid, int num_cores, float *C_valout_tmp, int *C_idxout_tmp,float** helper_queue_valarray,int** helper_queue_idxarray,float *C_val_extend, int *C_idx_extend);

#endif
