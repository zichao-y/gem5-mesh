#include <stdatomic.h>
extern const float valA[];
extern const float valB[];
extern const float valC[];
extern const int ptrA[];
extern const int idxA[];
extern const int ptrB[];
extern const int idxB[];
extern const int ptrC[];
extern const int idxC[];
extern const int mat_m;
extern const int mat_n;
extern const int mat_k;
extern const int matA_nnz;
extern const int matB_nnz;
extern const int matC_nnz;
extern int ini_acc_val;
extern atomic_int workq;
extern atomic_int write_q;
extern atomic_int val_q;
extern atomic_int idx_q;
extern atomic_int compen[];