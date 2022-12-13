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
//spmm_manycore(const float *A_val, const float *B_val, float *C_val, float **C_inter_val, int **C_inter_idx, const int *A_idx, const int *A_ptr, const int *B_idx, const int *B_ptr, int *C_idx, int *C_ptr,int m, int n,int k, int ptid, int num_cores, int ini_acc_val, int *idx_ref, float *val_ref, int *ptr_ref, float *C_valout_tmp, int *C_idxout_tmp)
spmm_manycore(const float *A_val, const float *B_val, float *C_val, float **C_inter_val, int **C_inter_idx, const int *A_idx, const int *A_ptr, const int *B_idx, const int *B_ptr, int *C_idx, int *C_ptr,int m, int n,int k, int ptid, int num_cores, float *C_valout_tmp, int *C_idxout_tmp,float** helper_queue_valarray,int** helper_queue_idxarray,float *C_val_extend, int *C_idx_extend)
{
  //three value queue in the scratchpad
  

  DTYPE *val_queue = (DTYPE *)getSpAddr(ptid, 0);
  //three idx queue in the scratchpad
  int *idx_queue = (int *)(val_queue+QUEUE_SIZE*3);
  //initiate complementary queues in DRAM in case scratchpad queues overflow
  int QUEUE_SIZE_DRAM = mat_k-QUEUE_SIZE;
  //float * val_queue_DRAM_ptr; 
  //float* val_queue_DRAM = (float*)malloc_cache_aligned(sizeof(float),QUEUE_SIZE_DRAM*3,(void**)&val_queue_DRAM_ptr);
  //int * idx_queue_DRAM_ptr; 
  //int* idx_queue_DRAM = (int*)malloc_cache_aligned(sizeof(int),QUEUE_SIZE_DRAM*3,(void**)&idx_queue_DRAM_ptr);
  float* val_queue_DRAM = helper_queue_valarray[ptid];
  int* idx_queue_DRAM = helper_queue_idxarray[ptid];
  int queue_len[3];

  //printf("core %d started!!!!!!!!!!!!!!!!!!!!!!!!!\n",ptid);
  
  //printf("ptid: %d  val_queue addr is: %d, idx_queue addr is: %d\n",ptid,val_queue,idx_queue);
  //pthread_barrier_wait(&start_barrier);
  //row-wise product SpgeMM
  //if(ptid==0) printf("arguments: A_val:%d, B_val:%d, C_val:%d, C_inter_val:%d, C_inter_idx:%d, A_idx:%d, A_ptr:%d, B_idx:%d, B_ptr:%d, C_idx:%d, C_ptr:%d, m:%d, n:%d, k:%d, ptid:%d, num_core:%d, C_valout_tmp:%d, C_idxout_tmp:%d,helper_queue_valarray:%d,helper_queue_idxarray:%d,C_val_extend:%d, C_idx_extend:%d\n",A_val, B_val, C_val, C_inter_val, C_inter_idx, A_idx, A_ptr, B_idx, B_ptr, C_idx, C_ptr,m,n,k,ptid,num_cores,C_valout_tmp, C_idxout_tmp,helper_queue_valarray,helper_queue_idxarray,C_val_extend, C_idx_extend);


  for(int i=atomic_fetch_add_explicit(&workq, 1, memory_order_relaxed); i<m; i =atomic_fetch_add_explicit(&workq, 1, memory_order_relaxed)){
    int first = 1;
    int wr_buf = 0;
    int rd_buf = 0;
    int merge_buf = 1;
    int out_len;
    int A_start=A_ptr[i];
    int A_end=A_ptr[i+1];
    int sp_b_val_offset;
    int sp_b_idx_offset;
    //if row A is empty, queue_len[0] will stop writing to C
    queue_len[0] = 0;
    //printf("Process for row[%d] in A; From core %d\n",i,ptid);
    //printf("enter kernel!\n");
    //fetch and merge rows in B matrix
    for(int j=A_start; j<A_end; j++){
      float a_val = A_val[j];
      int a_idx = A_idx[j];
      //load row[a_idx] of B
      int B_start = B_ptr[a_idx];
      int B_end = B_ptr[a_idx+1];
      int B_load_pass = (B_end - B_start)<QUEUE_SIZE ? (B_end - B_start + 15)/16 : QUEUE_SIZE/16;
      queue_len[wr_buf] = B_end - B_start;
      //printf("length of ROW[%d] in B is %d, From core %d during processing row[%d] in A \n",a_idx,queue_len[wr_buf],ptid,i);
      sp_b_val_offset = wr_buf*QUEUE_SIZE;
      sp_b_idx_offset = QUEUE_SIZE*3+wr_buf*QUEUE_SIZE;
      REMEM();
      for(int round=0; round<B_load_pass; round++){
        VPREFETCH_LR(sp_b_val_offset+round*16, B_val+B_start+round*16, 0, 16,TO_SELF);
        VPREFETCH_LR(sp_b_idx_offset+round*16, B_idx+B_start+round*16, 0, 16,TO_SELF);
      }
      FRAME_START(32*B_load_pass);
      
      

      if (first==1){
        first = 0;
        //for(int r=0; r<QUEUE_SIZE;r++){
          //printf("First Load: idx_queue[%d]: %d\n",r,idx_queue[r]);
          //printf("First Load: val_B[%d]: %f\n",r,B_val[B_start+r]);
        //}
        int rd_bund = (queue_len[wr_buf] > QUEUE_SIZE) ? QUEUE_SIZE : B_end-B_start;
        for(int row_idx=0; row_idx<rd_bund; row_idx++){
          val_queue[row_idx] = val_queue[row_idx] * a_val;
        }
        //if row length > QUEUE_SIZE, need to move data from original place to the DRAM helper queue
        if(queue_len[wr_buf] > QUEUE_SIZE){
          int remain = queue_len[wr_buf] - QUEUE_SIZE;
          int B_offset=B_start + QUEUE_SIZE;
          int queue_ptr=0;
          //float *val_queue_DRAM = helper_queue_valarray[ptid];
          //int *idx_queue_DRAM = helper_queue_idxarray[ptid];
          //queue[wr_buf] already used for storing the first part of B row, use a different scratchpad queue (queue 2) to do intermediate transport
          int queue_val_sp_offset = QUEUE_SIZE*2;
          int queue_idx_sp_offset = QUEUE_SIZE*3+QUEUE_SIZE*2;
          int queue_DRAM_offset = QUEUE_SIZE_DRAM*wr_buf;
          
          while(remain>0){
            int load_pass = remain > QUEUE_SIZE ? QUEUE_SIZE/16 : (remain +15)/16;              
            REMEM();
            for(int round=0; round<load_pass; round++){
              VPREFETCH_LR(queue_val_sp_offset+round*16,B_val+B_offset+round*16, 0, 16,TO_SELF);
              VPREFETCH_LR(queue_idx_sp_offset+round*16,B_idx+B_offset+round*16, 0, 16,TO_SELF);
            }
            FRAME_START(32*load_pass);
            B_offset +=16*load_pass;
            int wr_bund = remain > QUEUE_SIZE ? QUEUE_SIZE : remain;
            for(int wr_idx=0; wr_idx<wr_bund; wr_idx++){
              FSTORE_NOACK(val_queue[QUEUE_SIZE*2+wr_idx]* a_val,  val_queue_DRAM+queue_DRAM_offset+queue_ptr, 0);
              STORE_NOACK(idx_queue[QUEUE_SIZE*2+wr_idx], idx_queue_DRAM+queue_DRAM_offset+queue_ptr, 0);
              //printf("First fetch for row %d:store index %d in DRAM index queue[%d]\n",i,idx_queue[QUEUE_SIZE*2+wr_idx],queue_DRAM_offset+queue_ptr);
              queue_ptr ++;
            }
            remain -= wr_bund;
          }
        }
        wr_buf = 2;         
        continue;
      }
      else{
        //merge two rows
        int queue_a_head = QUEUE_SIZE*wr_buf;
        int queue_b_head = QUEUE_SIZE*rd_buf;
        int queue_c_head = QUEUE_SIZE*merge_buf;
        int queue_a_head_DRAM = QUEUE_SIZE_DRAM*wr_buf;
        int queue_b_head_DRAM = QUEUE_SIZE_DRAM*rd_buf;
        int queue_c_head_DRAM = QUEUE_SIZE_DRAM*merge_buf;
        int queue_a_ptr=0;
        int queue_b_ptr=0;
        int queue_c_ptr=0;

        queue_len[merge_buf]=0;
        if(queue_len[wr_buf]<=QUEUE_SIZE && queue_len[rd_buf]<=QUEUE_SIZE){
          while (queue_len[wr_buf]>0 || queue_len[rd_buf] >0){
            if(queue_len[wr_buf]>0 && queue_len[rd_buf] >0){            
              if(idx_queue[queue_a_head]<idx_queue[queue_b_head]){
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_a_head];
                  val_queue[queue_c_head] = a_val * val_queue[queue_a_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(a_val*val_queue[queue_a_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_a_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  queue_c_head_DRAM ++;
                }
                queue_a_head ++;
                queue_len[wr_buf] --;
                queue_len[merge_buf] ++;
                //printf("case0:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case0:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);

              }
              else if(idx_queue[queue_a_head]>idx_queue[queue_b_head]){
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_b_head];
                  val_queue[queue_c_head] = val_queue[queue_b_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(val_queue[queue_b_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_b_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  queue_c_head_DRAM ++;
                }
                queue_b_head ++;
                queue_len[rd_buf] --;
                queue_len[merge_buf] ++;
                //printf("case1:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case1:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);
              }
              else{
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_a_head];
                  val_queue[queue_c_head] = a_val * val_queue[queue_a_head]+val_queue[queue_b_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(a_val * val_queue[queue_a_head]+val_queue[queue_b_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_a_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  queue_c_head_DRAM ++;
                }
                queue_a_head ++;
                queue_b_head ++;
                queue_len[wr_buf] --;
                queue_len[rd_buf] --;
                queue_len[merge_buf] ++;
                //printf("case2:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case2:idx queue [%d] is: %d\n",queue_a_head,idx_queue[queue_a_head]);
                //printf("case2:val_a is: %f, queue_a is: %f, queue_b is: %f\n",a_val,val_queue[queue_a_head],val_queue[queue_b_head]);

              }
              //printf("finish one compare, wr_buf length: %d, read_buf length: %d\n",queue_len[wr_buf],queue_len[rd_buf]);
            }

            else if(queue_len[wr_buf]>0 ){
              for(int idx=0; idx<queue_len[wr_buf]; idx++){
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_a_head];
                  val_queue[queue_c_head] = a_val * val_queue[queue_a_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(a_val * val_queue[queue_a_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_a_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  queue_c_head_DRAM ++;
                }
                queue_a_head ++;
                queue_len[wr_buf] --;
                queue_len[merge_buf] ++;
                //printf("case3:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case3:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);
              }
            }
            else{
              for(int idx=0; idx<queue_len[rd_buf]; idx++){
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_b_head];
                  val_queue[queue_c_head] = val_queue[queue_b_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(val_queue[queue_b_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_b_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  queue_c_head_DRAM ++;
                }
                queue_b_head ++;
                queue_len[rd_buf] --;
                queue_len[merge_buf] ++;
                //printf("case4:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case4:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);
              }
            }
          }
          wr_buf = (wr_buf+1)%3;
          rd_buf = (rd_buf+1)%3;
          merge_buf = (merge_buf+1)%3;
          //REMEM();
          
        
        }   
        else{
          while (queue_len[wr_buf]>0 || queue_len[rd_buf] >0){
            if(queue_len[wr_buf]>0 && queue_len[rd_buf] >0){ 
              if(idx_queue[queue_a_head]<idx_queue[queue_b_head]){
                //printf("queue a head:%d, queue b head:%d,idx a:%d, idx b:%d,row:%d\n",queue_a_head,queue_b_head,idx_queue[queue_a_head],idx_queue[queue_b_head],i);
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_a_head];
                  val_queue[queue_c_head] = a_val * val_queue[queue_a_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(a_val*val_queue[queue_a_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_a_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  //printf("idx a < idx b: write idx: %d to DRAM idx queue[%d],length a:%d, length b:%d\n",idx_queue[queue_a_head],queue_c_head_DRAM,queue_len[wr_buf],queue_len[rd_buf]);
                  queue_c_head_DRAM ++;
                }
                queue_a_head ++;
                queue_a_ptr ++;
                queue_len[wr_buf] --;
                queue_len[merge_buf] ++;

                //queue_a pointer move forward, check whether it exceeds the QUEUE_SIZE limit
                //if yes, then read from DRAM to queue_a
                if(queue_len[wr_buf]>0 && queue_a_head-QUEUE_SIZE*wr_buf>=QUEUE_SIZE){
                  int queue_a_load_pass = queue_len[wr_buf] > QUEUE_SIZE ? QUEUE_SIZE/16 : (queue_len[wr_buf] +15)/16;
                  
                  int queue_val_sp_offset = QUEUE_SIZE*wr_buf;
                  int queue_idx_sp_offset = QUEUE_SIZE*3+QUEUE_SIZE*wr_buf;
                  int B_offset = B_start + queue_a_ptr ;

                  REMEM();
                  for(int round=0; round<queue_a_load_pass; round++){
                    VPREFETCH_LR(queue_val_sp_offset+round*16, B_val+B_offset+round*16, 0, 16,TO_SELF);
                    VPREFETCH_LR(queue_idx_sp_offset+round*16, B_idx+B_offset+round*16, 0, 16,TO_SELF);
                  }
                  FRAME_START(32*queue_a_load_pass);

                  //reset queue_a_head to the beginning of the queue
                  queue_a_head = QUEUE_SIZE*wr_buf;
                  //printf("Line 278:load write queue using data from DRAM queue, remain queue a length: %d; From core %d\n",queue_len[wr_buf],ptid);
                }
                //printf("case0:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case0:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);

              }
              else if(idx_queue[queue_a_head]>idx_queue[queue_b_head]){
                
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_b_head];
                  val_queue[queue_c_head] = val_queue[queue_b_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(val_queue[queue_b_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_b_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  //printf("idx a > idx b: write idx: %d to DRAM idx queue[%d],length a:%d, length b:%d\n",idx_queue[queue_b_head],queue_c_head_DRAM,queue_len[wr_buf],queue_len[rd_buf]);
                  queue_c_head_DRAM ++;
                }
                queue_b_head ++;
                queue_b_ptr ++;
                queue_len[rd_buf] --;
                queue_len[merge_buf] ++;

                //queue_b pointer move forward, check whether it exceeds the QUEUE_SIZE limit
                //if yes, then read from DRAM to queue_b
                if(queue_len[rd_buf]>0 && queue_b_head-QUEUE_SIZE*rd_buf>=QUEUE_SIZE){
                  int queue_b_load_pass = queue_len[rd_buf] > QUEUE_SIZE ? QUEUE_SIZE/16 : (queue_len[rd_buf] +15)/16;
                  
                  int queue_val_sp_offset = QUEUE_SIZE*rd_buf;
                  int queue_idx_sp_offset = QUEUE_SIZE*3+QUEUE_SIZE*rd_buf;
                  int queue_DRAM_offset = QUEUE_SIZE_DRAM*rd_buf + queue_b_ptr - QUEUE_SIZE;

                  REMEM();
                  for(int round=0; round<queue_b_load_pass; round++){
                    VPREFETCH_LR(queue_val_sp_offset+round*16, val_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
                    VPREFETCH_LR(queue_idx_sp_offset+round*16, idx_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
                  }
                  FRAME_START(32*queue_b_load_pass);

                  //reset queue_a_head to the beginning of the queue
                  queue_b_head = QUEUE_SIZE*rd_buf;
                  //printf("Line 320:load rd queue using data from DRAM queue; From core %d\n",ptid);
                }



                //printf("case1:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case1:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);
              }
              else{
                
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_a_head];
                  val_queue[queue_c_head] = a_val * val_queue[queue_a_head]+val_queue[queue_b_head];
                  queue_c_head ++;
                  //printf("equal----------------write val %d to queue index %d, with idx %d, for row %d\n",val_queue[queue_c_head],queue_c_head,idx_queue[queue_c_head],i);
                }
                else{
                  FSTORE_NOACK(a_val * val_queue[queue_a_head]+val_queue[queue_b_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_a_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  //printf("idx a == idx b: write idx: %d to DRAM idx queue[%d],length a:%d, length b:%d\n",idx_queue[queue_a_head],queue_c_head_DRAM,queue_len[wr_buf],queue_len[rd_buf]);
                  queue_c_head_DRAM ++;
                  
                }
                queue_a_head ++;
                queue_b_head ++;
                queue_a_ptr ++;
                queue_b_ptr ++;
                queue_len[wr_buf] --;
                queue_len[rd_buf] --;
                queue_len[merge_buf] ++;

                //queue_a pointer move forward, check whether it exceeds the QUEUE_SIZE limit
                //if yes, then read from DRAM to queue_a
                if(queue_len[wr_buf]>0 && queue_a_head-QUEUE_SIZE*wr_buf>=QUEUE_SIZE){
                  int queue_a_load_pass = queue_len[wr_buf] > QUEUE_SIZE ? QUEUE_SIZE/16 : (queue_len[wr_buf] +15)/16;
                  
                  int queue_val_sp_offset = QUEUE_SIZE*wr_buf;
                  int queue_idx_sp_offset = QUEUE_SIZE*3+QUEUE_SIZE*wr_buf;
                  int B_offset = B_start + queue_a_ptr ;
                  
                  REMEM();
                  for(int round=0; round<queue_a_load_pass; round++){
                    VPREFETCH_LR(queue_val_sp_offset+round*16, B_val+B_offset+round*16, 0, 16,TO_SELF);
                    VPREFETCH_LR(queue_idx_sp_offset+round*16, B_idx+B_offset+round*16, 0, 16,TO_SELF);
                  }
                  FRAME_START(32*queue_a_load_pass);

                  //reset queue_a_head to the beginning of the queue
                  queue_a_head = QUEUE_SIZE*wr_buf;
                  //printf("Line 367:load write queue using data from DRAM queue; From core %d\n",ptid);
                }

                if(queue_len[rd_buf]>0 && queue_b_head-QUEUE_SIZE*rd_buf>=QUEUE_SIZE){
                  //printf("In branch idx a == idx b, read from DRAM to queue b, rd queue length is: %d\n",queue_len[rd_buf]);
                  int queue_b_load_pass = queue_len[rd_buf] > QUEUE_SIZE ? QUEUE_SIZE/16 : (queue_len[rd_buf] +15)/16;
                  
                  int queue_val_sp_offset = QUEUE_SIZE*rd_buf;
                  int queue_idx_sp_offset = QUEUE_SIZE*3+QUEUE_SIZE*rd_buf;
                  int queue_DRAM_offset = QUEUE_SIZE_DRAM*rd_buf + queue_b_ptr - QUEUE_SIZE;
                  

                  REMEM();
                  for(int round=0; round<queue_b_load_pass; round++){
                    VPREFETCH_LR(queue_val_sp_offset+round*16, val_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
                    VPREFETCH_LR(queue_idx_sp_offset+round*16, idx_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
                  }
                  FRAME_START(32*queue_b_load_pass);

                  //reset queue_a_head to the beginning of the queue
                  queue_b_head = QUEUE_SIZE*rd_buf;
                  //printf("Line 387:load rd queue using data from DRAM queue starting from [%d]; From core %d\n",queue_DRAM_offset,ptid);
                }




                //printf("case2:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case2:idx queue [%d] is: %d\n",queue_a_head,idx_queue[queue_a_head]);
                //printf("case2:val_a is: %f, queue_a is: %f, queue_b is: %f\n",a_val,val_queue[queue_a_head],val_queue[queue_b_head]);

              }
            }
             
            else if(queue_len[wr_buf]>0 ){
              int remain_in_buf=QUEUE_SIZE*wr_buf+QUEUE_SIZE-queue_a_head;
              int rd_bund=(remain_in_buf >= queue_len[wr_buf])?queue_len[wr_buf] : remain_in_buf;
              for(int idx=0; idx<rd_bund; idx++){
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_a_head];
                  val_queue[queue_c_head] = a_val * val_queue[queue_a_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(a_val * val_queue[queue_a_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_a_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  //printf("only queue a left: write idx: %d to DRAM idx queue[%d]\n",idx_queue[queue_a_head],queue_c_head_DRAM);
                  queue_c_head_DRAM ++;
                }
                queue_a_head ++;
                queue_a_ptr ++;
                queue_len[wr_buf] --;
                queue_len[merge_buf] ++;
                //printf("case3:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case3:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);
              }

              //queue_a pointer move forward, check whether it exceeds the QUEUE_SIZE limit
              //if yes, then read from DRAM to queue_a
              if(queue_len[wr_buf]>0 && queue_a_head-QUEUE_SIZE*wr_buf>=QUEUE_SIZE){
                int queue_a_load_pass = queue_len[wr_buf] > QUEUE_SIZE ? QUEUE_SIZE/16 : (queue_len[wr_buf] +15)/16;
                
                int queue_val_sp_offset = QUEUE_SIZE*wr_buf;
                int queue_idx_sp_offset = QUEUE_SIZE*3+QUEUE_SIZE*wr_buf;
                int B_offset = B_start + queue_a_ptr ;
                
                REMEM();
                for(int round=0; round<queue_a_load_pass; round++){
                  VPREFETCH_LR(queue_val_sp_offset+round*16, B_val+B_offset+round*16, 0, 16,TO_SELF);
                  VPREFETCH_LR(queue_idx_sp_offset+round*16, B_idx+B_offset+round*16, 0, 16,TO_SELF);
                }
                FRAME_START(32*queue_a_load_pass);
                //reset queue_a_head to the beginning of the queue
                queue_a_head = QUEUE_SIZE*wr_buf;
                //printf("Line 439:load write queue using data from DRAM queue; From core %d\n",ptid);
              }

            }
            else{
              int remain_in_buf=QUEUE_SIZE*rd_buf+QUEUE_SIZE-queue_b_head;
              int rd_bund=(remain_in_buf >= queue_len[rd_buf])?queue_len[rd_buf] : remain_in_buf;
              for(int idx=0; idx<rd_bund; idx++){
               
                if(queue_len[merge_buf]<QUEUE_SIZE){
                  idx_queue[queue_c_head] = idx_queue[queue_b_head];
                  val_queue[queue_c_head] = val_queue[queue_b_head];
                  queue_c_head ++;
                }
                else{
                  FSTORE_NOACK(val_queue[queue_b_head], val_queue_DRAM+queue_c_head_DRAM, 0);
                  STORE_NOACK(idx_queue[queue_b_head], idx_queue_DRAM+queue_c_head_DRAM, 0);
                  //printf("only queue b left: write idx: %d to DRAM idx queue[%d]\n",idx_queue[queue_b_head],queue_c_head_DRAM);
                  queue_c_head_DRAM ++;
                }
                queue_b_head ++;
                queue_b_ptr ++;
                queue_len[rd_buf] --;
                queue_len[merge_buf] ++;
                //printf("case4:val queue [%d] is: %f\n",queue_c_head,val_queue[queue_c_head]);
                //printf("case4:idx queue [%d] is: %d\n",queue_c_head,idx_queue[queue_c_head]);
              }

              if(queue_len[rd_buf]>0 && queue_b_head-QUEUE_SIZE*rd_buf>=QUEUE_SIZE){
                int queue_b_load_pass = queue_len[rd_buf] > QUEUE_SIZE ? QUEUE_SIZE/16 : (queue_len[rd_buf] +15)/16;
                
                int queue_val_sp_offset = QUEUE_SIZE*rd_buf;
                int queue_idx_sp_offset = QUEUE_SIZE*3+QUEUE_SIZE*rd_buf;
                int queue_DRAM_offset = QUEUE_SIZE_DRAM*rd_buf + queue_b_ptr - QUEUE_SIZE;
              
                REMEM();
                for(int round=0; round<queue_b_load_pass; round++){
                  VPREFETCH_LR(queue_val_sp_offset+round*16, val_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
                  VPREFETCH_LR(queue_idx_sp_offset+round*16, idx_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
                }
                FRAME_START(32*queue_b_load_pass);
                //reset queue_a_head to the beginning of the queue
                queue_b_head = QUEUE_SIZE*rd_buf;
                //printf("Line 481:load rd queue using data from DRAM queue; From core %d\n",ptid);
              }



            }
          }
          wr_buf = (wr_buf+1)%3;
          rd_buf = (rd_buf+1)%3;
          merge_buf = (merge_buf+1)%3;
        }  
      } 
      
      //printf("finish read and merge of row[%d] in B; From core %d\n",a_idx,ptid);
      
      
      
    }

    //merge of row B complete, write back to memory 
    //for(int q_idx=0; q_idx<300; q_idx++){
    //  printf("val queue [%d] is: %f   ;",q_idx,val_queue[q_idx]);
    //}
    //printf("queue_len[%d] is %d\n",rd_buf,queue_len[rd_buf]);

    //Dynamic allocate space for output
    //if(ptid==0) printf("C_valout_tmp ADDRESS is: %d, C_idxout_tmp ADDRESS is: %d, last VAL ADDRESS is: %d, last IDX ADDRESS is: %d\n",C_valout_tmp,C_idxout_tmp,C_valout_tmp+mat_m*QUEUE_SIZE,C_idxout_tmp+mat_m*QUEUE_SIZE);
    int idx_bound = queue_len[rd_buf] <= QUEUE_SIZE ? queue_len[rd_buf] : QUEUE_SIZE;
    for(int c_idx=0; c_idx<idx_bound; c_idx++){
      //C_valout_tmp[i*QUEUE_SIZE + c_idx] = val_queue[QUEUE_SIZE*rd_buf+c_idx];
      //C_idxout_tmp[i*QUEUE_SIZE + c_idx] = idx_queue[QUEUE_SIZE*rd_buf+c_idx];
      FSTORE_NOACK(val_queue[QUEUE_SIZE*rd_buf+c_idx], C_valout_tmp + i*QUEUE_SIZE + c_idx, 0);
      STORE_NOACK(idx_queue[QUEUE_SIZE*rd_buf+c_idx], C_idxout_tmp + i*QUEUE_SIZE + c_idx, 0);
      //printf("Finish one row ----------------SPM queue idx is:%d, row[%d] index is: %d, value is: %f\n",c_idx, i, idx_queue[QUEUE_SIZE*rd_buf+c_idx],val_queue[QUEUE_SIZE*rd_buf+c_idx]);
      if(i==0){
        printf("ROW0 Check--------------store value %f to C_valout_tmp[%d]\n",val_queue[QUEUE_SIZE*rd_buf+c_idx],c_idx);
        //printf("C_valout_tmp result check: valout[%d]:%f, address in memory: %d\n",i*QUEUE_SIZE + c_idx,C_valout_tmp[i*QUEUE_SIZE + c_idx], &C_valout_tmp[i*QUEUE_SIZE + c_idx]);
      }
    }
    //FENCE();
    
    if(queue_len[rd_buf]>QUEUE_SIZE){
    /*  int val_extend_offset =  atomic_fetch_add_explicit(&val_q, queue_len[rd_buf] - QUEUE_SIZE, memory_order_relaxed);
      int idx_extend_offset =  atomic_fetch_add_explicit(&idx_q, queue_len[rd_buf] - QUEUE_SIZE, memory_order_relaxed);
      float *val_out = C_val_extend+val_extend_offset;
      int *idx_out = C_idx_extend+idx_extend_offset;
    */
      float * val_out_ptr; 
      float* val_out = (float*)malloc_cache_aligned(sizeof(float),queue_len[rd_buf]-QUEUE_SIZE,(void**)&val_out_ptr);
      int * idx_out_ptr; 
      int* idx_out = (int*)malloc_cache_aligned(sizeof(int),queue_len[rd_buf]-QUEUE_SIZE,(void**)&idx_out_ptr);
      //printf("allocate %d words for output val and idx of row %d, From core %d\n",queue_len[rd_buf]-QUEUE_SIZE,i,ptid);
      //printf("val out ADDRESS is: %d, idx_out ADDRESS is: %d, length is: %d; last val out is: %d, last idx out is: %d\n", val_out,idx_out,queue_len[rd_buf]-QUEUE_SIZE,val_out+queue_len[rd_buf]-QUEUE_SIZE,idx_out+queue_len[rd_buf]-QUEUE_SIZE);
      int remain_len = queue_len[rd_buf] - QUEUE_SIZE;
      int wr_pass = (remain_len + QUEUE_SIZE*3 -1)/(QUEUE_SIZE*3);
      int sp_val_offset=0;
      int sp_idx_offset=3*QUEUE_SIZE;
      int queue_DRAM_offset = QUEUE_SIZE_DRAM*rd_buf;
      int out_offset=0;
      for(int pass_idx=0; pass_idx<wr_pass; pass_idx++){
        REMEM();
        for(int round=0; round<QUEUE_SIZE*3/16; round++){
          VPREFETCH_LR(sp_val_offset+round*16, val_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
          VPREFETCH_LR(sp_idx_offset+round*16, idx_queue_DRAM+queue_DRAM_offset+round*16, 0, 16,TO_SELF);
        }
        FRAME_START(2*QUEUE_SIZE*3);
        queue_DRAM_offset += QUEUE_SIZE*3;
        int wr_bund = remain_len > QUEUE_SIZE*3 ? QUEUE_SIZE*3 : remain_len;
        for(int c_idx=0; c_idx<wr_bund; c_idx++){
          FSTORE_NOACK(val_queue[c_idx], val_out + out_offset, 0);
          STORE_NOACK(idx_queue[c_idx], idx_out + out_offset, 0);
          out_offset ++;
        }

      }
      //FENCE();
      C_inter_val[i] = val_out;
      C_inter_idx[i] = idx_out;

    }
    
    
    //C_inter_len[i] = queue_len[rd_buf];
    //row length begin from C_ptr[1], C_ptr[0] is 0
    STORE_NOACK(queue_len[rd_buf], C_ptr+i+1, 0);
    //STORE_NOACK(val_out, C_inter_val+i, 0);
    //STORE_NOACK(idx_out, C_inter_idx+i, 0);
    //printf("val_out is: %d, idx_out is: %d\n",val_out,idx_out);
    //printf("C_inter_val is: %d, C_inter_idx is: %d\n",C_inter_val,C_inter_idx);
    //printf("C_inter_len[%d] is: %d\n",i,queue_len[rd_buf]);
    //printf("in line 496\n");
    //printf("write back length{%d} of row[%d] in C; From core %d\n",queue_len[rd_buf],i,ptid);

  }


  //printf("come to barrier, core[%d]\n",ptid);
  /*if(ptid==0){
    printf("C value for row 0 before barrier is: ");
    for(int ele=0; ele<QUEUE_SIZE; ele++){
      printf("[%d]:%f ",ele,C_valout_tmp[ele]);
    }
    float *row0v_ptr = C_inter_val[0];
    for(int ele=0; ele<79; ele++){
      printf("[%d]:%f ",ele+QUEUE_SIZE,row0v_ptr[ele]);
    }
    printf("\n");
    printf("C index for row 0 before barrier is: ");
    for(int ele=0; ele<QUEUE_SIZE; ele++){
      printf("[%d]:%d ",ele,C_idxout_tmp[ele]);
    }
    int *row0i_ptr = C_inter_idx[0];
    for(int ele=0; ele<79; ele++){
      printf("[%d]:%f ",ele+QUEUE_SIZE,row0i_ptr[ele]);
    }
    printf("\n");
  }*/
  pthread_barrier_wait(&body_barrier);
  /*printf("finish merging, from core %d\n",ptid);
  if(ptid==0){
    printf("C value for row 0 after barrier is: ");
    for(int ele=0; ele<QUEUE_SIZE; ele++){
      printf("[%d]:%f, ADDRESS in memory:%d \n",ele,C_valout_tmp[ele],&C_valout_tmp[ele]);
    }
    //float *row0v_ptr = C_inter_val[0];
    //for(int ele=0; ele<79; ele++){
    //  printf("[%d]:%f ",ele+QUEUE_SIZE,row0v_ptr[ele]);
    //}
    printf("\n");
    printf("C index for row 0 after barrier is: ");
    for(int ele=0; ele<QUEUE_SIZE; ele++){
      printf("[%d]:%d ",ele,C_idxout_tmp[ele]);
    }
    //int *row0i_ptr = C_inter_idx[0];
    //for(int ele=0; ele<79; ele++){
    //  printf("[%d]:%f ",ele+QUEUE_SIZE,row0i_ptr[ele]);
    //}
    printf("\n");
  }*/
  
  //accumulate length of different segments in different cores

  int num_accpass = (m + QUEUE_SIZE*6*num_cores - 1)/(QUEUE_SIZE*6*num_cores);
  int ele_last_pass = m%(QUEUE_SIZE*6*num_cores);
  int num_ele_last = ptid<(ele_last_pass%num_cores) ? (ele_last_pass + num_cores - 1)/num_cores : ele_last_pass / num_cores;
  //if((num_ele_last==0)&&(num_accpass==1)&&(ele_last_pass>0)) num_accpass=0;
  //recast float pointer to int pointer to combine the val_queue and idx_queue
  int *comb_queue = (int *)val_queue;
  for(int i=0; i<num_accpass; i++){
    //in each pass, read segments of length into local scratchpad and accumulate locally
    
    int num_ele = ((i==num_accpass-1)&&(ele_last_pass>0)) ? num_ele_last : QUEUE_SIZE*6;
    int load_pass = (num_ele+15)/16; //Assume QUEUE_SIZE is multiple of 16 here so all load can be accomedated!
    int len_offset = 0;
    int adjust = ((i==num_accpass-1)&&(ele_last_pass>0)) ? (ptid<(ele_last_pass%num_cores) ? 0:(ele_last_pass%num_cores)):0;
    REMEM();
    for(int round=0; round<load_pass; round++){
      VPREFETCH_LR(len_offset+round*16, C_ptr+1+i*QUEUE_SIZE*6*num_cores+ptid*num_ele+adjust+round*16, 0, 16,TO_SELF);
    }
    //for(int load_ele=0; load_ele<num_ele; load_ele++){
      //comb_queue[load_ele] = C_ptr[1+i*QUEUE_SIZE*6*num_cores+ptid*num_ele+adjust+load_ele];
    //}

    FRAME_START(16*load_pass);
    //accumulate locally, the initial accumulation value should be read from last accumulation pass
    int acc_len=ini_acc_val; //0 in first load, updated by last core in the array
    for(int ele=0; ele<num_ele; ele++){
      acc_len += comb_queue[ele];
      comb_queue[ele] = acc_len;
    }
    //update the accumulation results to cores after this core in the array using atomic write
    for(int core_idx=ptid+1; core_idx<num_cores; core_idx++){
      int result=atomic_fetch_add_explicit(&compen[core_idx], acc_len, memory_order_relaxed);
    }
    //need a barrier to wait for all necessary updates on compen variable 
    pthread_barrier_wait(&start_barrier);
    //add the compensation value to the local lenth accumulation results and write back to C_ptr
    int comp_local = compen[ptid];
    for(int ele=0; ele<num_ele; ele++){
      comb_queue[ele] += comp_local;
    }
    
    for(int ele=0; ele<num_ele; ele++){
      STORE_NOACK(comb_queue[ele], C_ptr+1+i*QUEUE_SIZE*6*num_cores+ptid*num_ele+adjust+ele, 0);
    }

    if(ptid==num_cores-1){
      ini_acc_val = comb_queue[num_ele-1];
      //if(i==num_accpass-1) workq = 0;
      //if(i==num_accpass-1){
      //  //knows the length of the output, allocate a space to create CSR format
      //  float * C_val_out_ptr; 
      //  float* C_val_out = (float*)malloc_cache_aligned(sizeof(float),comb_queue[num_ele-1],(void**)&C_val_out_ptr);
      //  int * C_idx_out_ptr; 
      //  int* C_idx_out = (int*)malloc_cache_aligned(sizeof(int),comb_queue[num_ele-1],(void**)&C_idx_out_ptr);
      //  C_val[0] = C_val_out;
      //  C_idx[0] = C_idx_out;
      //  workq = 0;
      //}
    }
    
    pthread_barrier_wait(&start_barrier);

    //printf("finish length accumulation\n");
  }

  //FENCE();


  //now all the pointers are stored in C_ptr, write back value and idx in parallel
  /*if(ptid==0){
    printf("C_ptr is: ");
    for(int ele=0; ele<m+1; ele++){
      printf("%d ",C_ptr[ele]);
    }
    printf("\n");
  }*/

  float *C_val_local = C_val;
  int *C_idx_local = C_idx;
  for(int i=atomic_fetch_add_explicit(&write_q, 1, memory_order_relaxed); i<m; i =atomic_fetch_add_explicit(&write_q, 1, memory_order_relaxed)){
    int row_start = C_ptr[i];
    int row_len = C_ptr[i+1]-row_start;
    int val_offset = 0;
    int idx_offset = QUEUE_SIZE*3;
    float *val_ptr = C_inter_val[i];
    int *idx_ptr = C_inter_idx[i];
    int row_ptr=QUEUE_SIZE;
    int offset=0;

    int load_pass = row_len<QUEUE_SIZE ? (row_len + 15)/16 : QUEUE_SIZE/16;
    REMEM();
    for(int round=0; round<load_pass; round++){
      VPREFETCH_LR(val_offset+round*16, C_valout_tmp+i*QUEUE_SIZE+round*16, 0, 16,TO_SELF);
      VPREFETCH_LR(idx_offset+round*16, C_idxout_tmp+i*QUEUE_SIZE+round*16, 0, 16,TO_SELF);    
    }
    FRAME_START(32*load_pass);

    int rd_bund = row_len<QUEUE_SIZE ? row_len : QUEUE_SIZE;
    for(int j=0;j<rd_bund;j++){
      FSTORE_NOACK(val_queue[j], C_val_local + row_start + j, 0);
      STORE_NOACK(idx_queue[j], C_idx_local + row_start + j, 0);
      //if(i==0) printf("ROW0-FINALwrite----------------write idx[%d] with value %f, while value in c_val_tmp is %f \n",j,val_queue[j],C_valout_tmp[j]);
    }
    row_len -= QUEUE_SIZE;
    //printf("C_val pointer is %d\n", C_val);
    while(row_len>0){
      int load_pass = row_len<QUEUE_SIZE ? (row_len + 15)/16 : QUEUE_SIZE/16;
      REMEM();
      for(int round=0; round<load_pass; round++){
        VPREFETCH_LR(val_offset+round*16, val_ptr+offset+round*16, 0, 16,TO_SELF);
        VPREFETCH_LR(idx_offset+round*16, idx_ptr+offset+round*16, 0, 16,TO_SELF);
      }
      FRAME_START(32*load_pass);
      offset += 16*load_pass;
      
    
      int rd_bund = row_len<QUEUE_SIZE ? row_len : QUEUE_SIZE;
      for(int j=0;j<rd_bund;j++){
        FSTORE_NOACK(val_queue[j], C_val_local + row_start + row_ptr, 0);
        STORE_NOACK(idx_queue[j], C_idx_local + row_start + row_ptr, 0);
        //if(i==0) printf("ROW0-FINALwrite----------------write idx[%d] with idx %d, while index in DRAM is: %d \n",row_ptr,idx_queue[j],C_inter_idx[0][row_ptr-QUEUE_SIZE]);
        row_ptr ++;
      }
      
      row_len -=rd_bund;
    }
  }


/*  pthread_barrier_wait(&start_barrier);
  if (ptid == 0)
  {
    stats_off();
    int error_tmp=0;
    for(int i=0;i<ini_acc_val;i++){
      if(C_val_local[i]!=val_ref[i]){
        printf("VAL missmatch found in valC[%d]! Expect: %f, returned: %f\n",i,val_ref[i],C_val_local[i]);
        error_tmp +=1;
      }
    }
    for(int i=0;i<ini_acc_val;i++){
      if(C_idx_local[i]!=idx_ref[i]){
        printf("IDX missmatch found in idxC[%d]! Expect: %d, returned: %d\n",i,idx_ref[i],C_idx_local[i]);
        error_tmp +=1;
      }
    }
    for(int i=1;i<m+1;i++){
      if(C_ptr[i]!=ptr_ref[i]){
        printf("Pointer missmatch found in ptrC[%d]! Expect: %d, returned: %d\n",i,ptr_ref[i],C_ptr[i]);
        error_tmp +=1;
      }
    }
    printf("total errors are %d\n", error_tmp);
    //error = error_tmp;

  }
*/


  //free(val_queue_DRAM_ptr);
  //free(idx_queue_DRAM_ptr);
  

}