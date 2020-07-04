#include "cuckoo.cuh"
#include <stdint.h>
#include <stdio.h>
#include <vector>

__device__ uint32_t
get_next_loc(uint32_t k,uint32_t v,uint32_t num_table,cuckoo* table){
    num_table%=TABLE_NUM;
    //printf("hash: k:%d size:%d ,hash :%d \n",(( k%_PRIME )+num_table),(table->Lsize[num_table]/BUCKET_SIZE),(( k%_PRIME) +num_table) % (table->Lsize[num_table]/BUCKET_SIZE));
    return (( k%_PRIME )+num_table) % (table->Lsize[num_table]/BUCKET_SIZE);
}


__device__ void pbucket(bucket *b,int num,int hash){
    printf("table%d,%d \n",num,hash);
    for(int i=0;i<BUCKET_SIZE;i++){
        printf("%d,%d ",b->key[i],b->value[i]);
    }
    printf("\n");
}


__global__ void
cuckoo_insert(uint32_t* key, /// key to insert
              uint32_t* value, /// value to insert
              uint32_t size, /// insert size
              uint32_t* resize, /// insert error?
              cuckoo* table, /// hash table
              uint32_t table_size) {
    *resize = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /// for every k
    if(tid>= size) return;


    int lan_id = tid & 0x0000001f;
    int wrap_id = tid >> 5;

    while (tid < size) {

        int myk = key[tid];
        int myv = value[tid];
        int is_active = 1;/// mark for work

        int work_k = 0;
        int work_v = 0;

        /// for insert
        int hash;
        int hash_table_num;
        int ballot;

        volatile __shared__ int wrap[(NUM_BLOCK * NUM_THREADS)>>5 ];

        /// while have work to do
        while (__any_sync(0xFFFFFFFF,is_active != 0)) {

            hash_table_num=0;

            //printf("lan_id: %d, active:%d \n",lan_id,is_active);

/// step1   start voting ==================================
            if (is_active != 0)
                wrap[wrap_id] = lan_id;

            work_k = myk;
            work_v = myv;
            /// over ======



/// step2   broadcast ====================================
            work_k=__shfl(work_k, wrap[wrap_id]);
            work_v=__shfl(work_v, wrap[wrap_id]);

           
/// step3   insert to the table ===========================
            hash_table_num = 0;
            hash = get_next_loc(work_k, work_v, 0,table);

            /// find null or too long
            for (int i = 0; i < MAX_ITERATOR; i++) {

/// step3.1     
                //assert(0);
                /// block
                bucket *b = &(table->table[hash_table_num][hash]);

/// step3.2     check exist & insert
                ballot = __ballot_sync(0xFFFFFFFF,b->key[lan_id] == work_k);   
                if (ballot != 0) { /// update
                    if (lan_id == wrap[wrap_id])
                        is_active = 0;
                    break;
                }


                
/// step3.3      check null & insert
                ballot = __ballot_sync(0xFFFFFFFF,b->key[lan_id] == 0);
                if (ballot != 0) {
                    /// set kv
                    if (lan_id == __ffs(ballot)-1) {
                        b->key[lan_id] = work_k;
                        b->value[lan_id] = work_v;
                    }


                    /// mark active false

                    if (lan_id == wrap[wrap_id])
                        is_active = 0;

                    /// insert ok ,
                    break;
                }/// insert



/// step3.4     other,we need  cuckoo evict

                if(lan_id==wrap[wrap_id]){
                    work_k=b->key[lan_id];
                    work_v=b->value[lan_id];
                    b->key[lan_id]=myk;
                    b->value[lan_id]=myv;
                }
                __shfl(work_k, wrap[wrap_id]);
                __shfl(work_v, wrap[wrap_id]);


/// step3.5     keep evicted kv and reinsert
                hash = get_next_loc(work_k, work_v, hash_table_num++,table);

            }

        }


        tid += NUM_BLOCK * NUM_THREADS;
    }
}

__global__ void
rehash(uint32_t* rkey,uint32_t* rvalue,uint32_t old_size,uint32_t table_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //insert table
    while(tid<old_size){
        tid+=NUM_BLOCK*NUM_THREADS;
    }
}

void gpu_rehash(uint32_t old_size,uint32_t new_table_size){
    //malloc
    printf("----rehash size:  %d --> %d\n",old_size,new_table_size);
    uint32_t* d_key,*d_value;
    cudaMalloc((void**)&d_key, sizeof(uint32_t)*new_table_size);
    cudaMalloc((void**)&d_value, sizeof(uint32_t)*new_table_size);
    cudaMemset(d_key,0, sizeof(uint32_t)*new_table_size);
    cudaMemset(d_value,0, sizeof(uint32_t)*new_table_size);
#ifdef TIME
    GpuTimer timer;
    timer.Start();
#endif
    
    rehash<<<NUM_BLOCK,NUM_THREADS>>>(d_key,d_value,old_size,new_table_size);
    
#ifdef TIME
    timer.Stop();
    double diff = timer.Elapsed() * 1000000;
    printf("api<<<rehash>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double)diff, (double)(BATCH_SIZE) / diff);
#endif

#ifdef DEBUG
    show_table<<<1,1>>>(table);
#endif

}


__global__ void show_table(cuckoo* table){
    if(blockIdx.x * blockDim.x + threadIdx.x>0 ) return;

    for(int i=0;i<TABLE_NUM;i++){
        printf("\n\n\ntable:%d\n",i);
        for(int j=0;j<(table->Lsize[i])/BUCKET_SIZE;j++){
            for(int t=0;t<BUCKET_SIZE;t++)
                printf(" %d,%d ",table->table[i][j].key[t],table->table[i][j].value[t]);
            printf("\n");
        }

    }
}


void gpu_lp_insert(uint32_t* key,uint32_t* value,uint32_t size,uint32_t* resize,cuckoo *table,uint32_t table_size){
    //in main
    // st is you operator num
    unsigned int real_block=((unsigned int)size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;
#ifdef TIME
    GpuTimer timer;
    timer.Start();
#endif
    //printf("start gpulpi\n");
    
    cuckoo_insert<<<block,NUM_THREADS>>>(key,value,size,resize,table,table_size);
    int* a=new int[1];
    
    cudaMemcpy(a,resize,sizeof(uint32_t),cudaMemcpyDeviceToHost);
    
    
#ifdef TIME
    timer.Stop();
    double diff = timer.Elapsed() * 1000000;
    printf("api<<<insert>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double)diff, (double)(BATCH_SIZE) / diff);
#endif

#ifdef DEBUG
    show_table<<<1,1>>>(table);
#endif
  
}

void gpu_lp_search(uint32_t* key,uint32_t* ans,uint32_t size,uint32_t table_size){
    unsigned int real_block=(size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;

}


CuckooHashing::CuckooHashing(int size) {
    //rehash will be init by every insert function
    cudaMalloc((void**)&rehash,sizeof(uint32_t));
    //cudaMemset(remove(),0,size*sizeof(uint32_t));
    
    if(size<1000) size = 20 * TABLE_NUM * BUCKET_SIZE;
    /// malloc table
    int s_bucket  =  size  / TABLE_NUM / BUCKET_SIZE;
    int s_size = s_bucket *BUCKET_SIZE ;
    table_size= s_size * TABLE_NUM;

    cuckoo * h_table=(cuckoo*)malloc(sizeof(cuckoo));
    for(int i=0;i<TABLE_NUM;i++){
        cudaMalloc((void **) &h_table->table[i], sizeof(uint32_t) * s_size * 2);
        cudaMemset(h_table->table[i], 0, sizeof(uint32_t) * s_size * 2);
        h_table->Lsize[i]=s_size;
    }

    cudaMalloc((void**)&hash_table,sizeof(cuckoo));
    cudaMemcpy(hash_table,h_table,sizeof(cuckoo),cudaMemcpyHostToDevice);

    
    // printf("init ok\n");
}

void CuckooHashing::hash_insert(uint32_t *key, uint32_t *value,int size) {
    num_size+=size;
    uint32_t* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(uint32_t)*size);
    cudaMemcpy(d_keys, key, sizeof(uint32_t)*size, cudaMemcpyHostToDevice);

    uint32_t* d_value;
    cudaMalloc((void**)&d_value, sizeof(uint32_t)*size);
    cudaMemcpy(d_value, value, sizeof(uint32_t)*size, cudaMemcpyHostToDevice);

    // does size need be copy first
    gpu_lp_insert(d_keys,d_value,size,rehash,hash_table,table_size);
    //printf("self check success\n");
    // printf("insert ok\n");
    cudaFree(d_value);
    cudaFree(d_keys);
}

void CuckooHashing::hash_search(uint32_t *key, uint32_t *value,int size){
    
    uint32_t* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(uint32_t)*size);
    cudaMemcpy(d_keys, key, sizeof(uint32_t)*size, cudaMemcpyHostToDevice);

    uint32_t* d_value;
    cudaMalloc((void**)&d_value, sizeof(uint32_t)*size);
    cudaMemcpy(d_value, value, sizeof(uint32_t)*size, cudaMemcpyHostToDevice);

    //kernel
    gpu_lp_search(d_keys,d_value,size,table_size);
    
    
    cudaMemcpy(value, d_value, sizeof(uint32_t)*size, cudaMemcpyDeviceToHost);
    cudaFree(d_value);
    cudaFree(d_keys);
}



void CuckooHashing::hash_delete(int *key,int *ans,int size) {
   for(int i=0;i<size;i++){
       int tmp=ans[i];
       tmp/=1;
   }
}

CuckooHashing::~CuckooHashing() {
    cudaFree(rehash);
    cudaFree(hash_table);
}
