#ifndef CUDA_TEST_CLION_VECADD_H
#define CUDA_TEST_CLION_VECADD_H

#include "cudaHeaders.h"


/// isolated kv
struct bucket{
    uint32_t key[BUCKET_SIZE];
    uint32_t value[BUCKET_SIZE];
};

/// definition structure
typedef struct mycuckoo{
    /// table , store kv ,
    /// in detail , kv is  isolated
    bucket *table[TABLE_NUM];
    /// size of sub table , determine rehash : L min resize
    uint32_t Lsize[TABLE_NUM];
    /// Lock for bucket lock,use atomicCAS
    uint32_t *Lock;
    uint2 hash_fun[TABLE_NUM];
}cuckoo;

class CuckooHashing{
    //device pointer
    cuckoo* hash_table;
    uint32_t *rehash;
    int table_size;
    int num_size;
public:
    explicit CuckooHashing(int size);

    ~CuckooHashing();

    void hash_insert(uint32_t *k, uint32_t *v,int size);

    void hash_search(uint32_t *k, uint32_t *v,int size);

    void hash_delete(int *k,int *ans,int size);

};

__global__ void
cuckoo_insert(uint32_t* key, /// key to insert
              uint32_t* value, /// value to insert
              uint32_t size, /// insert size
              int* resize); /// insert error?

__global__ void
cuckoo_search(uint32_t* key, /// key to s
              uint32_t* value, /// value to s
              uint32_t siz); /// s size

/// dubug
void GPU_show_table();

// all the pointer are device pointer
void gpu_lp_insert(uint32_t* key,
                   uint32_t* value,
                   uint32_t size,
                   int* resize);

void gpu_lp_search(uint32_t* key,
                   uint32_t* ans,
                   uint32_t size);

void gpu_lp_delete(uint32_t* key,
                   uint32_t* ans,
                   uint32_t size);

void gpu_lp_set_table(cuckoo* h_table);




#endif //CUDA_TEST_CLION_VECADD_H
