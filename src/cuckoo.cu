#include <stdio.h>
#include <ctime>
#include <malloc.h>
#include <map>
#include <cstdlib>
#include <cmath>
#include "cuckoo.cuh"
#include "cudaHeaders.h"
#include "mt19937ar.h"
#define CUCKOO_GPU


using namespace std;
/* Redefine CuckooConf outside struct */
typedef struct {
    int rv;     // Randomized XOR value.
    int ss;     // Randomized shift filter start position.
} CuckooConfre;

/* Randomize Generation */
void rand_gen(int *vals, const int n) {
    map<int, bool> val_map;
    int i = 0;
    while (i < n) {
        int value=genrand_int31();
        if (val_map.find(value) != val_map.end()) {
            continue;
        }
        val_map[value] = true;
        vals[i] = value;
        i++;
    }
}

/* Single gpu implementation */
#ifdef CUCKOO_GPU
template <typename T>
static inline __device__ int hash1(const T val, const CuckooConfre *const config, const int index, const int size) {
    CuckooConfre func_config = config[index];
    return ((val ^ func_config.rv) >> func_config.ss) % size;
}

template <typename T> static inline __device__ T make(const T val, const int func, const int position) {
    return (val << position) ^ func;
}

template <typename T> static inline __device__ T fetch_val(const T data, const int position) {
    return data >> position;
}

template <typename T> static inline __device__ int fetch_func(const T data, const int position) {
    return data & ((0x1 << position) - 1);
}

template <typename T>
__global__ void cuckooInsertKernel(const T *const vals, const int n, T *const data, const int size,
                                   const CuckooConfre *const config, const int num, const int bound, const int width,
                                   int *const rehash_requests) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Only threads within range are active.
    if (idx < n) {
        // Set initial conditions.
        T cur_val = vals[idx];
        int cur_func = 0;
        int evict_count = 0;
        // Start the test-kick-and-reinsert loops.
        do {
            int pos = hash1(cur_val, config, cur_func, size);
            T old_data = atomicExch(&data[cur_func * size + pos], make(cur_val, cur_func, width));
            if (old_data != EMPTY_CELL) {
                cur_val = fetch_val(old_data, width);
                cur_func = (fetch_func(old_data, width) + 1) % num;
                evict_count++;
            } else
                return;
        } while (evict_count < num * bound);
        // Exceeds eviction bound, needs rehashing.
        atomicAdd(rehash_requests, 1);
    }
}

template <typename T> int CuckooHashing<T>::insert(const T * vals, const int n, const int depth) {

    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    CuckooConfre *d_config;
    int rehash_requests = 0;
    int *d_rehash_requests;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data, (num * size) * sizeof(T));
    cudaMalloc((void **) &d_config, num * sizeof(CuckooConfre));
    cudaMalloc((void **) &d_rehash_requests, sizeof(int));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data, (num * size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_config, config, num * sizeof(CuckooConfre),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_rehash_requests, &rehash_requests, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the insert kernel.
    cuckooInsertKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, n,
                                                                      d_data, size,
                                                                      d_config, num,
                                                                      bound, width,
                                                                      d_rehash_requests);

    // If need rehashing, do rehash with original data + VALS. Else, retrieve results into data.
    cudaMemcpy(&rehash_requests, d_rehash_requests, sizeof(int), cudaMemcpyDeviceToHost);
    if (rehash_requests > 0) {
        cudaFree(d_vals);
        cudaFree(d_data);
        cudaFree(d_config);
        cudaFree(d_rehash_requests);
        int beneath = rehash(vals, n, depth + 1);
        if (beneath == ERROR_DEPTH)
            return ERROR_DEPTH;
        else
            return beneath + 1;
    } else {
        cudaMemcpy(data, d_data, (num * size) * sizeof(T), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < sizeof(data) / sizeof(T); i++) {
        //     cout << data[i];
        // }
        cudaFree(d_vals);
        cudaFree(d_data);
        cudaFree(d_config);
        cudaFree(d_rehash_requests);
        return 0;
    }
}

/**
 *
 * Cuckoo: delete operation (kernel + host function).
 *
 */
template <typename T>
__global__ void
cuckooDeleteKernel(const T * const vals, const int n,
                   T * const data, const int size,
                   const CuckooConfre * const config, const int num,
                   const int width) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        for (int i = 0; i < num; ++i) {
            int pos = hash1(val, config, i, size);
            if (fetch_val(data[i * size + pos], width) == val) {
                data[i * size + pos] = EMPTY_CELL;
                return;
            }
        }
    }
}

template <typename T> void CuckooHashing<T>::del(const T *const vals, const int n) {

    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    CuckooConfre *d_config;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data, (num * size) * sizeof(T));
    cudaMalloc((void **) &d_config, num * sizeof(CuckooConfre));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data, (num * size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_config, config, num * sizeof(CuckooConfre),
               cudaMemcpyHostToDevice);

    // Launch the delete kernel.
    cuckooDeleteKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, n,
                                                                      d_data, size,
                                                                      d_config, num,
                                                                      width);

    // Retrieve results.
    cudaMemcpy(data, d_data, (num * size) * sizeof(T), cudaMemcpyDeviceToHost);

    // Free GPU memories.
    cudaFree(d_vals);
    cudaFree(d_data);
    cudaFree(d_config);
}

template <typename T>
__global__ void cuckooLookupKernel(const T *const vals, bool *const results, const int n, const T *const data,
                                   const int size, const CuckooConfre *const config, const int num, const int width) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        for (int i = 0; i < num; ++i) {
            int pos = hash1(val, config, i, size);
            if (fetch_val(data[i * size + pos], width) == val) {
                results[idx] = true;
                return;
            }
        }
        results[idx] = false;
    }
}

template <typename T> void CuckooHashing<T>::lookup(const T *const vals, bool *const results, const int n) {
    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    bool *d_results;
    CuckooConfre *d_config;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_results, n * sizeof(bool));
    cudaMalloc((void **) &d_data, (num * size) * sizeof(T));
    cudaMalloc((void **) &d_config, num * sizeof(CuckooConfre));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data, (num * size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_config, config, num * sizeof(CuckooConfre),
               cudaMemcpyHostToDevice);

    // Launch the lookup kernel.
    cuckooLookupKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, d_results, n,
                                                                      d_data, size,
                                                                      d_config, num,
                                                                      width);

    // Retrieve results.
    cudaMemcpy(results, d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free GPU memories.
    cudaFree(d_vals);
    cudaFree(d_results);
    cudaFree(d_data);
    cudaFree(d_config);
}

#endif