#include <stdio.h>
#include <ctime>
#include <malloc.h>
#include <map>
#include <cstdlib>
#include <cmath>
#include "cuckoo.cuh"
#include "cudaHeaders.h"
#include "mt19937ar.h"

using namespace std;
#define CUCKOO_GPU
/* Redefine CuckooConf outside struct */
typedef struct {
    int rv;     // Randomized XOR value.
    int ss;     // Randomized shift filter start position.
} CuckooConf;

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
#if (defined (CUCKOO_GPU))
template <typename T>
static inline __device__ int CuckooHashing<T>::hash(const T val, const CuckooHashing<T>::CuckooConf * const config, const int index,
        const int size) {
    CuckooConf funcConfig = config[index];
    return ((val ^ funcConfig.rv) >> funcConfig.ss) % size;
}

template <typename T>
static inline __device__ T
fetch(const T data, const int width) {
    return data >> width;
}

template <typename T>
__global__ void
cuckooInsertKernel(const T * const vals, const int n,
                   T * const data, const int size,
                   const CuckooConf * const hash_func_configs, const int num_funcs,
                   const int evict_bound, const int pos_width,
                   int * const rehash_requests) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {

        // Set initial conditions.
        T cur_val = vals[idx];
        int cur_func = 0;
        int evict_count = 0;

        // Start the test-kick-and-reinsert loops.
        do {
            int pos = do_hash(cur_val, hash_func_configs, cur_func, size);
            T old_data = atomicExch(&data[cur_func * size + pos], make_data(cur_val, cur_func, pos_width));
            if (old_data != EMPTY_CELL) {
                cur_val = fetch_val(old_data, pos_width);
                cur_func = (fetch_func(old_data, pos_width) + 1) % num_funcs;
                evict_count++;
            } else
                return;
        } while (evict_count < num_funcs * evict_bound);

        // Exceeds eviction bound, needs rehashing.
        atomicAdd(rehash_requests, 1);
    }
}

template <typename T>
int
CuckooHashing<T>::insert(const T * const vals, const int n, const int depth) {

    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    FuncConfig *d_hash_func_configs;
    int rehash_requests = 0;
    int *d_rehash_requests;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));
    cudaMalloc((void **) &d_rehash_requests, sizeof(int));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, _data, (_num_funcs * _size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_rehash_requests, &rehash_requests, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the insert kernel.
    cuckooInsertKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, n,
                                                                      d_data, _size,
                                                                      d_hash_func_configs, _num_funcs,
                                                                      _evict_bound, _pos_width,
                                                                      d_rehash_requests);

    // If need rehashing, do rehash with original data + VALS. Else, retrieve results into data.
    cudaMemcpy(&rehash_requests, d_rehash_requests, sizeof(int), cudaMemcpyDeviceToHost);
    if (rehash_requests > 0) {
        cudaFree(d_vals);
        cudaFree(d_data);
        cudaFree(d_hash_func_configs);
        cudaFree(d_rehash_requests);
        int levels_beneath = rehash(vals, n, depth + 1);
        if (levels_beneath == ERR_DEPTH)
            return ERR_DEPTH;
        else
            return levels_beneath + 1;
    } else {
        cudaMemcpy(_data, d_data, (_num_funcs * _size) * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(d_vals);
        cudaFree(d_data);
        cudaFree(d_hash_func_configs);
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
                   const FuncConfig * const hash_func_configs, const int num_funcs,
                   const int pos_width) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        for (int i = 0; i < num_funcs; ++i) {
            int pos = do_hash(val, hash_func_configs, i, size);
            if (fetch_val(data[i * size + pos], pos_width) == val) {
                data[i * size + pos] = EMPTY_CELL;
                return;
            }
        }
    }
}

template <typename T>
void
CuckooHashing<T>::del(const T * const vals, const int n) {

    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    CuckooConf *d_hash_func_configs;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, _data, (_num_funcs * _size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);

    // Launch the delete kernel.
    cuckooDeleteKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, n,
                                                                      d_data, _size,
                                                                      d_hash_func_configs, _num_funcs,
                                                                      _pos_width);

    // Retrieve results.
    cudaMemcpy(_data, d_data, (_num_funcs * _size) * sizeof(T), cudaMemcpyDeviceToHost);

    // Free GPU memories.
    cudaFree(d_vals);
    cudaFree(d_data);
    cudaFree(d_hash_func_configs);
}


/**
 *
 * Cuckoo: lookup operation (kernel + host function).
 *
 */
template <typename T>
__global__ void
cuckooLookupKernel(const T * const vals, bool * const results, const int n,
                   const T * const data, const int size,
                   const CuckooConf * const hash_func_configs, const int num_funcs,
                   const int pos_width) {

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        T val = vals[idx];
        for (int i = 0; i < num_funcs; ++i) {
            int pos = do_hash(val, hash_func_configs, i, size);
            if (fetch_val(data[i * size + pos], pos_width) == val) {
                results[idx] = true;
                return;
            }
        }
        results[idx] = false;
    }
}

template <typename T>
void
CuckooHashing<T>::lookup(const T * const vals, bool * const results, const int n) {

    // Allocate GPU memory space.
    T *d_vals;
    T *d_data;
    bool *d_results;
    FuncConfig *d_hash_func_configs;
    cudaMalloc((void **) &d_vals, n * sizeof(T));
    cudaMalloc((void **) &d_results, n * sizeof(bool));
    cudaMalloc((void **) &d_data, (_num_funcs * _size) * sizeof(T));
    cudaMalloc((void **) &d_hash_func_configs, _num_funcs * sizeof(FuncConfig));

    // Copy values onto GPU memory.
    cudaMemcpy(d_vals, vals, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, _data, (_num_funcs * _size) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_func_configs, _hash_func_configs, _num_funcs * sizeof(FuncConfig),
               cudaMemcpyHostToDevice);

    // Launch the lookup kernel.
    cuckooLookupKernel<<<ceil((double) n / BLOCK_SIZE), BLOCK_SIZE>>>(d_vals, d_results, n,
                                                                      d_data, _size,
                                                                      d_hash_func_configs, _num_funcs,
                                                                      _pos_width);

    // Retrieve results.
    cudaMemcpy(results, d_results, n * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free GPU memories.
    cudaFree(d_vals);
    cudaFree(d_results);
    cudaFree(d_data);
    cudaFree(d_hash_func_configs);
}


template <typename T>
int CuckooHashing<T>::rehash(const T * const vals, const int n, const int depth) {

    // If exceeds max rehashing depth, abort.
    if (depth > MAX_DEPTH)
        return ERR_DEPTH;

    // Generate new set of hash functions.
    gen_hash_funcs();

    // Clear data and map, put values into a buffer.
    std::vector<T> val_buffer;
    for (int i = 0; i < _num_funcs; ++i) {
        for (int j = 0; j < _size; ++j) {
            if (fetch_val(_data[i * _size + j]) != EMPTY_CELL)
                val_buffer.push_back(fetch_val(_data[i * _size + j]));
            _data[i * _size + j] = EMPTY_CELL;
        }
    }
    for (int i = 0; i < n; ++i)
        val_buffer.push_back(vals[i]);

    // Re-insert all values.
    int levels_beneath = insert_vals(val_buffer.data(), val_buffer.size(), depth);
    if (levels_beneath == ERR_DEPTH)
        return ERR_DEPTH;
    else
        return levels_beneath;
}


/** Cuckoo: print content out. */
template <typename T>
void
CuckooHashing<T>::show() {
    std::cout << "Funcs: ";
    for (int i = 0; i < _num_funcs; ++i) {
        FuncConfig fc = _hash_func_configs[i];
        std::cout << "(" << fc.rv << ", " << fc.ss << ") ";
    }
    std::cout << std::endl;
    for (int i = 0; i < _num_funcs; ++i) {
        std::cout << "Table " << i << ": ";
        for (int j = 0; j < _size; ++j)
            std::cout << std::setw(10) << fetch_val(_data[i * _size + j]) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
#endif