#include <stdio.h>
#include <ctime>
#include <malloc.h>
#include <map>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <ctime>
#include "cuckoo.cuh"
#include "cudaHeaders.h"

using namespace std;

/* Randomize Generation */
void rand_gen(int *vals, const int n) {
    map<int, bool> val_map;
    int i = 0;
    while (i < n) {
        int value = (rand() % (LIMIT - 1)) + 1;
        if (val_map.find(value) != val_map.end()) {
            continue;
        }
        val_map[value] = true;
        vals[i] = value;
        i++;
    }
}

/* Single gpu implementation */
template <typename T>
static inline __device__ int
hash(const T val, const CuckooHashing<T>::CuckooConf config, const int func_idx,
        const int size) {
    CuckooHashing<T>::CuckooConf funcConfig = config[index];
    return ((val ^ funcConfig.rv) >> funcConfig.ss) % size;
}

