#ifndef CUDA_TEST_CLION_VECADD_H
#define CUDA_TEST_CLION_VECADD_H

#include "cudaHeaders.h"


#define HASHING_DEPTH (100)
#define ERROR_DEPTH (-1)

#define BLOCK_SIZE (512)
#define LIMIT (0x1 << 30)
#define EMPTY_CELL (0)
#define CUCKOO_GPU

#ifdef CUCKOO_MUL_CPU
#include <omp.h>
#endif

#include <map>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <iostream>

using namespace std;
#ifdef CUCKOO_GPU
/* Redefine CuckooConf outside struct */
typedef struct {
    int rv;     // Randomized XOR value.
    int ss;     // Randomized shift filter start position.
} CuckooConf;
#endif
/* class for cuckoo hash table. */
template<typename T>
class CuckooHashing {
private:
#if (!defined (CUCKOO_GPU)) && (!defined(CUCKOO_MUL_GPU))
    //basic config
    typedef struct {
        int rv;
        int ss;
    } CuckooConf;
#endif
    //parameter
    int size;
    int bound;
    int num;
    int width;

    //data
    T *data;
    int *position;
    CuckooConf *config;

#if (!defined (CUCKOO_GPU)) && (!defined(CUCKOO_MUL_GPU))

    template<typename ST>
    void swap(ST *const pos1, ST *const pos2) {
        ST temp = *pos1;
        *pos1 = *pos2;
        *pos2 = temp;
    };

    int hash(const T val, const int i) {
        CuckooConf func_config = config[i];
        return ((val ^ func_config.rv) >> func_config.ss) % size;
    };

    //operations
    int rehash(const T val, const int depth) {
        if (depth > HASHING_DEPTH) {
            return ERROR_DEPTH;
        }
        gen();
        vector<T> buffer;
        #ifdef CUCKOO_MUL_CPU
        #pragma omp parallel for
        #endif
        for (int i = 0; i < size; i++) {
            if (data[i] != -1) {
                buffer.emplace_back(data[i]);
            }
            data[i] = -1;
            position[i] = -1;
        }
        buffer.emplace_back(val);

        // Re-insert all values.
        int level = 0;
        for (auto temp : buffer) {
            int beneath = insert(temp, depth);
            if (beneath == ERROR_DEPTH)
                return ERROR_DEPTH;
            else if (beneath > level)
                level = beneath;
        }
        return level;
    };
#endif

    void gen() {
        int width = 8 * sizeof(T) - ceil(log2((double) num));
        int swidth = ceil(log2((double) size));
        for (int i = 0; i < num; i++) {
            if (width <= swidth)
                config[i] = {rand(), 0};
            else
                config[i] = {rand(), rand() % (width - swidth + 1)};
        }
    };


#ifdef CUCKOO_GPU
    /** Inline helper functions. */
    inline T fetch_val(const T data) {
        return data >> width;
    }
    inline int fetch_func(const T data) {
        return data & ((0x1 << width) - 1);
    }
    int rehash(const T * const val, const int n, const int depth) {
        if (depth > HASHING_DEPTH) {
            return ERROR_DEPTH;
        }
        gen();
        vector<T> buffer;
        for (int i = 0; i < num; ++i) {
            for (int j = 0; j < size; i++) {
                if (fetch_val(data[i * size + j]) != -1) {
                    buffer.emplace_back(fetch_val(data[i * size + j]));
                }
                data[i * size + j] = -1;
            }
        }
        for (int i = 0; i < n; ++i)
            buffer.emplace_back(val[i]);

        // Re-insert all values.
        int beneath = insert(buffer.data(), buffer.size(), depth);
        if (beneath == ERROR_DEPTH)
            return ERROR_DEPTH;
        else
            return beneath;
    };
#endif
public:
    //constructor
    CuckooHashing(const int size, const int bound, const int num)
        : size(size), bound(bound), num(num), width(ceil(log2((double)num))) {

#if (!defined(CUCKOO_MUL_GPU)) && (!defined(CUCKOO_GPU))
        data = new T[size]();
#endif
#ifdef CUCKOO_GPU
        data = new T[num * size]();
#endif
        position = new int[size]();
        config = new CuckooConf[num + 1];
        gen();
    };

    //destructor
    ~CuckooHashing() {
        delete[] data;
        delete[] position;
        delete[] config;
    };


#if (!defined(CUCKOO_MUL_GPU)) && (!defined(CUCKOO_GPU))
    //hashing insert operation, the return is the bottom of the rehashed index.
    int insert(const T val, const int depth);

    //hashing del operation, the return is whether is delete success.
    bool del(const T val);

    //hashing lookup operation, the return is whether can be looked up.
    bool lookup(const T val);

#endif
#ifdef CUCKOO_GPU
    // hashing insert operation, the return is the bottom of the rehashed index.
    int insert(const T * const val,const int n, const int depth);

    //hashing del operation, the return is whether is delete success.
    void del(const T * const vals, const int n);

    // hashing lookup operation, the return is whether can be looked up.
    void lookup(const T * const vals, bool * const results, const int n);
#endif

    //hashing show operation, generate new set of hash functions and rehash, the return is the bottom of the rehashed index.
    void show();

};

//Implementation of Rand_gen
void rand_gen(int *vals, const int n);

//Implementation for CPU
#if (!defined (CUCKOO_GPU)) && (!defined(CUCKOO_MUL_GPU))
template <typename T> void CuckooHashing<T>::show() {
    cout << "Funcs: ";
    for (int i = 0; i < num; ++i) {
        CuckooConf func_config = config[i];
        cout << "(" << func_config.rv << ", " << func_config.ss << ") ";
    }
    cout << endl << "Table: ";
    for (int i = 0; i < size; ++i)
        cout << " " << data[i] << " ";
    cout << endl << "Fcmap: ";
    for (int i = 0; i < size; ++i)
        cout << " " << position[i] << " ";
    cout << endl << endl;
};
#endif //(!defined (CUCKOO_GPU)) || (!defined(CUCKOO_MUL_GPU))

//Implementation for GPU
template <typename T> void CuckooHashing<T>::show() {
    cout << "Funcs: ";
    for (int i = 0; i < num; ++i) {
        CuckooConf func_config = config[i];
        cout << "(" << func_config.rv << ", " << func_config.ss << ") ";
    }
    cout << endl;
    for (int i = 0; i < num; ++i) {
       cout << "Table " << i << ": ";
        for (int j = 0; j < size; ++j)
            cout << " " << fetch_val(data[i * size + j]) << " ";
        cout << endl;
    }
    cout << endl;
}

#endif //CUDA_TEST_CLION_VECADD_H
