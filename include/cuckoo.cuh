#ifndef CUDA_TEST_CLION_VECADD_H
#define CUDA_TEST_CLION_VECADD_H

#include "cudaHeaders.h"
#define CUCKOO_GPU

#define HASHING_DEPTH (100)
#define ERROR_DEPTH (-1)

#define BLOCK_SIZE (512)
#define LIMIT (0x1 << 30)

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

/* class for cuckoo hash table. */
template<typename T>
class CuckooHashing {
private:
    //basic config
    typedef struct {
        int rv;
        int ss;
    } CuckooConf;

    //parameter
    int size;
    int bound;
    int num;
    int width;

    //data
    T *data;
    int *position;
    CuckooConf *config;

#if (!defined (CUCKOO_GPU)) || (!defined(CUCKOO_MUL_GPU))

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

#if (defined (CUCKOO_GPU)) || (defined(CUCKOO_MUL_GPU))
    /** Inline helper functions. */
    inline T fetch_val(const T data) {
        return data >> width;
    }
    inline int fetch_func(const T data) {
        return data & ((0x1 << width) - 1);
    }
    int rehash(const T *val, const int n, const int depth) {
        if (depth > HASHING_DEPTH) {
            return ERROR_DEPTH;
        }
        gen();
        vector<T> buffer;
        for (int i = 0; i < _num_funcs; ++i) {
            for (int j = 0; j < size; i++) {
                if (fetch_val(data[i * size + j]) != -1) {
                    buffer.emplace_back(data[i * size + j]);
                }
                data[i] = -1;
            }
        }
        for (int i = 0; i < n; ++i)
            buffer.emplace_back(vals[i]);

        // Re-insert all values.
        int beneath = insert(buffer.data(),buffer.size(), depth);
        if (beneath == ERROR_DEPTH)
            return ERROR_DEPTH;
        else if (beneath > level)
            level = beneath;

        return level;
    };
#endif
public:
    //constructor
    CuckooHashing(const int size, const int bound, const int num) : size(size), bound(bound), num(num) {
        data = new T[size]();
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


#if (!defined (CUCKOO_GPU)) || (!defined(CUCKOO_MUL_GPU))
    //hashing insert operation, the return is the bottom of the rehashed index.
    int insert(const T val, const int depth);

    //hashing del operation, the return is whether is delete success.
    bool del(const T val);

    //hashing lookup operation, the return is whether can be looked up.
    bool lookup(const T val);

#endif
#if (defined (CUCKOO_GPU)) || (defined(CUCKOO_MUL_GPU))
    int insert(const T * val,const int n, const int depth);
    void del(const T * const vals, const int n);
    void lookup(const T * const vals, bool * const results, const int n);
#endif

    //hashing show operation, generate new set of hash functions and rehash, the return is the bottom of the rehashed index.
    void show();

};

//Implementation of Rand_gen
void rand_gen(int *vals, const int n);

//Implementation for CPU
#if (!defined (CUCKOO_GPU)) || (!defined(CUCKOO_MUL_GPU))
template<typename T>
int CuckooHashing<T>::insert(const T val, const int depth) {
    T current = val;
    int current_func = 1;
    int count = 0;
    for (; count < bound; count++) {
        #ifdef CUCKOO_MUL_CPU
        #pragma omp parallel for
        #endif
        for (int i = 0; i < num; i++) {
            int index = (current + i) % num + 1;
            int pos = hash(current, index);
            if (data[pos] == 0) {
                data[pos] = current;
                position[pos] = index;
                return 0;
            }
        }
        int pos = hash(current, current_func);
        swap<T>(&current, &data[pos]);
        swap<int>(&current_func, &position[pos]);
        current_func = current_func % num + 1;
    }
    //evict the unnecessary one
    int level = rehash(current, depth + 1);
    if (level != ERROR_DEPTH) {
        return level + 1;
    }
    return ERROR_DEPTH;
};

template<typename T>
bool CuckooHashing<T>::del(const T val) {
    for (int i = 0; i < num; i++) {
        int pos = hash(val, i + 1);
        if (data[pos] != val) {
            continue;
        } else {
            data[pos] = -1;
            position[pos] = -1;
            return true;
        }
    }
    return false;
};

template<typename T>
bool CuckooHashing<T>::lookup(const T val) {
    for (int i = 0; i < num; i++) {
        int pos = hash(val, i + 1);
        if (data[pos] == val)
            return true;
    }
    return false;
};

template<typename T>
void CuckooHashing<T>::show() {
    cout << "Funcs: ";
    for (int i = 1; i <= num; ++i) {
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

#endif //CUDA_TEST_CLION_VECADD_H
