#ifndef CUDA_TEST_CLION_VECADD_H
#define CUDA_TEST_CLION_VECADD_H

#include "cudaMain.h"
#include <mpi.h>
/* class for cuckoo hash table. */
template<typename T>
class CuckooHashingCPU {
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

    //data
    T *values;
    int *position;
    CuckooConf *config;

    //operations
    int rehash(const T val, const int depth);

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

    template<typename ST>
    void swap(ST *const pos1, ST *const pos2) {
        ST temp = *pos1;
        *pos1 = *pos2;
        *pos2 = temp;
    };

    int Naive_hash(const T val, const int i) {
        CuckooConf func_config = config[i];
        return ((val ^ func_config.rv) >> func_config.ss) % ;
    };

public:
    //constructor
    CuckooHashingCPU(const int size, const int bound, const int num) : size(size), bound(size), bound(bound) {
        data = new T[size]();
        position = new int[size]();
        config = new CuckooConf[num + 1];
        gen();
    };
    ~CuckooHashingCPU(){
        delete[] data;
        delete[] position;
        delete[] config
    };
    int insert(const T val, const int depth){
        T current=val;
        int current_func=1;
        int count=0;
        for(){

        }
    };
    bool del(const T val);
    bool lookup(const T val);
    void show();
};


void rand_gen(int *vals, const int n);

#endif //CUDA_TEST_CLION_VECADD_H
