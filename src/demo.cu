#include <map>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include "cudaHeaders.h"
#include "cuckoo.cuh"
#include "cuckoo.cu"

using namespace std;
#define DEMO
#define DEBUG
#define CUCKOO_GPU
#ifdef DEMO
int main(){
    int repeat = 100;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
#if (!defined(CUCKOO_MUL_GPU)) && (!defined(CUCKOO_GPU))
    #ifndef  CUCKOO_MUL_CPU
    cout << "Start Serial CPU implementation DEMO -->" << endl;
    #endif
    #ifdef  CUCKOO_MUL_CPU
    cout << "Start Parallel CPU implementation DEMO -->" << endl;
    #endif
     for (int i = 0; i < repeat; i++) {
#ifdef DEBUG
        cout << "Serial implementation DEMO for basic insert-->" << i << endl;
#endif
        {
            CuckooHashing<uint32_t> table_serial(8, 4 * ceil(log2((double) 8)), 3);
#ifdef DEBUG
            table_serial.show();
            cout << "Insert 8 values -" << endl;
#endif
            int vals_to_insert[8];
            rand_gen(vals_to_insert, 8);
            for (int i = 0; i < 8; ++i)
                table_serial.insert(vals_to_insert[i], 0);
#ifdef DEBUG
            table_serial.show();
            cout << "Delete values [0..4] -" << endl;
#endif
            for (int i = 0; i < 4; ++i)
                table_serial.del(vals_to_insert[i]);
#ifdef DEBUG
            table_serial.show();
            cout << "Lookup values [2..6] -" << endl;
#endif
            bool results[4];
            for (int i = 0; i < 4; ++i)
                results[i] = table_serial.lookup(vals_to_insert[i + 2]);
#ifdef DEBUG
            cout << "Results - ";
            for (int i = 0; i < 4; ++i)
                cout << results[i] << " ";
            cout << endl;
            table_serial.show();
#endif
        }
    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    double time = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    #ifndef  CUCKOO_MUL_CPU
    cout << "Time for CPU serial demo = " << time / repeat / 1000 << endl;
    #endif
    #ifdef  CUCKOO_MUL_CPU
    cout << "Time for CPU parallel demo = " << time / repeat / 1000 << endl;
    #endif
#endif



#ifdef CUCKOO_GPU
    cout << "Start Parallel GPU implementation DEMO -->" << endl;
    CuckooHashing<int> table_cuda(8, 4 * ceil(log2((double)8)), 3);
    table_cuda.show();

    cout << "Insert 8 values -";
    int vals_to_insert[8];
    rand_gen(vals_to_insert, 8);
    for (int i = 0; i < 8; i++) {
        std::cout << vals_to_insert[i] << " ";
    }
    std::cout<< std::endl;
    table_cuda.insert(vals_to_insert, 8, 0);
    table_cuda.show();

    // std::cout << "Delete values [0..4] -" << std::endl;
    // int vals_to_delete[4];
    // for (int i = 0; i < 4; ++i)
    //     vals_to_delete[i] = vals_to_insert[i];
    // table_cuda.del(vals_to_delete, 4);
    // table_cuda.show();

    // std::cout << "Lookup values [2..6] -" << std::endl;
    // int vals_to_lookup[4];
    // for (int i = 0; i < 4; ++i)
    //     vals_to_lookup[i] = vals_to_insert[i + 2];
    // bool results[4];
    // table_cuda.lookup(vals_to_lookup, results, 4);
    // std::cout << "Results - ";
    // for (int i = 0; i < 4; ++i)
    //     std::cout << results[i] << " ";
    // std::cout << std::endl;
    // table_cuda.show();

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    double time = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    #ifdef  CUCKOO_GPU
    cout << "Time for GPU demo = " << time / repeat / 1000 << endl;
    #endif
#endif

#ifdef CUCKOO_GPU
    cout << "Start Parallel GPU implementation DEMO -->" << endl;
#endif

}
#endif