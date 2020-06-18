#include <map>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdint>
#include "cudaHeaders.h"
#include "cuckoo.cuh"
#include "cudaMain.h"
using namespace std;
#ifdef DEMO
int main(){
    int repeat = 100;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    cout << "Start Serial implementation DEMO -->" << endl;
    for (int i = 0; i < repeat; i++) {
#ifdef DEBUG
        cout << "Serial implementation DEMO for basic insert-->" << i << endl;
#endif
        {
            CuckooHashing<uint32_t> table_serial(8, 4 * ceil(log2((double) 8)), 3);
#ifdef DEBUG
            table_serial.show();
            cout << "Insert 8 values -" << std::endl;
#endif
            int vals_to_insert[8];
            rand_gen(vals_to_insert, 8);
            for (int i = 0; i < 8; ++i)
                table_serial.insert(vals_to_insert[i], 0);
#ifdef DEBUG
            table_serial.show();
            cout << "Delete values [0..4] -" << std::endl;
#endif
            for (int i = 0; i < 4; ++i)
                table_serial.del(vals_to_insert[i]);
#ifdef DEBUG
            table_serial.show();
            cout << "Lookup values [2..6] -" << std::endl;
#endif
            bool results[4];
            for (int i = 0; i < 4; ++i)
                results[i] = table_serial.lookup(vals_to_insert[i + 2]);
#ifdef DEBUG
            cout << "Results - ";
            for (int i = 0; i < 4; ++i)
                cout << results[i] << " ";
            cout << std::endl;
            table_serial.show();
#endif
        }
    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    double time = chrono::duration_cast<chrono::microseconds>(end - begin).count();

    cout << "Time for CPU serial demo = "
         << time / repeat / 1000
         << endl;
}
#endif