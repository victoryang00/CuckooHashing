#include "cuckoo.cuh"
#include <algorithm>
#include <chrono>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "cudaHeaders.h"
#include <random>
#include <algorithm>

using namespace std;

using Time = std::chrono::_V2::system_clock::time_point;

Time start_timer() 
{
    return std::chrono::high_resolution_clock::now();
}

double get_elapsed_time(Time start) 
{
    Time end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> d = end - start;
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
    return us.count() / 1000.0f;
}
void generate_random_keyvalues(std::mt19937 &rnd, uint32_t numkvs,uint32_t* set) {
    std::uniform_int_distribution<uint32_t> dis(0, INT_MAX - 1);
    for (uint32_t i = 0; i < numkvs; i++) {
        set[i]=dis(rnd);
    }
}

int main(int argc, char **argv) {
    std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);
    printf("Random number generator seed = %u\n", seed);

    cudaDeviceProp  prop;
    int count;
    cudaGetDeviceCount( &count ); 
    for (int i=0; i< 1; i++) {
        cudaGetDeviceProperties( &prop, i );
        printf( " --- General Information for device %d ---\n", i ); printf( "Name: %s\n", prop.name );
        printf( "Compute capability: %d.%d\n", prop.major, prop.minor ); printf( "Clock rate: %d\n", prop.clockRate );
        printf( "Device copy overlap: " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" ); 
        else
            printf( "Disabled\n" );
        printf( "Kernel execition timeout : " ); 
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" ); 
        else
            printf( "Disabled\n" );
        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1],prop.maxGridSize[2] );
        printf( "\n" );
    } 

    cout<<"experiment 1"<<endl;
    int input_size = pow(2, 25);
for(int s=0;s<25;s++){
    int size = pow(2, s);
    auto *key1=new uint32_t[size];
    auto *value1=new uint32_t[size];
    auto *search1=new uint32_t[size];
    auto *chck1=new uint32_t[size];

    generate_random_keyvalues(rnd, size, key1);
    generate_random_keyvalues(rnd, size, value1);
    generate_random_keyvalues(rnd, size, search1);

    Time timer = start_timer();
    CuckooHashing h(input_size);
    h.hash_insert(key1,value1,size);
    h.hash_search(key1,chck1,size);
    double milliseconds = get_elapsed_time(timer);
    printf("Total time (including memory copies, readback, etc): %f ms for 2^%d random integers to 2^25 hash table\n", milliseconds,s);

    //check
    int tmp=0;
    for(int i=0;i<size;i++){
        if(chck1[i]!=value1[i]){
            tmp++;
        }
    }

    delete[] key1;
    delete[] value1;
    delete[] search1;
    delete[] chck1;
}
    cout<<"experiment 2"<<endl;

        int size = pow(2, 24);
        auto *key = new uint32_t[size];
        auto *value = new uint32_t[size];
        auto *search = new uint32_t[size];
        auto *chck = new uint32_t[size];
        generate_random_keyvalues(rnd, size, value);
        generate_random_keyvalues(rnd, size, key);
        for (int percent = 0; percent <= 10; ++percent) {
            int bound = ceil((1 - 0.1 * percent) * size);
                // printf("sb");
                for (int i = 0; i < bound; ++i) {
                    chck[i] = value[rand() % size];
                    // cout << chck[i] << " ";
                }
                generate_random_keyvalues(rnd, size, chck+bound);

                // for(int k=0;k<sizeof(chck)/sizeof(uint32_t);k++){
                //     cout<<chck[k]<<" ";
                // }
                std::cout <<"Let percentile be" << " " << percent << " and we can get:";
                CuckooHashing h(input_size);
                // printf("sb");
                h.hash_insert(key, value, size);
                
                Time timer = start_timer();
                h.hash_search(key, chck, size);
                double milliseconds = get_elapsed_time(timer);
                printf("Total time : %f ms for (100 − 10*%d) probability to "
                       "insert.\n",
                       milliseconds, percent);
        }

    cout<<"experiment 3"<<endl;

    int n = pow(2, 24);
    float ratios[] = {1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01};
    generate_random_keyvalues(rnd, size, value);
    auto *key2 = new uint32_t[size];
    auto *value2 = new uint32_t[size];
    auto *search2 = new uint32_t[size];
    auto *chck2 = new uint32_t[size];
    for (int ri = 0; ri < 12; ++ri) {
        int size = ceil(ratios[ri] * n);

        std::cout << "Let ratios be"
                  << " " << ratios[ri] << " and we can get:";
        CuckooHashing h(size);
        // printf("sb");
        h.hash_insert(key2, value2, n);

        Time timer = start_timer();
        h.hash_search(key2, chck2, n);
        double milliseconds = get_elapsed_time(timer);
        printf("Total time : %f ms for to "
               "insert.\n",
               milliseconds);
    }
    // delete[] key;
    // delete[] value;
    // delete[] search;
    // delete[] chck;
    
    cout<<"experiment 4"<<endl;
    int size4 = ceil(1.4 * n);
    auto *key4 = new uint32_t[size4];
    auto *value4 = new uint32_t[size4];
    auto *search4 = new uint32_t[size4];
    auto *chck4 = new uint32_t[size4];
    generate_random_keyvalues(rnd, size4, value4);
    generate_random_keyvalues(rnd, size4, key4);
    for (int percent = 0; percent <= 10; ++percent) {
        int bound = ceil((1 - 0.1 * percent) * size4);
        for (int i = 0; i < bound; ++i) {
            chck4[i] = value4[rand() % size4];
            // cout << chck[i] << " ";
        }
        generate_random_keyvalues(rnd, size4, chck4 + bound);
        printf("Let ratios be 1.4 and bound %d we can get:", percent);
        CuckooHashing h(size4);
        h.hash_insert(key4, value4, n);

        Time timer = start_timer();
        h.hash_search(key4, chck4, n);
        double milliseconds = get_elapsed_time(timer);
        printf("Total time : %f ms to "
               "insert.\n",
               milliseconds);
    }

    // delete[] key;
    // delete[] value;
    // delete[] search;
    // delete[] chck;
    return 0;
}