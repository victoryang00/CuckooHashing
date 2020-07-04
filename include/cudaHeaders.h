#ifndef CUDA_BASE_CUDAHEADERS_H
#define CUDA_BASE_CUDAHEADERS_H

#include "mt19937ar.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <utility>
#include <vector>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/// hash function
#define _PRIME 4294967291u

/// cuckoo prar
#define BUCKET_SIZE 32
#define MAX_ITERATOR 15
#define TABLE_NUM 4

#define NUM_THREADS 512
#define NUM_BLOCK 64

#ifdef __JETBRAINS_IDE__
#include "math.h"
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __noinline__
#define __forceinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__
// CUDA Synchronization
inline void __syncthreads() {};
inline void __threadfence_block() {};
inline void __threadfence() {};
inline void __threadfence_system();
inline int __syncthreads_count(int predicate) {return predicate;};
inline int __syncthreads_and(int predicate) {return predicate;};
inline int __syncthreads_or(int predicate) {return predicate;};
template<class T> inline T __clz(const T val) { return val; }
template<class T> inline T __ldg(const T* address){return *address;};
// CUDA uint32_tS
uint32_tdef unsigned short uchar;
uint32_tdef unsigned short ushort;
uint32_tdef unsigned int uint;
uint32_tdef unsigned long ulong;
uint32_tdef unsigned long long ulonglong;
uint32_tdef long long longlong;
#endif

class GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;

public:

	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() {
		cudaEventRecord(start, 0);
	}

	void Stop() {
		cudaEventRecord(stop, 0);
	}

	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed / 1000;
	}

};

#endif //CUDA_BASE_CUDAHEADERS_H
