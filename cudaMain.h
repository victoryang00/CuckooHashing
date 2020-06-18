#ifndef CUDA_TEST_CLION_CUDAMAIN_H
#define CUDA_TEST_CLION_CUDAMAIN_H

#include <iostream>
#include <cstdio>

#define HASHING_DEPTH (100)
#define ERROR_DEPTH (-1)

#define BLOCK_SIZE (512)
#define LIMIT (0x1 << 20)

int cudaMain(int argc, char **argv);

#endif //CUDA_TEST_CLION_CUDAMAIN_H
