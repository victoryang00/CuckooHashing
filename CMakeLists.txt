cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
#ifdef(WIN32)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin/nvcc.exe")
#elif(UNIX)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#endif
project(CuckooHashing LANGUAGES CUDA CXX)

###########################################
## User reqiured to update this variable ##
###########################################

## Set for CXX standard ##
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
## Set for CUDA standar ##
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

##################################################
## Special treatment for current graphical card ##
##################################################
set(CUDA_ARCH "-arch=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH}")

##################################################
## To specify which function is the tested one  ##
##################################################
option(CUCKOO_CPU "select to use specified function" ON)
#ifdef(CUCKOO_CPU)
find_package(MPI REQUIRED)
# Backwards compatibility
include_directories(${MPI_INCLUDE_PATH})
#elif(CUCKOO_GPU)
option(CUCKOO_GPU "select to use specified function" ON)
#elif(CUCKOO_MUL_GPU)
option(CUCKOO_MUL_GPU "select to use specified function" ON)
#endif

# ========================================================================== #
# source code
# ========================================================================== #

set(CXX_SRCS demo.cpp)
set(CUDA_SRCS cudaMain.cu cuckoo.cu)
set(CXX_CUDA_HDR cudaHeaders.h cudaMain.h cuckoo.h)

# ========================================================================== #
# main program
# ========================================================================== #

add_executable(${PROJECT_NAME} ${CXX_SRCS} ${CXX_CUDA_HDR} ${CUDA_SRCS} main.cpp)

target_link_libraries(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )