cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
#ifdef(WIN32)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin/nvcc.exe")
#elif(UNIX)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
include_directories(/usr/local/cuda/include)
#endif
project(CuckooHashing LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

###########################################
## User reqiured to update this variable ##
###########################################

## Set for CXX standard ##
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
## Set for CUDA standar ##
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

##################################################
## Special treatment for current graphical card ##
##################################################

set(CUDA_ARCH "-arch=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH}")

##################################################
## To specify which function is the tested one  ##
##################################################

IF (DEBUG)
    option(DEBUG "select to use specified function" ON)
    message(STATUS "DEBUG is ${DEBUG}")
    add_definitions(-DDEBUG)
ENDIF ()

###############
# source code #
###############

set(CXX_SRCS src/mt19937ar.cpp)
set(CXX_CUDA_HDR include/cudaHeaders.h)
cuda_add_library(kernel_obj src/cuckoo.cu)

################
# main program #
################
include_directories(./include)
add_executable(${PROJECT_NAME} ${CXX_SRCS} ${CXX_CUDA_HDR} ${CUDA_SRCS} src/main.cu)
target_link_libraries(${PROJECT_NAME} kernel_obj)
set_target_properties(${PROJECT_NAME} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )