cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if(WIN32)
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin/nvcc.exe")
else()
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    include_directories(/usr/local/cuda/include)
endif()

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

cuda_add_library(kernel_obj src/cuckoo.cu)

################
# main program #
################
add_executable(${PROJECT_NAME} src/cuckoo.cu src/main.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )