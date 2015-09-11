#Minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

#Project Atari
project(ATARI)

#Required packages
find_package(CUDA REQUIRED)

#Set gcc compiler flags
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -use_ffast-math -D__USE_SDL")

#Set nvcc compiler flags (set architecture)	
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O4 -std=c++11 -gencode arch=compute_30,code=sm_30 -use_fast_math -lm)

#Set include directories (SYSTEM treats headers as system headers which suppresses annoying warnings)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

#Add a library search paths
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/ ${PROJECT_SOURCE_DIR}/lib)

#Add the ddnet project
add_subdirectory(ddnet)

#Add the ilm project
add_subdirectory(ilm)

#Add the breakout project
add_subdirectory(bounce)

#Add the acrobot project
add_subdirectory(acrobot)
