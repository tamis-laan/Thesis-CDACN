#Minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

#Project name
project(DDNET)

#Gather source files
file(GLOB SOURCES ./src/*.cu ./src/*/*.cu ./src/*.cpp ./src/*/*.cpp)

#Gather header files
file(GLOB HEADERS ./src/*.h ./src/*/*.h)

#Add a new executable and the files required to build it
cuda_add_executable(ddnet_run main.cpp ${SOURCES})
target_link_libraries(ddnet_run cuda cudart curand cudnn)

#Add the unit test executable
cuda_add_executable(ddnet_test test.cpp ${SOURCES})
target_link_libraries(ddnet_test cuda cudart curand cudnn)

#Add the profiling executable
cuda_add_executable(ddnet_profile profile.cpp ${SOURCES})
target_link_libraries(ddnet_profile cuda cudart curand nvToolsExt cudnn)

#Add the library form
cuda_add_library(ddnet SHARED ${SOURCES} ${HEADERS})

# TODO LIST:
# 1.  Implement streaming for network class //Prosponed for CUDA 7.0
# 2.  The Cost function J is (h(s)-y)². We devide by 1/batch_size but we might have to device by 1/(2*batch_size) for proper behavior?
# 3.  The Constraint Layer could be more optimized?????
# 4.  Fix const correctness ( const parameters (refs) ) (function const correct) (output const correct)
# 5.  Tensor::transform_src and Tensor::transform_dst are completely incomprehencable!!! Understand and properly document the functions.
# 6.  Write a test case for the update accumulate case
# 7.  Do a clean up of all the classes
# 8.  Implement GatingLayer
# 9.  Implement BatchNormalizationLayer
# 10. Implement a Source type, which can generate data like a bit stream or gaussian noise.
# 11. Implement a Save and Load capability (for layers and network)
# 12. Implement Pretraining Module that uses denoising auto encoder or Variational auto encoder to pre-train layers.
# 	NOTE: Always profile and check async capability!
