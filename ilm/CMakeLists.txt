#Minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

#Project name
project(ILM)

#Gather source files
file(GLOB SOURCES ./src/*.cu ./src/*/*.cu ./src/*/*/*.cu ./src/*/*/*/*.cu ./src/*.cpp ./src/*/*.cpp ./src/*/*/*.cpp ./src/*/*/*/*.cpp)

#Gather header files
file(GLOB HEADERS ./src/*.h ./src/*/*.h ./src/*/*/*.h ./src/*/*/*/*.h)

#Add ddnet headers
include_directories(SYSTEM ${DDNET_SOURCE_DIR}/src)

#Add a new executable and the files required to build it
cuda_add_executable(ilm_run main.cpp ${SOURCES})
target_link_libraries(ilm_run ddnet cuda curand cudnn lz4 ale SDL SDL_image SDL_gfx)

#Add a new executable and the files required to build it
cuda_add_executable(ilm_test test.cpp ${SOURCES})
target_link_libraries(ilm_test ddnet cuda curand cudnn lz4)

#Add a new executable and the files required to build it
cuda_add_executable(ilm_profile profile.cpp ${SOURCES})
target_link_libraries(ilm_profile ddnet cuda curand cudnn lz4 ale SDL SDL_image SDL_gfx nvToolsExt)

#Add the library form
cuda_add_library(ilm SHARED ${SOURCES} ${HEADERS})
target_link_libraries(ilm ddnet cuda curand cudnn lz4)

# TODO LIST:
# 1. NVIDIA/VALGRIND profile the system
# 2. Implement the Deep Actor Critic Network
# 3. Optimize code
# 4. Add a Log class as a Singleton
# 5. Include profiling ability.
# 6. Better Naming!!
# 7. We need a way of specifing the action dimentionality and their domain and if a action dim is discrete/continues
# 8. Create test casses, make sure every object works.
# 9. Create a profile version and check efficiency.
# 10. Check the weight initialization if it is realy correct (Xavier initalization).
# 11. The current class structure stinks!! Make things nicer!! And more generic!!

# MICRO TODO LIST:
# 0.  Try the system with all the actions to see if we still move towards the correct actions. 
# 1.  Commit
# 2.  Consolidate the two networks.
# 3.  Commit
# 6.  Clean up and optimize DACN class.
# 7.  Commit
# 10. Improve peformance through kernalization. (DACN and DQN)
# 11. Commit
# 12. Upgrade to cuda 7.0 toolkit and implement improvements based on streaming.
# 13. Commit

# NOTES:
# 1. In order to get better result we need an accurate Q value and a accurate mu value. If the learning rate of the mu network is to large we may jump around the 
#    the optimal acation. If the Q learning rate is to large we might not have enough accuracy in the Q values.
# 2. Q value bias might be playing a role in hindering convergence. 
