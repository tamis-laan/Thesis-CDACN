#CMake version
cmake_minimum_required (VERSION 2.8)

#Project name
project (ACROBOT)

#Required packages
find_package(OpenGL REQUIRED)
find_package(GLUT   REQUIRED)

#Gather source
file(GLOB SOURCES ./src/*.cpp)

#Gather headers
file(GLOB HEADERS ./src/*.h)

#Includes
include_directories(SYSTEM ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

#Include ddnet/ilm
include_directories(SYSTEM ${DDNET_SOURCE_DIR}/src ${ILM_SOURCE_DIR}/src)

#Binary
add_executable(acrobot main.cpp ${SOURCES})
target_link_libraries(acrobot ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ddnet cudnn curand ilm lz4 SDL2)