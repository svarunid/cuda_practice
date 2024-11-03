cmake_minimum_required(VERSION 3.30)
project(CUDA_Practice LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang++-18)

set(SOURCES vector_add.cu matrix_add.cu matrix_mul.cu)

foreach(SRC_FILE ${SOURCES})
    get_filename_component(TARGET_NAME ${SRC_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${SRC_FILE})
endforeach()