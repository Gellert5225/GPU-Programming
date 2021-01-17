cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(square LANGUAGES CXX CUDA)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

FILE(GLOB source *.cu)
  
add_executable(square.out ${source})