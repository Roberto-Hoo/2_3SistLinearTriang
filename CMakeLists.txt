cmake_minimum_required(VERSION 3.17)
project(2_3SistLinearTriang)

add_executable(2_3SistLinearTriang SistLinearTriang.cpp)

#Acesso ao GSL-GNU
target_link_libraries(2_3SistLinearTriang -lgsl -lgslcblas -lm)

#Acesso ao OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
set(SOURCE_FILES SistLinearTriang.cpp)