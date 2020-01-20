cmake_minimum_required(VERSION 3.13)
project(RPForest)

add_executable(RPForest main.cpp RPForest.h common.h)
target_link_libraries(RPForest -fopenmp)


