cmake_minimum_required(VERSION 3.16)
project(AnisoPDE LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(find_spdlog)
include(find_Cholmod)
include(find_eigen3)
include(find_libigl)
include(find_yaml_cpp)
include(find_CLI11)

add_subdirectory(source/pde)
add_subdirectory(test)
