cmake_minimum_required(VERSION 3.16)
project(MetricRecover LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(find_spdlog)
include(find_autodiff)
include(find_eigen3)
include(find_libigl)
include(find_lsqcpp)
include(find_CLI11)

add_subdirectory(source)
add_subdirectory(test)
