cmake_minimum_required(VERSION 3.16)
project(MetricUtils)

set(default_build_type "Release")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(find_spdlog)
include(find_eigen3)
include(find_libigl)

add_subdirectory(source)

# Detects whether this is a top-level project
get_directory_property(LIB_METRIC_UTILS_PARENT_DIR PARENT_DIRECTORY)

if(NOT LIB_METRIC_UTILS_PARENT_DIR)
    set(LIB_METRIC_UTILS_TOPLEVEL_PROJECT ON)
else()
    set(LIB_METRIC_UTILS_TOPLEVEL_PROJECT OFF)
endif()

option(LIB_METRIC_UTILS_BUILD_TESTS "Build unit test" ${LIB_METRIC_UTILS_TOPLEVEL_PROJECT})

if(LIB_METRIC_UTILS_BUILD_TESTS)
    add_subdirectory(test)
endif()