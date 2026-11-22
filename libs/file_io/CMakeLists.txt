cmake_minimum_required(VERSION 3.16)
project(file_io)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)

    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(find_eigen3)

add_subdirectory(src)

# Detects whether this is a top-level project
get_directory_property(LIB_FILE_IO_PARENT_DIR PARENT_DIRECTORY)

if(NOT LIB_FILE_IO_PARENT_DIR)
    set(LIB_FILE_IO_TOPLEVEL_PROJECT ON)
else()
    set(LIB_FILE_IO_TOPLEVEL_PROJECT OFF)
endif()

option(LIB_FILE_IO_BUILD_TESTS "Build unit test" ${LIB_FILE_IO_TOPLEVEL_PROJECT})

if(LIB_FILE_IO_BUILD_TESTS)
    add_subdirectory(test)
endif()