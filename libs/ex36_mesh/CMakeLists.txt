list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(find_eigen3)
include(find_file_io)
include(find_spdlog)
include(find_libigl)
include(find_yaml_cpp)

# Include the binary directory to find generated headers
include_directories(BEFORE ${PROJECT_BINARY_DIR})

add_executable(ex36_mesh ex36_mesh.cpp)
target_link_libraries(
    ex36_mesh
    PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    zjucad::file_io
    yaml_cpp::yaml_cpp
    igl::core
    mfem
)

add_executable(ex36_mesh_AMR ex36_mesh_AMR.cpp)
target_link_libraries(
    ex36_mesh_AMR
    PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    zjucad::file_io
    yaml_cpp::yaml_cpp
    igl::core
    mfem
)