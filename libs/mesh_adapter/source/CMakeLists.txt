set(target_name "mesh_adapter")
add_library(${target_name} STATIC
    mmg_adapter.cpp
    metric_normalize.cpp
)
add_library(zjucad::${target_name} ALIAS ${target_name})
target_include_directories(${target_name}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${target_name}
    PUBLIC
    Eigen3::Eigen
    igl::core
    mmg::mmg
    spdlog::spdlog
)
