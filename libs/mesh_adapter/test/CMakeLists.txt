add_executable(mmg_2d_metric mmg_2d_metric.cpp)
target_link_libraries(mmg_2d_metric
    PUBLIC
    CLI11::CLI11
    spdlog::spdlog
    igl::core
    zjucad::mesh_adapter
    zjucad::file_io
)
