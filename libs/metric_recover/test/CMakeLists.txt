add_executable(metric_from_edge metric_from_edge.cpp)
target_link_libraries(metric_from_edge
    PUBLIC
    CLI11::CLI11
    spdlog::spdlog
    zjucad::metric_utils
    zjucad::metric_recover_utils
    zjucad::file_io
)
