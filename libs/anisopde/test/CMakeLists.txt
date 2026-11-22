set(target_names
    discrete_error
    field_projection
    field_metric_ref
)

foreach(target_name IN LISTS target_names)
    add_executable(${target_name} ${target_name}.cpp)
    target_include_directories(${target_name}
        PUBLIC
        ${CHOLMOD_INCLUDES}
    )
    target_link_libraries(${target_name}
        PUBLIC
        CLI11::CLI11
        spdlog::spdlog
        Eigen3::Eigen
        igl::core
        yaml_cpp::yaml_cpp
        zjucad::file_io
        zjucad::metric_utils
        zjucad::pde_utils
        ${CHOLMOD_LIBRARIES}
    )
endforeach()
