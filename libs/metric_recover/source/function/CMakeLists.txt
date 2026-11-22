file(GLOB_RECURSE headers *.h)
file(GLOB_RECURSE sources *.cpp)

set(target_name "metric_recover_function")
add_library(${target_name} STATIC ${sources} ${headers})
add_library(zjucad::${target_name} ALIAS ${target_name})
target_include_directories(${target_name}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${target_name}
    PUBLIC
    spdlog::spdlog
    Eigen3::Eigen
    autodiff::autodiff

    igl::core
    lsqcpp::lsqcpp
)