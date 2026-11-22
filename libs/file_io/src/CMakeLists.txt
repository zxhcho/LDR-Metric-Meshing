file(GLOB_RECURSE src *.cpp)

set(target_name "file_io")
add_library(${target_name} STATIC)
add_library(zjucad::${target_name} ALIAS ${target_name})
target_sources(${target_name}
    PUBLIC
    ${src}
)
target_compile_features(${target_name}
    PUBLIC
    cxx_std_17
)
target_include_directories(${target_name}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${target_name}
    PUBLIC
    Eigen3::Eigen
)