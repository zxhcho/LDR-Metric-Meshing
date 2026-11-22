set(target_name "metric_recover_utils")
add_library(${target_name} INTERFACE)
add_library(zjucad::${target_name} ALIAS ${target_name})
target_include_directories(${target_name} INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

file(GLOB subdirs LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(subdir ${subdirs})
    if(IS_DIRECTORY ${subdir})
        get_filename_component(subdir_name ${subdir} NAME)
        add_subdirectory(${subdir_name})
        target_link_libraries(${target_name} INTERFACE
            zjucad::metric_recover_${subdir_name}
        )
    endif()
endforeach()
