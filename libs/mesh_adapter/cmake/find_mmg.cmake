set(MMG_VERSION v5.8.0)

if(NOT TARGET mmg)
    message(STATUS "${PROJECT_NAME} fetch mmg ${MMG_VERSION}")
    include(FetchContent)
    FetchContent_Declare(
        mmg
        GIT_REPOSITORY https://github.com/MmgTools/mmg.git
        GIT_TAG ${MMG_VERSION}

        GIT_SHALLOW TRUE
        UPDATE_COMMAND ""
    )
    # mmg_2d_metric links libmmg2d, libmmgs, and libmmg3d. MMG's own
    # CMakeLists derives those three flags from BUILD, so set BUILD as well.
    set(BUILD "MMG" CACHE STRING "Build mmg2d, mmgs, and mmg3d" FORCE)
    set(BUILD_MMG2D ON CACHE BOOL "2D build" FORCE)
    set(BUILD_MMGS ON CACHE BOOL "2D Surface build" FORCE)
    set(BUILD_MMG3D ON CACHE BOOL "3D build" FORCE)

    FetchContent_MakeAvailable(mmg)

    find_package(SCOTCH REQUIRED)
    message(STATUS "SCOTCH ${SCOTCH_LIBRARIES}")
    if(NOT TARGET libmmg2d_a OR NOT TARGET libmmgs_a OR NOT TARGET libmmg3d_a)
        message(FATAL_ERROR
            "mmg_2d_metric requires the static libraries libmmg2d_a, libmmgs_a, and libmmg3d_a")
    endif()

    add_library(mmg INTERFACE)
    add_library(mmg::mmg ALIAS mmg)
    target_include_directories(mmg
        INTERFACE
        ${mmg_BINARY_DIR}/include
        ${SCOTCH_INCLUDE_DIRS}
    )
    target_link_libraries(mmg
        INTERFACE
        libmmg2d_a
        libmmgs_a
        libmmg3d_a
        ${SCOTCH_LIBRARIES}
    )
else()
    message(STATUS "${PROJECT_NAME} mmg ${MMG_VERSION} FOUND ")
endif(NOT TARGET mmg)