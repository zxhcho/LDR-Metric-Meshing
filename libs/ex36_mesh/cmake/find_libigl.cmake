set(IGL_VERSION 2.5.0)
find_package(LIBIGL ${IGL_VERSION} QUIET)

if(NOT TARGET igl::core)
  message(STATUS "${PROJECT_NAME} fetch libigl ${IGL_VERSION} ")
  include(FetchContent)
  FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl
    GIT_TAG v${IGL_VERSION}
    GIT_SHALLOW TRUE
    UPDATE_COMMAND ""
    GIT_SUBMODULES ""
  )

  set(LIBIGL_USE_STATIC_LIBRARY ON CACHE BOOL "Use libigl as static library" FORCE)
  set(LIBIGL_BUILD_TESTS OFF CACHE BOOL "Disable libigl tests" FORCE)
  set(LIBIGL_BUILD_TUTORIALS OFF CACHE BOOL "Disable libigl tutorials" FORCE)

  FetchContent_MakeAvailable(libigl)
else()
  message(STATUS "${PROJECT_NAME} igl ${IGL_VERSION} FOUND ")
endif(NOT TARGET igl::core)