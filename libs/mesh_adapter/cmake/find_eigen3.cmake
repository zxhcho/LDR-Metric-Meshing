set(EIGEN3_VERSION 3.4.0)
find_package(Eigen3 ${EIGEN3_VERSION} QUIET)

if(NOT TARGET Eigen3::Eigen)
  message(STATUS "${PROJECT_NAME} Fetch Eigen3 ${EIGEN3_VERSION} from source ")
  include(FetchContent)
  FetchContent_Declare(
    eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG ${EIGEN3_VERSION}
    GIT_SHALLOW TRUE
    UPDATE_COMMAND ""
  )

  # Eigen's unit tests define a custom target named autodiff, which
  # collides with the autodiff library fetched for metric_from_edge.
  set(BUILD_TESTING OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
  option(EIGEN_BUILD_DOC " Build doc target for Eigen3 project " OFF)
  FetchContent_MakeAvailable(eigen3)
else()
  message(STATUS "${PROJECT_NAME} Eigen3 ${EIGEN3_VERSION} FOUND ")
endif(NOT TARGET Eigen3::Eigen)