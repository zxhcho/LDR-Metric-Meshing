set(LSQCPP_VERSION 0.3.0)

if(NOT TARGET lsqcpp::lsqcpp)
  message(STATUS "${PROJECT_NAME} fetch LSQCPP ${LSQCPP_VERSION} ")
  include(FetchContent)
  FetchContent_Declare(
    least-squares-cpp
    GIT_REPOSITORY https://github.com/Rookfighter/least-squares-cpp
    GIT_TAG v${LSQCPP_VERSION}
    GIT_SHALLOW TRUE
    UPDATE_COMMAND ""
    GIT_SUBMODULES ""
  )

  FetchContent_MakeAvailable(least-squares-cpp)
else()
  message(STATUS "${PROJECT_NAME} LSQCPP ${LSQCPP_VERSION} FOUND ")
endif(NOT TARGET lsqcpp::lsqcpp)