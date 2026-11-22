find_package(AUTODIFF ${AUTODIFF_VERSION} QUIET)

if(NOT TARGET autodiff::autodiff)
  message(STATUS "${PROJECT_NAME} Fetch autodiff ${AUTODIFF_VERSION} from source ")
  include(FetchContent)
  FetchContent_Declare(
    autodiff
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG 2e2f3a2b16afcd9c04e76c8a689e9fd23ff78679
    GIT_SHALLOW TRUE
    UPDATE_COMMAND ""
  )

  option(AUTODIFF_BUILD_TESTS "Enable the compilation of the test files." OFF)
  option(AUTODIFF_BUILD_PYTHON "Enable the compilation of the python bindings." OFF)
  option(AUTODIFF_BUILD_EXAMPLES "Enable the compilation of the example files." OFF)
  option(AUTODIFF_BUILD_DOCS "Enable the build of the documentation and website." OFF)

  FetchContent_MakeAvailable(autodiff)
else()
  message(STATUS "${PROJECT_NAME} autodiff ${AUTODIFF_VERSION} FOUND ")
endif(NOT TARGET autodiff::autodiff)