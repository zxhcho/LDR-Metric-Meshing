if(NOT TARGET yaml-cpp)
    message(STATUS "${PROJECT_NAME} fetch yaml-cpp ${YAML_CPP_VERSION} ")
    include(FetchContent)
    FetchContent_Declare(
        yaml_cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 2f86d13775d119edbb69af52e5f566fd65c6953b

        # GIT_SHALLOW TRUE
        UPDATE_COMMAND ""
    )

    option(YAML_CPP_BUILD_TOOLS "Enable parse tools" OFF)
    FetchContent_MakeAvailable(yaml_cpp)
    add_library(yaml_cpp::yaml_cpp ALIAS yaml-cpp)
else()
    message(STATUS "${PROJECT_NAME} yaml-cpp ${YAML_CPP_VERSION} FOUND ")
endif(NOT TARGET yaml-cpp)