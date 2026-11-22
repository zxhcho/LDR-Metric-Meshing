if(NOT TARGET zjucad::file_io)
    if(NOT FILE_IO_SOURCE_DIR)
        get_filename_component(
            FILE_IO_SOURCE_DIR
            "${CMAKE_CURRENT_LIST_DIR}/../../file_io"
            ABSOLUTE)
    endif()
    if(NOT EXISTS "${FILE_IO_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "file_io was not found at ${FILE_IO_SOURCE_DIR}. "
            "Set FILE_IO_SOURCE_DIR to the libs/file_io directory of this release.")
    endif()
    add_subdirectory("${FILE_IO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/file_io")
endif()
