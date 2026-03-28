# Python bindings are optional - require pybind11
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    file(GLOB SOURCES "*.cpp")
    set(PYTHON_EXECUTABLE "python3")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print([s for s in sys.path if 'packages' in s][0])"
        OUTPUT_VARIABLE PYTHON_SYS_PATH
    )
    string(STRIP ${PYTHON_SYS_PATH} PYTHON_SYS_PATH)

    if("${PYTHON_SYS_PATH}" STREQUAL "")
        message(WARNING "Python path is empty! Skipping python bindings.")
    else()
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

        message("${PYTHON_SYS_PATH}")

        set(PYTHON_PACKAGE_NAME "tent-core")

        pybind11_add_module(tentpy ${SOURCES})
        set_target_properties(tentpy PROPERTIES
            INSTALL_RPATH "$ORIGIN"
        )

        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        target_link_libraries(tentpy PRIVATE ${Python3_LIBRARIES})
        target_include_directories(tentpy PRIVATE ${Python3_INCLUDE_DIRS})
        target_link_libraries(tentpy PUBLIC tent)
    endif()
else()
    message(STATUS "pybind11 not found - skipping python bindings")
endif()
