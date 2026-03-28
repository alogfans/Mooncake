cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(mc_tent LANGUAGES CXX)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build transport plugins as shared libraries for dynamic loading
option(USE_TENT_PLUGIN "Build transport plugins for dynamic loading" ON)

add_subdirectory(src)
add_subdirectory(plugins)
add_subdirectory(tests)
