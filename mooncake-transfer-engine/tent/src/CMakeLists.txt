SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(tent_interface INTERFACE)
target_include_directories(tent_interface
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(tent_interface INTERFACE cxx_std_20)

if (TARGET asio_shared)
    target_link_libraries(tent_interface INTERFACE asio_shared)
endif()

# CUDA
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA: Enabled")
    target_compile_definitions(tent_interface INTERFACE USE_CUDA)
    target_include_directories(tent_interface INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(tent_interface INTERFACE CUDA::cudart)
else()
    message(STATUS "CUDA: Disabled")
endif()

# GDS
find_library(CUFILE_LIB cufile PATHS /usr/local/cuda/lib64)
find_path(CUFILE_INCLUDE cufile.h PATHS /usr/local/cuda/include)
if (CUDAToolkit_FOUND AND CUFILE_LIB AND CUFILE_INCLUDE)
    message(STATUS "GDS: Enabled")
    target_compile_definitions(tent_interface INTERFACE USE_GDS)
    target_include_directories(tent_interface INTERFACE ${CUFILE_INCLUDE})
    target_link_libraries(tent_interface INTERFACE ${CUFILE_LIB})
else()
    message(STATUS "GDS: Disabled")
endif()

# RDMA (ibverbs)
find_library(IBVERBS_LIB ibverbs PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64)
find_path(IBVERBS_INCLUDE infiniband/verbs.h PATHS /usr/include /usr/local/include)
if (IBVERBS_LIB AND IBVERBS_INCLUDE)
    message(STATUS "RDMA: Enabled")
    target_compile_definitions(tent_interface INTERFACE USE_RDMA)
    target_include_directories(tent_interface INTERFACE ${IBVERBS_INCLUDE})
    target_link_libraries(tent_interface INTERFACE ${IBVERBS_LIB})
else()
    message(STATUS "RDMA: Disabled")
endif()

# io_uring
find_library(URING_LIB uring PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64)
find_path(URING_INCLUDE liburing.h PATHS /usr/include /usr/local/include)
if (URING_LIB AND URING_INCLUDE)
    message(STATUS "Uring: Enabled")
    target_compile_definitions(tent_interface INTERFACE USE_URING)
    target_include_directories(tent_interface INTERFACE ${URING_INCLUDE})
    target_link_libraries(tent_interface INTERFACE ${URING_LIB})
else()
    message(STATUS "Uring: Disabled")
endif()

set(YALANTING_TARGET yalantinglibs::yalantinglibs)
if (NOT TARGET ${YALANTING_TARGET})
    find_package(yalantinglibs CONFIG REQUIRED)
endif()

# ============================================================================
# Build mode selection
# ============================================================================
# Three modes:
#   static  - Pure static library (libtent.a), all transports linked
#   shared  - Single shared library (libtent.so), all transports linked
#   plugin  - Shared library (libtent.so) + transport plugins (*.so)

set(TENT_BUILD_MODE "static" CACHE STRING "TENT build mode: static, shared, or plugin")
set_property(CACHE TENT_BUILD_MODE PROPERTY STRINGS "static;shared;plugin")

if(TENT_BUILD_MODE STREQUAL "plugin")
    # Mode 3: Plugin mode - core .so + transport plugins
    message(STATUS "TENT Build Mode: PLUGIN (libtent.so + transport_*.so)")
    set(TENT_USE_PLUGINS ON)
    set(TENT_BUILD_SHARED ON)
elseif(TENT_BUILD_MODE STREQUAL "shared")
    # Mode 2: Shared mode - single .so with all transports
    message(STATUS "TENT Build Mode: SHARED (libtent.so with all transports)")
    set(TENT_USE_PLUGINS OFF)
    set(TENT_BUILD_SHARED ON)
else()
    # Mode 1: Static mode (default)
    message(STATUS "TENT Build Mode: STATIC (libtent.a with all transports)")
    set(TENT_USE_PLUGINS OFF)
    set(TENT_BUILD_SHARED OFF)
endif()

add_subdirectory(common)
add_subdirectory(rpc)
add_subdirectory(metastore)
add_subdirectory(runtime)
add_subdirectory(platform)
add_subdirectory(transport)
add_subdirectory(metrics)
add_subdirectory(python)

file(GLOB TENT_ENGINE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(TENT_BUILD_MODE STREQUAL "static")
    # Static mode only: build static library
    add_library(tent STATIC ${TENT_ENGINE_SOURCES})
    target_link_libraries(
        tent
        PUBLIC
            tent_runtime
            tent_interface
    )
else()
    # Shared and plugin modes: build shared library only
    add_library(tent_shared SHARED ${TENT_ENGINE_SOURCES})
    target_link_libraries(
        tent_shared
        PUBLIC
            tent_runtime
            tent_interface
    )
    install(TARGETS tent_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

    # Create alias for tests/examples that use 'tent'
    add_library(tent ALIAS tent_shared)
endif()
