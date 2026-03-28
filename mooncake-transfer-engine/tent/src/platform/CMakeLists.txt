# Platform Backend Plugins
# Single backend.cpp compiled multiple times for different platforms

# Set common include directory for all platform plugins
set(TENT_PLATFORM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

file(GLOB BACKEND_SOURCES "*.cpp")

# Find available platforms
set(PLATFORMS_FOUND "")

# CUDA
if (CUDAToolkit_FOUND)
    list(APPEND PLATFORMS_FOUND "cuda")
    add_library(tent_platform_cuda SHARED ${BACKEND_SOURCES})
    target_compile_definitions(tent_platform_cuda PRIVATE USE_CUDA TENT_BUILD_PLUGIN)
    target_include_directories(tent_platform_cuda PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENT_PLATFORM_INCLUDE_DIR}
    )
    target_link_libraries(tent_platform_cuda PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        numa
    )
    set_target_properties(tent_platform_cuda PROPERTIES
        OUTPUT_NAME "tent_platform_cuda"
    )
    install(TARGETS tent_platform_cuda LIBRARY DESTINATION lib/tent/platform)
    message(STATUS "  Platform plugin: CUDA")
endif()

# MUSA (Moore Threads)
if (USE_MUSA)
    list(APPEND PLATFORMS_FOUND "musa")
    add_library(tent_platform_musa SHARED ${BACKEND_SOURCES})
    target_compile_definitions(tent_platform_musa PRIVATE USE_MUSA TENT_BUILD_PLUGIN)
    target_include_directories(tent_platform_musa PUBLIC
        ${TENT_PLATFORM_INCLUDE_DIR}
    )
    target_link_libraries(tent_platform_musa PUBLIC musa musart numa)
    set_target_properties(tent_platform_musa PROPERTIES
        OUTPUT_NAME "tent_platform_musa"
    )
    install(TARGETS tent_platform_musa LIBRARY DESTINATION lib/tent/platform)
    message(STATUS "  Platform plugin: MUSA")
endif()

# HIP (AMD)
if (USE_HIP)
    list(APPEND PLATFORMS_FOUND "hip")
    add_library(tent_platform_hip SHARED ${BACKEND_SOURCES})
    target_compile_definitions(tent_platform_hip PRIVATE USE_HIP TENT_BUILD_PLUGIN)
    target_include_directories(tent_platform_hip PUBLIC
        ${TENT_PLATFORM_INCLUDE_DIR}
    )
    target_link_libraries(tent_platform_hip PUBLIC amdhip64 numa)
    set_target_properties(tent_platform_hip PROPERTIES
        OUTPUT_NAME "tent_platform_hip"
    )
    install(TARGETS tent_platform_hip LIBRARY DESTINATION lib/tent/platform)
    message(STATUS "  Platform plugin: HIP")
endif()

# MACA (Iluvatar)
if (USE_MACA)
    list(APPEND PLATFORMS_FOUND "maca")
    add_library(tent_platform_maca SHARED ${BACKEND_SOURCES})
    target_compile_definitions(tent_platform_maca PRIVATE USE_MACA TENT_BUILD_PLUGIN)
    target_include_directories(tent_platform_maca PUBLIC
        ${TENT_PLATFORM_INCLUDE_DIR}
    )
    target_link_libraries(tent_platform_maca PUBLIC mcr numa)
    set_target_properties(tent_platform_maca PROPERTIES
        OUTPUT_NAME "tent_platform_maca"
    )
    install(TARGETS tent_platform_maca LIBRARY DESTINATION lib/tent/platform)
    message(STATUS "  Platform plugin: MACA")
endif()

# Ascend (Huawei)
if (USE_ASCEND OR USE_ASCEND_DIRECT)
    list(APPEND PLATFORMS_FOUND "ascend")
    add_library(tent_platform_ascend SHARED ${BACKEND_SOURCES})
    target_compile_definitions(tent_platform_ascend PRIVATE USE_ASCEND TENT_BUILD_PLUGIN)
    target_include_directories(tent_platform_ascend PUBLIC
        ${TENT_PLATFORM_INCLUDE_DIR}
    )
    target_link_libraries(tent_platform_ascend PUBLIC ascendcl numa)
    set_target_properties(tent_platform_ascend PROPERTIES
        OUTPUT_NAME "tent_platform_ascend"
    )
    install(TARGETS tent_platform_ascend LIBRARY DESTINATION lib/tent/platform)
    message(STATUS "  Platform plugin: Ascend")
endif()

# CPU fallback (always available)
add_library(tent_platform_cpu SHARED ${BACKEND_SOURCES})
target_compile_definitions(tent_platform_cpu PRIVATE TENT_BUILD_PLUGIN)
target_include_directories(tent_platform_cpu PUBLIC
    ${TENT_PLATFORM_INCLUDE_DIR}
)
target_link_libraries(tent_platform_cpu PUBLIC numa)
set_target_properties(tent_platform_cpu PROPERTIES
    OUTPUT_NAME "tent_platform_cpu"
)
install(TARGETS tent_platform_cpu LIBRARY DESTINATION lib/tent/platform)
message(STATUS "  Platform plugin: CPU (fallback)")

message(STATUS "Platform backends found: ${PLATFORMS_FOUND}")
