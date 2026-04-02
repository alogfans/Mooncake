# Platform Backend Plugins
# Single backend.cpp compiled multiple times for different platforms

# Set common include directory for all platform plugins
set(TENT_PLATFORM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Set output directory for plugins
set(PLATFORM_PLUGIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/tent/platform")

file(GLOB BACKEND_SOURCES "*.cpp")

# Find available platforms
set(PLATFORMS_FOUND "")

# Common function to build a platform plugin
# Usage: add_platform_plugin(<name> <compile_flag> <lib_name> [include_dirs])
function(add_platform_plugin NAME COMPILE_FLAG LIB_NAME)
    list(APPEND PLATFORMS_FOUND "${NAME}")
    add_library(tent_platform_${NAME} SHARED ${BACKEND_SOURCES})

    # Set compile definitions (PUBLIC for global visibility)
    target_compile_definitions(tent_platform_${NAME} PUBLIC
        ${COMPILE_FLAG}
        TENT_BUILD_PLUGIN
        HAS_DEVICE_SUPPORT
    )

    # Set include directories
    target_include_directories(tent_platform_${NAME} PUBLIC
        ${TENT_PLATFORM_INCLUDE_DIR}
        ${ARGN}  # Optional additional include dirs
    )

    # Set output properties
    set_target_properties(tent_platform_${NAME} PROPERTIES
        OUTPUT_NAME "${LIB_NAME}"
        LIBRARY_OUTPUT_DIRECTORY "${PLATFORM_PLUGIN_OUTPUT_DIR}"
    )

    # Install target
    install(TARGETS tent_platform_${NAME} LIBRARY DESTINATION lib/tent/platform)

    message(STATUS "  Platform plugin: ${NAME}")
endfunction()

# ============================================================================
# GPU Platform Plugins
# ============================================================================

# CUDA
if (CUDAToolkit_FOUND)
    add_platform_plugin(cuda USE_CUDA tent_platform_cuda ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(tent_platform_cuda PUBLIC CUDA::cudart CUDA::cuda_driver numa)
endif()

# MUSA (Moore Threads)
if (USE_MUSA)
    add_platform_plugin(musa USE_MUSA tent_platform_musa)
    target_link_libraries(tent_platform_musa PUBLIC musa musart numa)
endif()

# HIP (AMD)
if (USE_HIP)
    add_platform_plugin(hip USE_HIP tent_platform_hip)
    target_link_libraries(tent_platform_hip PUBLIC amdhip64 numa)
endif()

# MACA (Iluvatar)
if (USE_MACA)
    add_platform_plugin(maca USE_MACA tent_platform_maca)
    target_link_libraries(tent_platform_maca PUBLIC mcr numa)
endif()

# Ascend (Huawei)
if (USE_ASCEND OR USE_ASCEND_DIRECT)
    add_platform_plugin(ascend USE_ASCEND tent_platform_ascend)
    target_link_libraries(tent_platform_ascend PUBLIC ascendcl numa)
endif()

# ============================================================================
# CPU Fallback (always available)
# ============================================================================
add_library(tent_platform_cpu SHARED ${BACKEND_SOURCES})
target_compile_definitions(tent_platform_cpu PUBLIC TENT_BUILD_PLUGIN)
target_include_directories(tent_platform_cpu PUBLIC ${TENT_PLATFORM_INCLUDE_DIR})
target_link_libraries(tent_platform_cpu PUBLIC numa)
set_target_properties(tent_platform_cpu PROPERTIES
    OUTPUT_NAME "tent_platform_cpu"
    LIBRARY_OUTPUT_DIRECTORY "${PLATFORM_PLUGIN_OUTPUT_DIR}"
)
install(TARGETS tent_platform_cpu LIBRARY DESTINATION lib/tent/platform)
message(STATUS "  Platform plugin: CPU (fallback)")

message(STATUS "Platform backends found: ${PLATFORMS_FOUND}")
