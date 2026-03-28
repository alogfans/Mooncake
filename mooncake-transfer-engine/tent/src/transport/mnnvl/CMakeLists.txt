if (CUDAToolkit_FOUND)
    file(GLOB XPORT_SOURCES "*.cpp")
    add_library(tent_xport_mnnvl STATIC ${XPORT_SOURCES})
    target_link_libraries(tent_xport_mnnvl PUBLIC CUDA::cuda_driver CUDA::cudart tent_platform tent_common)

    # Build as plugin shared library
    if (USE_TENT_PLUGIN)
        add_library(tent_mnnvl SHARED ${XPORT_SOURCES})
        target_link_libraries(tent_mnnvl PUBLIC tent_platform CUDA::cuda_driver CUDA::cudart tent_common)
        target_compile_definitions(tent_mnnvl PRIVATE TENT_BUILD_PLUGIN)
        set_target_properties(tent_mnnvl PROPERTIES OUTPUT_NAME "tent_mnnvl")
        install(TARGETS tent_mnnvl
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
    endif()
endif()