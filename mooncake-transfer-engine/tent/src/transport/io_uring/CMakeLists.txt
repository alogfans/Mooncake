if (URING_LIB AND URING_INCLUDE)
    file(GLOB TEV1_XPORT_URING_SOURCES "*.cpp")
    add_library(tent_xport_uring STATIC ${TEV1_XPORT_URING_SOURCES})
    target_include_directories(tent_xport_uring PUBLIC ${URING_INCLUDE})
    target_link_libraries(tent_xport_uring PUBLIC ${URING_LIB} tent_platform tent_common)
    if (CUDAToolkit_FOUND)
        target_link_libraries(tent_xport_uring PUBLIC
            CUDA::cudart CUDA::cuda_driver)
    endif()

    # Build as plugin shared library
    if (USE_TENT_PLUGIN)
        add_library(tent_iouring SHARED ${TEV1_XPORT_URING_SOURCES})
        target_include_directories(tent_iouring PUBLIC ${URING_INCLUDE})
        target_link_libraries(tent_iouring PUBLIC ${URING_LIB} tent_platform tent_common)
        if (CUDAToolkit_FOUND)
            target_link_libraries(tent_iouring PUBLIC
                CUDA::cudart CUDA::cuda_driver)
        endif()
        target_compile_definitions(tent_iouring PRIVATE TENT_BUILD_PLUGIN)
        set_target_properties(tent_iouring PROPERTIES OUTPUT_NAME "tent_iouring")
        install(TARGETS tent_iouring
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
    endif()
endif()
