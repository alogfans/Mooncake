if (USE_ASCEND OR USE_ASCEND_DIRECT)
    file(GLOB TEV1_XPORT_ASCEND_DIRECT_SOURCES "*.cpp")
    add_library(tent_xport_ascend_direct STATIC ${TEV1_XPORT_ASCEND_DIRECT_SOURCES})
    target_link_libraries(tent_xport_ascend_direct PUBLIC tent_platform tent_common)
    target_link_libraries(tent_xport_ascend_direct PUBLIC ascendcl cann_hixl metadef)

    # Build as plugin shared library
    if (USE_TENT_PLUGIN)
        add_library(tent_ascend SHARED ${TEV1_XPORT_ASCEND_DIRECT_SOURCES})
        target_link_libraries(tent_ascend PUBLIC tent_platform tent_common)
        target_link_libraries(tent_ascend PUBLIC ascendcl cann_hixl metadef)
        target_compile_definitions(tent_ascend PRIVATE TENT_BUILD_PLUGIN)
        set_target_properties(tent_ascend PROPERTIES OUTPUT_NAME "tent_ascend")
        install(TARGETS tent_ascend
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
    endif()
endif()
