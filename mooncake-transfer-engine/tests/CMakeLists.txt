add_executable(rdma_transport_test rdma_transport_test.cpp)
target_link_libraries(rdma_transport_test PUBLIC transfer_engine)
# add_test(NAME rdma_transport_test COMMAND rdma_transport_test)

add_executable(transport_uint_test transport_uint_test.cpp)
target_link_libraries(transport_uint_test PUBLIC transfer_engine gtest gtest_main )
add_test(NAME transport_uint_test COMMAND transport_uint_test)

add_executable(rdma_transport_test2 rdma_transport_test2.cpp)
target_link_libraries(rdma_transport_test2 PUBLIC transfer_engine gtest gtest_main )
# add_test(NAME rdma_transport_test2 COMMAND rdma_transport_test2)

add_executable(rdma_loopback_test rdma_loopback_test.cpp)
target_link_libraries(rdma_loopback_test PUBLIC transfer_engine gtest gtest_main )
# add_test(NAME rdma_loopback_test COMMAND rdma_loopback_test)

if (USE_NVMEOF)
    add_executable(nvmeof_transport_test nvmeof_transport_test.cpp)
    target_link_libraries(nvmeof_transport_test PUBLIC transfer_engine gtest gtest_main )
    # add_test(NAME nvmeof_transport_test COMMAND nvmeof_transport_test)
endif()

if (USE_TCP)
add_executable(tcp_transport_test tcp_transport_test.cpp)
target_link_libraries(tcp_transport_test PUBLIC transfer_engine gtest gtest_main )
add_test(NAME tcp_transport_test COMMAND tcp_transport_test)
endif()

if (USE_MNNVL)
    add_executable(nvlink_transport_test nvlink_transport_test.cpp)
    target_link_libraries(nvlink_transport_test PUBLIC transfer_engine gtest gtest_main )
    add_test(NAME nvlink_transport_test COMMAND nvlink_transport_test)  
endif()

add_executable(transfer_metadata_test transfer_metadata_test.cpp)
target_link_libraries(transfer_metadata_test PUBLIC transfer_engine gtest gtest_main)
add_test(NAME transfer_metadata_test COMMAND transfer_metadata_test)

add_executable(topology_test topology_test.cpp)
target_link_libraries(topology_test PUBLIC transfer_engine gtest gtest_main)
add_test(NAME topology_test COMMAND topology_test)

add_executable(memory_location_test memory_location_test.cpp)
target_link_libraries(memory_location_test PUBLIC transfer_engine gtest gtest_main)
add_test(NAME memory_location_test COMMAND memory_location_test)

add_executable(common_test common_test.cpp)
target_link_libraries(common_test PUBLIC transfer_engine gtest gtest_main)
add_test(NAME common_test COMMAND common_test)
