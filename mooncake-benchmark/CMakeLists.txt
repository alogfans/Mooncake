add_executable(dynamo_benchmark dynamo/benchmark.cpp)
include_directories(/opt/nvidia/nvda_nixl/include)
target_link_directories(dynamo_benchmark PUBLIC /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu)
target_link_libraries(dynamo_benchmark PUBLIC glog gflags nixl nixl_build boost_system)
