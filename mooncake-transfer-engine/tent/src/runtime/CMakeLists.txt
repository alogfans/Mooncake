# Split runtime into two parts to avoid circular dependency:
# 1. tent_platform - Platform class (used by transports)
# 2. tent_runtime - Rest of runtime (depends on transports)

# Platform library - used by both transports and runtime
add_library(tent_platform STATIC
    platform.cpp
    slab.cpp
    topology.cpp
    segment.cpp
    segment_manager.cpp
    segment_registry.cpp
    metastore.cpp
)

if (TARGET metastore_redis)
  add_definitions(-DUSE_REDIS)
  target_link_libraries(tent_platform PUBLIC metastore_redis)
endif()

if (TARGET metastore_http)
  add_definitions(-DUSE_HTTP)
  target_link_libraries(tent_platform PUBLIC metastore_http)
endif()

if (TARGET metastore_etcd)
  add_definitions(-DUSE_ETCD)
  target_link_libraries(tent_platform PUBLIC metastore_etcd)
endif()

target_link_libraries(tent_platform PUBLIC numa tent_common)

# Rest of runtime (depends on transports)
add_library(tent_runtime_impl STATIC
    control_plane.cpp
    memory_prober.cpp
    proxy_manager.cpp
    segment_tracker.cpp
    transfer_engine_impl.cpp
    transport_loader.cpp
    transport_selector.cpp
)

target_link_libraries(tent_runtime_impl PUBLIC
    numa
    tent_common
    tent_rpc
    tent_transport_all
    tent_metrics
    tent_platform
)

# Combine into final tent_runtime alias library
add_library(tent_runtime ALIAS tent_runtime_impl)
add_library(tent_runtime_all INTERFACE)
target_link_libraries(tent_runtime_all INTERFACE tent_runtime_impl)

message(STATUS "Runtime: Platform-independent core (no GPU dependencies)")
