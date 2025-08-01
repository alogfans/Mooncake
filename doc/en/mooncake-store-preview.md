# Mooncake Store Preview

## Introduction

Mooncake Store is a high-performance **distributed key-value (KV) cache storage engine** designed specifically for LLM inference scenarios.

Unlike traditional caching systems such as Redis or Memcached, Mooncake Store is positioned as **a distributed KV cache rather than a generic caching system**. The key difference is that in the latter, the key is derived from the value through hashing, so value is immutable after inserting (although the key/value pair may be garbage collected).

Mooncake Store provides low-level object storage and management capabilities, including configurable caching and eviction strategies that offers high memory efficiency and is specifically designed to accelerate LLM inference performance.

Key features of Mooncake Store include:
- **Object-level storage operations**: Mooncake Store provides simple and easy-to-use object-level APIs, including `Put`, `Get`, and `Remove` operations.
- **Multi-replica support**: Mooncake Store supports storing multiple data replicas for the same object, effectively alleviating hotspots in access pressure.
- **Strong consistency**: Mooncake Store Guarantees that `Get` operations always read accurate and complete data, and after a successful write, all subsequent Gets will return the most recent value.
- **Zero-copy, bandwidth-saturating transfers**: Powered by the Transfer Engine, Mooncake Store eliminates redundant memory copies and exploits multi-NIC GPUDirect RDMA pooling to drive data across the network at full line rate while keeping CPU overhead negligible.
- **High bandwidth utilization**: Mooncake Store supports striping and parallel I/O transfer of large objects, fully utilizing multi-NIC aggregated bandwidth for high-speed data reads and writes.
- **Dynamic resource scaling**: Mooncake Store supports dynamically adding and removing nodes to flexibly handle changes in system load, achieving elastic resource management.
- **Fault tolerance**: Mooncake store is designed with robust fault tolerance. Failures of any number of master and client nodes will not result in incorrect data being read. As long as at least one master and one client remain operational, Mooncake Store continues to function correctly and serve requests.​
- **Multi-layer storage support**​​: Mooncake Store supports offloading cached data from RAM to SSD, further balancing cost and performance to improve storage system efficiency.

## Architecture

![architecture](../../image/mooncake-store-preview.png)

As shown in the figure above, there are two key components in Mooncake Store: **Master Service** and **Client**.

**Master Service**: The `Master Service` orchestrates the logical storage space pool across the entire cluster, managing node join and leave events. It is responsible for object space allocation and metadata maintenance. Its memory allocation and eviction strategies are specifically designed and optimized to meet the demands of LLM inference workloads.

The `Master Service` runs as an independent process and exposes RPC services to external components. Note that the `metadata service` required by the `Transfer Engine` (via etcd, Redis, or HTTP, etc.) is not included in the `Master Service` and needs to be deployed separately.

**Client**: In Mooncake Store, the `Client` class is the only class defined to represent the client-side logic, but it serves **two distinct roles**:
1. As a **client**, it is invoked by upper-layer applications to issue `Put`, `Get` and other requests.
2. As a **store server**, it hosts a segment of contiguous memory that contributes to the distributed KV cache, making its memory available to other `Clients`. Data transfer is actually from one `Client` to another, bypassing the `Master Service`.

It is possible to configure a `Client` instance to act in only one of its two roles:
* If `global_segment_size` is set to zero, the instance functions as a **pure client**, issuing requests but not contributing memory to the system.
* If `local_buffer_size` is set to zero, it acts as a **pure server**, providing memory for storage. In this case, request operations such as `Get` or `Put` are not permitted from this instance.

The `Client` can be used in two modes:
1. **Embedded mode**: Runs in the same process as the LLM inference program (e.g., a vLLM instance), by being imported as a shared library.
2. **Standalone mode**: Runs as an independent process.

Mooncake store supports two deployment methods to accommodate different availability requirements:
1. **Default mode**: In this mode, the master service consists of a single master node, which simplifies deployment but introduces a single point of failure. If the master crashes or becomes unreachable, the system cannot continue to serve requests until it is restored.
2. **High availability mode (unstable)**: This mode enhances fault tolerance by running the master service as a cluster of multiple master nodes coordinated through an etcd cluster. The master nodes use etcd to elect a leader, which is responsible for handling client requests.
If the current leader fails or becomes partitioned from the network, the remaining master nodes automatically perform a new leader election, ensuring continuous availability.
The leader monitors the health of all client nodes through periodic heartbeats. If a client crashes or becomes unreachable, the leader quickly detects the failure and takes appropriate action. When a client node recovers or reconnects, it can automatically rejoin the cluster without manual intervention.

## Client C++ API

### Constructor and Initialization `Init`

```C++
ErrorCode Init(const std::string& local_hostname,
               const std::string& metadata_connstring,
               const std::string& protocol,
               void** protocol_args,
               const std::string& master_server_entry);
```

Initializes the Mooncake Store client. The parameters are as follows:
- `local_hostname`: The `IP:Port` of the local machine or an accessible domain name (default value used if port is not included)
- `metadata_connstring`: The address of the metadata service (e.g., etcd/Redis) required for Transfer Engine initialization
- `protocol`: The protocol supported by the Transfer Engine, including RDMA and TCP
- `protocol_args`: Protocol parameters required by the Transfer Engine
- `master_server_entry`: The address information of the Master (`IP:Port` for default mode and `etcd://IP:Port;IP:Port;...;IP:Port` for high availability mode)

### Get

```C++
tl::expected<void, ErrorCode> Get(const std::string& object_key, 
                                  std::vector<Slice>& slices);
```

![mooncake-store-simple-get](../../image/mooncake-store-simple-get.png)


Used to retrieve the value corresponding to `object_key`. The retrieved data is guaranteed to be complete and correct. The retrieved value is stored in the memory region pointed to by `slices` via the Transfer Engine, which can be local DRAM/VRAM memory space registered in advance by the user through `registerLocalMemory(addr, len)`. Note that this is not the logical storage space pool (Logical Memory Pool) managed internally by Mooncake Store.​​(When persistence is enabled, if a query request fails in memory, the system will attempt to locate and load the corresponding data from SSD.)​

> In the current implementation, the Get interface has an optional TTL feature. When the value corresponding to `object_key` is fetched for the first time, the corresponding entry is automatically deleted after a certain period of time (1s by default).

### Put

```C++
tl::expected<void, ErrorCode> Put(const ObjectKey& key,
                                  std::vector<Slice>& slices,
                                  const ReplicateConfig& config);
```

![mooncake-store-simple-put](../../image/mooncake-store-simple-put.png)

Used to store the value corresponding to `key`. The required number of replicas can be set via the `config` parameter.​​(When persistence is enabled, after a successful in-memory put request, an asynchronous persistence operation to SSD will be initiated.)​ The data structure details of `ReplicateConfig` are as follows:

```C++
struct ReplicateConfig {
    size_t replica_num{1};                    // Total number of replicas for the object
    bool with_soft_pin{false};               // Whether to enable soft pin mechanism for this object
    std::string preferred_segment{};         // Preferred segment for allocation
};
```

### Remove

```C++
tl::expected<void, ErrorCode> Remove(const ObjectKey& key);
```

Used to delete the object corresponding to the specified key. This interface marks all data replicas associated with the key in the storage engine as deleted, without needing to communicate with the corresponding storage node (Client).

### Master Service

The cluster's available resources are viewed as a large resource pool, managed centrally by a Master process for space allocation and guiding data replication 

**Note: The Master Service does not take over any data flow, only providing corresponding metadata information.**

#### Master Service APIs

The protobuf definition between Master and Client is as follows:

```protobuf
message BufHandle {
  required uint64 segment_name = 1;  // Storage segment name (can be simply understood as the name of the storage node)
  required uint64 size = 2;          // Size of the allocated space
  required uint64 buffer = 3;        // Pointer to the allocated space

  enum BufStatus {
    INIT = 0;          // Initial state, space reserved but not used
    COMPLETE = 1;      // Completed usage, space contains valid data
    FAILED = 2;        // Usage failed, upstream should update the handle state to this value
    UNREGISTERED = 3;  // Space has been unregistered, metadata deleted
  }
  required BufStatus status = 4 [default = INIT]; // Space status
};

message ReplicaInfo {
  repeated BufHandle handles = 1; // Specific locations of the stored object data

  enum ReplicaStatus {
    UNDEFINED = 0;   // Uninitialized
    INITIALIZED = 1; // Space allocated, waiting for write
    PROCESSING = 2;  // Writing data in progress
    COMPLETE = 3;    // Write completed, replica available
    REMOVED = 4;     // Replica has been removed
    FAILED = 5;      // Replica write failed, consider reallocation
  }
  required ReplicaStatus status = 2 [default = UNDEFINED]; // Replica status
};

service MasterService {
  // Get the list of replicas for an object
  rpc GetReplicaList(GetReplicaListRequest) returns (GetReplicaListResponse);

  // Start Put operation, allocate storage space
  rpc PutStart(PutStartRequest) returns (PutStartResponse);

  // End Put operation, mark object write completion
  rpc PutEnd(PutEndRequest) returns (PutEndResponse);

  // Delete all replicas of an object
  rpc Remove(RemoveRequest) returns (RemoveResponse);

  // Storage node (Client) registers a storage segment
  rpc MountSegment(MountSegmentRequest) returns (MountSegmentResponse);

  // Storage node (Client) unregisters a storage segment
  rpc UnmountSegment(UnmountSegmentRequest) returns (UnmountSegmentResponse);
}
```

1. GetReplicaList

```protobuf
message GetReplicaListRequest {
  required string key = 1; 
};

message GetReplicaListResponse {
  required int32 status_code = 1;
  repeated ReplicaInfo replica_list = 2; // List of replica information
};
```

- **Request**: `GetReplicaListRequest` containing the key to query.
- **Response**: `GetReplicaListResponse` containing the status code status_code and the list of replica information `replica_list`.
- **Description**: Used to retrieve information about all available replicas for a specified key. The Client can select an appropriate replica for reading based on this information.

2. PutStart

```protobuf
message PutStartRequest {
  required string key = 1;             // Object key
  required int64 value_length = 2;     // Total length of data to be written
  required ReplicateConfig config = 3; // Replica configuration information
  repeated uint64 slice_lengths = 4;   // Lengths of each data slice
};

message PutStartResponse {
  required int32 status_code = 1; 
  repeated ReplicaInfo replica_list = 2;  // Replica information allocated by the Master Service
};
```

- **Request**: `PutStartRequest` containing the key, data length, and replica configuration config.
- **Response**: `PutStartResponse` containing the status code status_code and the allocated replica information replica_list.
- **Description**: Before writing an object, the Client must call PutStart to request storage space from the Master Service. The Master Service allocates space based on the config and returns the allocation results (`replica_list`) to the Client. The Client then writes data to the storage nodes where the allocated replicas are located. The need for both start and end steps ensures that other Clients do not read partially written values, preventing dirty reads.

3. PutEnd

```protobuf
message PutEndRequest {
  required string key = 1; 
};

message PutEndResponse {
  required int32 status_code = 1;
};
```

- **Request**: `PutEndRequest` containing the key.
- **Response**: `PutEndResponse` containing the status code status_code.
- **Description**: After the Client completes data writing, it calls `PutEnd` to notify the Master Service. The Master Service updates the object's metadata, marking the replica status as `COMPLETE`, indicating that the object is readable.

4. Remove

```protobuf
message RemoveRequest {
  required string key = 1; 
};

message RemoveResponse {
  required int32 status_code = 1;
};
```

- **Request**: `RemoveRequest` containing the key of the object to be deleted.
- **Response**: `RemoveResponse` containing the status code `status_code`.
- **Description**: Used to delete the object and all its replicas corresponding to the specified key. The Master Service marks all replicas of the corresponding object as deleted.

5. MountSegment

```protobuf
message MountSegmentRequest {
  required uint64 buffer = 1;       // Starting address of the space
  required uint64 size = 2;         // Size of the space
  required string segment_name = 3; // Storage segment name
}

message MountSegmentResponse {
  required int32 status_code = 1;
};
```

The storage node (Client) allocates a segment of memory and, after calling `TransferEngine::registerLocalMemory` to complete local mounting, calls this interface to mount the allocated continuous address space to the Master Service for allocation.

6. UnmountSegment

```protobuf
message UnmountSegmentRequest {
  required string segment_name = 1;  // Storage segment name used during mounting
}

message UnMountSegmentResponse {
  required int32 status_code = 1;
};
```

When the space needs to be released, this interface is used to remove the previously mounted resources from the Master Service.

#### Object Information Maintenance

The Master Service needs to maintain mappings related to buffer allocators and object metadata to efficiently manage memory resources and precisely control replica states in multi-replica scenarios. Additionally, the Master Service uses read-write locks to protect critical data structures, ensuring data consistency and security in multi-threaded environments. The following are the interfaces maintained by the Master Service for storage space information:

- MountSegment

```C++
tl::expected<void, ErrorCode> MountSegment(uint64_t buffer,
                                          uint64_t size,
                                          const std::string& segment_name);
```

The storage node (Client) registers the storage segment space with the Master Service.

- UnmountSegment

```C++
tl::expected<void, ErrorCode> UnmountSegment(const std::string& segment_name);
```

The storage node (Client) unregisters the storage segment space with the Master Service.

The Master Service handles object-related interfaces as follows:

- Put

```C++
    ErrorCode PutStart(const std::string& key,
                       uint64_t value_length,
                       const std::vector<uint64_t>& slice_lengths,
                       const ReplicateConfig& config,
                       std::vector<ReplicaInfo>& replica_list);

    ErrorCode PutEnd(const std::string& key);
```

Before writing an object, the Client calls PutStart to request storage space allocation from the Master Service. After completing data writing, the Client calls PutEnd to notify the Master Service to mark the object write as completed.

- GetReplicaList

```C++
ErrorCode GetReplicaList(const std::string& key,
                         std::vector<ReplicaInfo>& replica_list);
```

The Client requests the Master Service to retrieve the replica list for a specified key, allowing the Client to select an appropriate replica for reading based on this information.

- Remove

```C++
tl::expected<void, ErrorCode> Remove(const std::string& key);
```

The Client requests the Master Service to delete all replicas corresponding to the specified key.

### Buffer Allocator

The buffer allocator serves as a low-level memory management component within the Mooncake Store system, primarily responsible for efficient memory allocation and deallocation. It builds upon underlying memory allocators to perform its functions.

Importantly, the memory managed by the buffer allocator does not reside within the `Master Service` itself. Instead, it operates on memory segments registered by `Clients`. When the `Master Service` receives a `MountSegment` request to register a contiguous memory region, it creates a corresponding buffer allocator via the `AddSegment` interface.

Mooncake Store provides two concrete implementations of `BufferAllocatorBase`:

**CachelibBufferAllocator**: This allocator leverages Facebook's [CacheLib](https://github.com/facebook/CacheLib) to manage memory using a slab-based allocation strategy. It provides efficient memory allocation with good fragmentation resistance and is well-suited for high-performance scenarios.

**OffsetBufferAllocator**: This allocator is derived from [OffsetAllocator](https://github.com/sebbbi/OffsetAllocator), which uses a custom bin-based allocation strategy that supports fast hard realtime `O(1)` offset allocation with minimal fragmentation.

Mooncake Store optimizes both allocators based on the specific memory usage characteristics of LLM inference workloads, thereby enhancing memory utilization in LLM scenarios. The allocators can be used interchangeably based on specific performance requirements and memory usage patterns. This is configurable via the startup parameter `--memory-allocator` of `master_service`.

Both allocators implement the same interface as `BufferAllocatorBase`. The main interfaces of the `BufferAllocatorBase` class are as follows:

```C++
class BufferAllocatorBase {
    virtual ~BufferAllocatorBase() = default;
    virtual std::unique_ptr<AllocatedBuffer> allocate(size_t size) = 0;
    virtual void deallocate(AllocatedBuffer* handle) = 0;
};
```

1. **Constructor**: When a `BufferAllocator` instance is created, the upstream component must provide the base address and size of the memory region to be managed. This information is used to initialize the internal allocator, enabling unified memory management.

2. **`allocate` Function**: When the upstream issues read or write requests, it needs a memory region to operate on. The `allocate` function invokes the internal allocator to reserve a memory block and returns metadata such as the starting address and size. The status of the newly allocated memory is initialized as `BufStatus::INIT`.

3. **`deallocate` Function**: This function is automatically triggered by the `BufHandle` destructor. It calls the internal allocator to release the associated memory and updates the handle’s status to `BufStatus::UNREGISTERED`.

### AllocationStrategy
AllocationStrategy is a strategy class for efficiently managing memory resource allocation and replica storage location selection in a distributed environment. It is mainly used in the following scenarios:
- Determining the allocation locations for object storage replicas.
- Selecting suitable read/write paths among multiple replicas.
- Providing decision support for resource load balancing between nodes in distributed storage.

AllocationStrategy is used in conjunction with the Master Service and the underlying buffer allocator:
- Master Service: Determines the target locations for replica allocation via `AllocationStrategy`.
- Buffer Allocator: Executes the actual memory allocation and release tasks.

#### APIs

`Allocate`: Finds a suitable storage segment from available storage resources to allocate space of a specified size.

```C++
virtual std::unique_ptr<AllocatedBuffer> Allocate(
        const std::vector<std::shared_ptr<BufferAllocatorBase>>& allocators,
        const std::unordered_map<std::string, std::vector<std::shared_ptr<BufferAllocatorBase>>>& allocators_by_name,
        size_t objectSize, const ReplicateConfig& config) = 0;
```

- **Input Parameters**:
  - `allocators`: A vector of all mounted buffer allocators
  - `allocators_by_name`: A map of allocators organized by segment name for preferred segment allocation
  - `objectSize`: The size of the object to be allocated
  - `config`: Replica configuration including preferred segment and other allocation preferences
- **Output**: Returns a unique pointer to an `AllocatedBuffer` if allocation succeeds, or `nullptr` if no suitable allocator is found

#### Implementation Strategies

`RandomAllocationStrategy` is a subclass implementing `AllocationStrategy` that provides intelligent allocation with the following features:

1. **Preferred Segment Support**: If a preferred segment is specified in the `ReplicateConfig`, the strategy first attempts to allocate from that segment before falling back to random allocation.

2. **Random Allocation with Retry Logic**: When multiple allocators are available, it uses a randomized approach with up to 10 retry attempts to find a suitable allocator.

3. **Deterministic Randomization**: Uses a Mersenne Twister random number generator with proper seeding for consistent behavior.

The strategy automatically handles cases where the preferred segment is unavailable, full, or doesn't exist by gracefully falling back to random allocation among all available segments.

### Eviction Policy

When the mounted segments are full, i.e., when a `PutStart` request fails due to insufficient memory, an eviction task will be launched to free up space by evicting some objects. Just like `Remove`, evicted objects are simply marked as deleted. No data transfer is needed.

Currently, an approximate LRU policy is adopted, where the least recently used objects are preferred for eviction. To avoid data races and corruption, objects currently being read or written by clients should not be evicted. For this reason, objects that have leases or have not been marked as complete by `PutEnd` requests will be ignored by the eviction task.

Each time the eviction task is triggered, in default it will try to evict about 10% of objects. This ratio is configurable via a startup parameter of `master_service`.

To minimize put failures, you can set the eviction high watermark via the `master_service` startup parameter `-eviction_high_watermark_ratio=<RATIO>`(Default to 1). When the eviction thread detects that current space usage reaches the configured high watermark,
it initiates evict operations. The eviction target is to clean an additional `-eviction_ratio` specified proportion beyond the high watermark, thereby reaching the space low watermark.

### Lease

To avoid data conflicts, a per-object lease will be granted whenever an `ExistKey` request or a `GetReplicaListRequest` request succeeds. An object is guaranteed to be protected from `Remove` request, `RemoveAll` request and `Eviction` task until its lease expires. A `Remove` request on a leased object will fail. A `RemoveAll` request will only remove objects without a lease.

The default lease TTL is 5 seconds and is configurable via a startup parameter of `master_service`.

### Soft Pin

For important and frequently used objects, such as system prompts, Mooncake Store provides a soft pin mechanism. When putting an object, it can be configured to enable soft pin. During eviction, objects that are not soft pinned are prioritized for eviction. Soft pinned objects are only evicted when memory is insufficient and no other objects are eligible for eviction.

If a soft pinned object is not accessed for an extended period, its soft pin status will be removed. If it is accessed again later, it will automatically be soft pinned once more.

There are two startup parameters in `master_service` related to the soft pin mechanism:

- `default_kv_soft_pin_ttl`: The duration (in milliseconds) after which a soft pinned object will have its soft pin status removed if not accessed. The default value is `30 minutes`.

- `allow_evict_soft_pinned_objects`: Whether soft pinned objects are allowed to be evicted. The default value is `true`.

Notably, soft pinned objects can still be removed using APIs such as `Remove` or `RemoveAll`.

### Preferred Segment Allocation

Mooncake Store provides a **preferred segment allocation** feature that allows users to specify a preferred storage segment (node) for object allocation. This feature is particularly useful for optimizing data locality and reducing network overhead in distributed scenarios.

#### How It Works

The preferred segment allocation feature is implemented through the `AllocationStrategy` system and is controlled via the `preferred_segment` field in the `ReplicateConfig` structure:

```cpp
struct ReplicateConfig {
    size_t replica_num{1};                    // Total number of replicas for the object
    bool with_soft_pin{false};               // Whether to enable soft pin mechanism for this object
    std::string preferred_segment{};         // Preferred segment for allocation
};
```

When a `Put` operation is initiated with a non-empty `preferred_segment` value, the allocation strategy follows this process:

1. **Preferred Allocation Attempt**: The system first attempts to allocate space from the specified preferred segment. If the preferred segment has sufficient available space, the allocation succeeds immediately.

2. **Fallback to Random Allocation**: If the preferred segment is unavailable, full, or doesn't exist, the system automatically falls back to the standard random allocation strategy among all available segments.

3. **Retry Logic**: The allocation strategy includes built-in retry mechanisms with up to 10 attempts to find suitable storage space across different segments.

- **Data Locality**: By preferring local segments, applications can reduce network traffic and improve access performance for frequently used data.
- **Load Balancing**: Applications can distribute data across specific nodes to achieve better load distribution.

### Multi-layer Storage Support

This system provides support for a hierarchical cache architecture, enabling efficient data access through a combination of in-memory caching and persistent storage. Data is initially stored in memory cache and asynchronously backed up to a Distributed File System (DFS), forming a two-tier "memory-SSD persistent storage" cache structure.

#### Enabling Persistence Functionality

When a user specifies the environment variable `MOONCAKE_STORAGE_ROOT_DIR` at client startup, and the path is a valid existing directory, the client-side data persistence feature will be activated. During initialization, the client requests a `cluster_id` from the master. This ID can be specified when initializing the master; if not provided, the default value `mooncake_cluster` will be used. The root directory for persistence is then set to `<MOONCAKE_STORAGE_ROOT_DIR>/<cluster_id>`. Note that when using DFS, each client must specify the corresponding DFS mount directory to enable data sharing across SSDs.

#### Data Access Mechanism

In the current implementation, all operations on kvcache objects (e.g., read/write/query) are performed entirely on the client side, with no awareness by the master. The file system maintains the key-to-kvcache-object mapping through a fixed indexing mechanism, where each file corresponds to one kvcache object (the filename is the associated key).

When persistence is enabled, every successful `Put`or`BatchPut` operation in memory triggers an asynchronous persistence write to DFS. During subsequent `Get`or `BatchGet` operations, if the requested kvcache is not found in the memory pool, the system attempts to read the corresponding file from DFS and returns the data to the user.

## Mooncake Store Python API

**Complete Python API Documentation**: [https://kvcache-ai.github.io/Mooncake/mooncake-store-api/python-binding.html](https://kvcache-ai.github.io/Mooncake/mooncake-store-api/python-binding.html)

## Compilation and Usage
Mooncake Store is compiled together with other related components (such as the Transfer Engine).

For default mode:
```
mkdir build && cd build
cmake .. # default mode
make
sudo make install # Install Python interface support package
```

High availability mode:
```
mkdir build && cd build
cmake .. -DSTORE_USE_ETCD # compile etcd wrapper that depends on go
make
sudo make install # Install Python interface support package
```

### Starting the Transfer Engine's Metadata Service
Mooncake Store uses the Transfer Engine as its core transfer engine, so it is necessary to start the metadata service (etcd/redis/http). The startup and configuration of the `metadata` service can be referred to in the relevant sections of [Transfer Engine](./transfer-engine.md). **Special Note**: For the etcd service, by default, it only provides services for local processes. You need to modify the listening options (IP to 0.0.0.0 instead of the default 127.0.0.1). You can use commands like curl to verify correctness.

### Starting the Master Service
The Master Service runs as an independent process, provides gRPC interfaces externally, and is responsible for the metadata management of Mooncake Store (note that the Master Service does not reuse the metadata service of the Transfer Engine). The default listening port is `50051`. After compilation, you can directly run `mooncake_master` located in the `build/mooncake-store/src/` directory. After starting, the Master Service will output the following content in the log:
```
Starting Mooncake Master Service
Port: 50051
Max threads: 4
Master service listening on 0.0.0.0:50051
```

**High availability mode**:

HA mode relies on an etcd service for coordination. If Transfer Engine also uses etcd as its metadata service, the etcd cluster used by Mooncake Store can either be shared with or separate from the one used by Transfer Engine.

HA mode allows deployment of multiple master instances to eliminate the single point of failure. Each master instance must be started with the following parameters:
```
--enable-ha: enables high availability mode
--etcd-endpoints: specifies endpoints for etcd service, separated by ';'
--rpc-address: the RPC address of this instance
```

For example:
```
./build/mooncake-store/src/mooncake_master \
    --enable-ha=true \
    --etcd-endpoints="0.0.0.0:2379;0.0.0.0:2479;0.0.0.0:2579" \
    --rpc-address=10.0.0.1
```

### Starting the Sample Program
Mooncake Store provides various sample programs, including interface forms based on C++ and Python. Below is an example of how to run using `stress_cluster_benchmark`.

1. Open `stress_cluster_benchmark.py` and update the initialization settings based on your network environment. Pay particular attention to the following fields:
`local_hostname`: the IP address of the local machine
`metadata_server`: the address of the Transfer Engine metadata service
`master_server_address`: the address of the Master Service
**Note**: The format of `master_server_address` depends on the deployment mode. In default mode, use the format `IP:Port`, specifying the address of a single master node. In HA mode, use the format `etcd://IP:Port;IP:Port;...;IP:Port`, specifying the addresses of the etcd cluster endpoints.
For example: 
```python
import os
import time

from distributed_object_store import DistributedObjectStore

store = DistributedObjectStore()
# Protocol used by the transfer engine, optional values are "rdma" or "tcp"
protocol = os.getenv("PROTOCOL", "tcp")
# Device name used by the transfer engine
device_name = os.getenv("DEVICE_NAME", "ibp6s0")
# Hostname of this node in the cluster, port number is randomly selected from (12300-14300)
local_hostname = os.getenv("LOCAL_HOSTNAME", "localhost")
# Metadata service address of the Transfer Engine, here etcd is used as the metadata service
metadata_server = os.getenv("METADATA_ADDR", "127.0.0.1:2379")
# The size of the Segment mounted by each node to the cluster, allocated by the Master Service after mounting, in bytes
global_segment_size = 3200 * 1024 * 1024
# Local buffer size registered with the Transfer Engine, in bytes
local_buffer_size = 512 * 1024 * 1024
# Address of the Master Service of Mooncake Store
master_server_address = os.getenv("MASTER_SERVER", "127.0.0.1:50051")
# Data length for each put()
value_length = 1 * 1024 * 1024
# Total number of requests sent
max_requests = 1000
# Initialize Mooncake Store Client
retcode = store.setup(
    local_hostname,
    metadata_server,
    global_segment_size,
    local_buffer_size,
    protocol,
    device_name,
    master_server_address,
)
```

2. Run `ROLE=prefill python3 ./stress_cluster_benchmark.py` on one machine to start the Prefill node.
   For "rdma" protocol, you can also enable topology auto discovery and filters, e.g., `ROLE=prefill MC_MS_AUTO_DISC=1 MC_MS_FILTERS="mlx5_1,mlx5_2" python3 ./stress_cluster_benchmark.py`.
   To enable the persistence feature, run:
`ROLE=prefill MOONCAKE_STORAGE_ROOT_DIR=/path/to/dir python3 ./stress_cluster_benchmark.py`

3. Run `ROLE=decode python3 ./stress_cluster_benchmark.py` on another machine to start the Decode node.
   For "rdma" protocol, you can also enable topology auto discovery and filters, e.g., `ROLE=decode MC_MS_AUTO_DISC=1 MC_MS_FILTERS="mlx5_1,mlx5_2" python3 ./stress_cluster_benchmark.py`.
   To enable the persistence feature, run:
`ROLE=decode MOONCAKE_STORAGE_ROOT_DIR=/path/to/dir python3 ./stress_cluster_benchmark.py`

The absence of error messages indicates successful data transfer.

### Starting the Client as Standalone Process

Use `mooncake-wheel/mooncake/mooncake_store_service.py` to start the `Client` as a standalone process.

First, create and save a configuration file in JSON format. For example:

```
{
    "local_hostname": "localhost",
    "metadata_server": "http://localhost:8080/metadata",
    "global_segment_size": 268435456,
    "local_buffer_size": 268435456,
    "protocol": "tcp",
    "device_name": "",
    "master_server_address": "localhost:50051"
}
```

Then run `mooncake_store_service.py`. This program starts an HTTP server alongside the `Client`. Through this server, users can manually perform operations such as `Get` and `Put`, which is useful for debugging.

The main startup parameters include:

* `config`: Path to the configuration file.
* `port`: Port number for the HTTP server.

Suppose the `mooncake_transfer_engine` wheel package is already installed, the following command starts the program:
```bash
python -m mooncake.mooncake_store_service --config=[config_path] --port=8081
```

## Example Code

#### Python Usage Example
We provide a reference example `distributed_object_store_provider.py`, located in the `mooncake-store/tests` directory. To check if the related components are properly installed, you can run etcd and Master Service (`mooncake_master`) in the background on the same server, and then execute this Python program in the foreground. It should output a successful test result.

#### C++ Usage Example
The C++ API of Mooncake Store provides more low-level control capabilities. We provide a reference example `client_integration_test`, located in the `mooncake-store/tests` directory. To check if the related components are properly installed, you can run etcd and Master Service (`mooncake_master`) on the same server, and then execute this C++ program (located in the `build/mooncake-store/tests` directory). It should output a successful test result.
