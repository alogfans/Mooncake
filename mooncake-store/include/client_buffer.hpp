#pragma once

#include <optional>
#include <vector>

#include "offset_allocator/offset_allocator.hpp"
#include "types.h"

namespace mooncake {

class BufferHandle;

/**
 * ClientBufferAllocator manages a contiguous memory buffer using an
 * offset-based allocation strategy. It provides thread-safe allocation and
 * automatic deallocation through RAII handles.
 *
 * This allocator is designed for client-side memory management where you need:
 * - Efficient sub-allocation within a larger buffer
 * - Automatic memory cleanup via RAII
 * - Thread-safe allocation operations
 */
class ClientBufferAllocator
    : public std::enable_shared_from_this<ClientBufferAllocator> {
   public:
    static std::shared_ptr<ClientBufferAllocator> create(size_t size) {
        return std::shared_ptr<ClientBufferAllocator>(
            new ClientBufferAllocator(size));
    }

    ~ClientBufferAllocator();

    // Disable copy constructor and copy assignment operator
    ClientBufferAllocator(const ClientBufferAllocator&) = delete;
    ClientBufferAllocator& operator=(const ClientBufferAllocator&) = delete;

    // Disable move constructor and move assignment operator
    ClientBufferAllocator(ClientBufferAllocator&&) = delete;
    ClientBufferAllocator& operator=(ClientBufferAllocator&&) = delete;

    [[nodiscard]] void* getBase() const { return buffer_; }

    [[nodiscard]] size_t size() const { return buffer_size_; }

    [[nodiscard]] std::optional<BufferHandle> allocate(size_t size);

   private:
    ClientBufferAllocator(size_t size);
    std::shared_ptr<offset_allocator::OffsetAllocator> allocator_;

    void* buffer_;
    size_t buffer_size_;
};

/**
 * BufferHandle provides RAII management for allocated memory regions within
 * a ClientBufferAllocator. It automatically deallocates the memory when
 * destroyed and provides access to the allocated memory pointer and size.
 *
 * This class is move-only to ensure proper ownership semantics and prevent
 * accidental double-free errors.
 */
class BufferHandle {
   public:
    BufferHandle(std::shared_ptr<ClientBufferAllocator> allocator,
                 offset_allocator::OffsetAllocationHandle handle);
    ~BufferHandle();

    BufferHandle(BufferHandle&&) = default;  // Allow move operations
    BufferHandle& operator=(BufferHandle&&) = default;

    // Disable copy constructor and copy assignment operator
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;

    [[nodiscard]] void* ptr() const;
    [[nodiscard]] size_t size() const;

   private:
    std::shared_ptr<ClientBufferAllocator> allocator_;
    offset_allocator::OffsetAllocationHandle handle_;
};

// Utility functions for buffer and slice management
/**
 * @brief Split a BufferHandle into slices of maximum size kMaxSliceSize
 * @param handle The buffer handle to split
 * @return Vector of slices covering the entire buffer
 */
std::vector<Slice> split_into_slices(BufferHandle& handle);

/**
 * @brief Calculate the total size of a replica descriptor
 * @param replica The replica descriptor to calculate size for
 * @return Total size in bytes
 */
uint64_t calculate_total_size(const Replica::Descriptor& replica);

/**
 * @brief Allocate slices from a buffer handle based on replica descriptor
 * @param slices Output vector to store the allocated slices
 * @param replica The replica descriptor defining the slice structure
 * @param buffer_handle The buffer handle to allocate slices from
 * @return 0 on success, non-zero on error
 */
int allocateSlices(std::vector<Slice>& slices,
                   const Replica::Descriptor& replica,
                   BufferHandle& buffer_handle);

}  // namespace mooncake