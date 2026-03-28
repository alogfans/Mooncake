// Copyright 2025 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file transport_loader.cpp
 * @brief Platform-independent transport plugin loader
 *
 * This file contains NO platform-specific code. All transports are
 * loaded as plugins at runtime.
 */

#include "tent/runtime/transfer_engine_impl.h"
#include "tent/transport/shm/shm_transport.h"
#include "tent/transport/tcp/tcp_transport.h"

#include <dlfcn.h>
#include <glob.h>
#include <glog/logging.h>

namespace mooncake {
namespace tent {

// Function pointer type for transport creation
using CreateTransportFunc = std::shared_ptr<Transport> (*)();

// Loaded plugin handles (for cleanup)
struct PluginHandle {
    void* handle = nullptr;
    CreateTransportFunc create_func = nullptr;
};

static std::unordered_map<TransportType, PluginHandle> g_loaded_plugins;

// Platform detection - returns first available platform
static std::string detectPlatform() {
    // Detection order: CUDA -> MUSA -> HIP -> Ascend -> CPU
    static const char* libraries[] = {"libcuda.so", "libmusa.so",
                                      "libamdhip64.so", "libascendcl.so",
                                      nullptr};

    static const char* platforms[] = {"cuda", "musa", "hip", "ascend", "cpu"};

    for (int i = 0; libraries[i] != nullptr; ++i) {
        void* handle = dlopen(libraries[i], RTLD_NOW | RTLD_NOLOAD);
        if (handle) {
            dlclose(handle);
            return platforms[i];
        }
    }
    return "cpu";
}

// Plugin search paths
static const char* PLUGIN_SEARCH_PATHS[] = {"/usr/local/lib/tent/transport",
                                            "/usr/lib/tent/transport",
                                            "/opt/tent/lib/transport",
                                            "./lib/tent/transport",
                                            "./lib",
                                            nullptr};

/**
 * @brief Try to find and load a platform-specific plugin
 */
static std::shared_ptr<Transport> tryLoadPlatformPlugin(
    const std::string& base_name, TransportType type, bool optional = true) {
    // Check if already loaded
    auto it = g_loaded_plugins.find(type);
    if (it != g_loaded_plugins.end() && it->second.create_func) {
        return it->second.create_func();
    }

    std::string platform = detectPlatform();
    LOG(INFO) << "Detected platform: " << platform << " for transport "
              << base_name;

    // List of library names to try, in priority order
    std::vector<std::string> lib_names = {
        "libtent_" + base_name + "_" + platform + ".so",  // Platform-specific
        "libtent_" + base_name + ".so",                   // Generic fallback
    };

    for (const auto& lib_name : lib_names) {
        // Try direct path first
        void* handle = dlopen(lib_name.c_str(), RTLD_NOW | RTLD_LOCAL);

        // If not found, search in plugin directories
        if (!handle) {
            for (int i = 0; PLUGIN_SEARCH_PATHS[i] != nullptr; ++i) {
                std::string full_path =
                    std::string(PLUGIN_SEARCH_PATHS[i]) + "/" + lib_name;
                handle = dlopen(full_path.c_str(), RTLD_NOW | RTLD_LOCAL);
                if (handle) {
                    LOG(INFO) << "Found plugin at: " << full_path;
                    break;
                }
            }
        }

        if (!handle) {
            continue;  // Try next library name
        }

        // Build symbol name: "CreateRdmaTransport", "CreateNvlinkTransport",
        // etc.
        std::string symbol_name = "Create" + base_name;
        // Capitalize first letter
        if (!symbol_name.empty()) {
            symbol_name[0] = toupper(symbol_name[0]);
        }
        symbol_name += "Transport";

        // Find the factory function
        auto create_func = reinterpret_cast<CreateTransportFunc>(
            dlsym(handle, symbol_name.c_str()));

        if (!create_func) {
            dlclose(handle);
            LOG(WARNING) << "Plugin " << lib_name << " missing symbol "
                         << symbol_name;
            continue;  // Try next library name
        }

        // Success! Store the plugin handle
        PluginHandle plugin;
        plugin.handle = handle;
        plugin.create_func = create_func;
        g_loaded_plugins[type] = plugin;

        LOG(INFO) << "Loaded transport plugin: " << lib_name
                  << " (symbol=" << symbol_name << ")";

        return create_func();
    }

    // All attempts failed
    if (optional) {
        LOG(INFO) << "Transport " << base_name
                  << " not available as plugin (tried platform=" << platform
                  << ")";
    } else {
        LOG(WARNING) << "Failed to load transport " << base_name;
    }
    return nullptr;
}

/**
 * @brief Load all transports as plugins
 *
 * All transports are loaded as plugins. No static linking fallback.
 * This ensures the core library is platform-independent.
 */
Status TransferEngineImpl::loadTransports() {
    // Built-in transports (platform-independent, always available)
    if (conf_->get("transports/tcp/enable", true))
        transport_list_[TCP] = std::make_shared<TcpTransport>();

    if (conf_->get("transports/shm/enable", false))
        transport_list_[SHM] = std::make_shared<ShmTransport>();

    // Load all optional transports as plugins
    // These are platform-specific and loaded at runtime

    // RDMA transport
    if (conf_->get("transports/rdma/enable", true)) {
        auto rdma = tryLoadPlatformPlugin("rdma", RDMA, true);
        if (rdma) {
            transport_list_[RDMA] = rdma;
        }
    }

    // IOUring transport
    if (conf_->get("transports/io_uring/enable", true)) {
        auto iouring = tryLoadPlatformPlugin("iouring", IOURING, true);
        if (iouring) {
            transport_list_[IOURING] = iouring;
        }
    }

    // NVLink/MNNVL transport
    bool enable_mnnvl = getenv("MC_ENABLE_MNNVL") != nullptr;
    if (enable_mnnvl) {
        auto mnnvl = tryLoadPlatformPlugin("mnnvl", MNNVL, true);
        if (mnnvl) {
            transport_list_[MNNVL] = mnnvl;
        }
    } else {
        auto nvlink = tryLoadPlatformPlugin("nvlink", NVLINK, true);
        if (nvlink) {
            transport_list_[NVLINK] = nvlink;
        }
    }

    // GDS transport
    if (conf_->get("transports/gds/enable", false)) {
        auto gds = tryLoadPlatformPlugin("gds", GDS, true);
        if (gds) {
            transport_list_[GDS] = gds;
        }
    }

    // AscendDirect transport
    if (conf_->get("transports/ascend_direct/enable", true)) {
        auto ascend = tryLoadPlatformPlugin("ascend", AscendDirect, true);
        if (ascend) {
            transport_list_[AscendDirect] = ascend;
        }
    }

    return Status::OK();
}

/**
 * @brief Cleanup loaded plugins on shutdown
 */
void TransferEngineImpl::unloadPlugins() {
    for (auto& kv : g_loaded_plugins) {
        if (kv.second.handle) {
            dlclose(kv.second.handle);
        }
    }
    g_loaded_plugins.clear();
}

}  // namespace tent
}  // namespace mooncake
