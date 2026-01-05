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

#include "tent/transport/rdma/quota.h"
#include "tent/transport/rdma/shared_quota.h"
#include "tent/common/utils/random.h"

#include <assert.h>
#include <unordered_set>

namespace mooncake {
namespace tent {

static inline double ClampFast(double x, double lo, double hi) {
    if (std::isnan(x)) return lo;  // avoid NaN poisoning
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline double SafeMax(double a, double b) { return (a > b) ? a : b; }

Status DeviceQuota::loadTopology(std::shared_ptr<Topology>& local_topology) {
    local_topology_ = local_topology;
    std::unordered_set<int> used_numa_id;
    for (size_t dev_id = 0; dev_id < local_topology->getNicCount(); ++dev_id) {
        auto entry = local_topology->getNicEntry(dev_id);
        if (entry->type != Topology::NIC_RDMA) continue;
        DeviceInfo& info = devices_[dev_id];
        info.dev_id = dev_id;
        info.bw_gbps = 200.0;
        info.numa_id = entry->numa_node;
        used_numa_id.insert(entry->numa_node);
    }
    if (used_numa_id.size() == 1) allow_cross_numa_ = true;
    return Status::OK();
}

Status DeviceQuota::enableSharedQuota(const std::string& shm_name) {
    shared_quota_ = std::make_shared<SharedQuotaManager>(this);
    auto status = shared_quota_->attach(shm_name);
    if (!status.ok()) shared_quota_.reset();
    return status;
}

// Per-thread learning state to avoid contention.
// We also keep EWMA/samples here to stabilize updates without header changes.
struct TlsDeviceInfo {
    uint64_t active_bytes{0};

    // Model params: pred = beta0 + beta1 * theory
    double beta0{0.0};
    double beta1{1.0};

    // Stabilizers
    double obs_ewma{0.0};  // EWMA of observed latency
    uint64_t samples{0};   // number of release samples observed (per thread)
    uint64_t last_picked_ts{0};  // for starvation protection (per thread)
};

thread_local std::unordered_map<int, TlsDeviceInfo> tl_device_info;

Status DeviceQuota::allocate(uint64_t length, const std::string& location,
                             int& chosen_dev_id) {
    auto entry = local_topology_->getMemEntry(location);
    if (!entry) return Status::InvalidArgument("Unknown location" LOC_MARK);

    // Original non-quota behavior: round-robin among eligible devices.
    if (!enable_quota_) {
        thread_local int id = 0;
        for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
            auto& list = entry->device_list[rank];
            if (list.empty()) continue;
            chosen_dev_id = list[id % list.size()];
            id++;
            return Status::OK();
        }
        return Status::DeviceNotFound("no eligible devices for " + location);
    }

    // -----------------------------
    // Selection robustness improvements:
    // 1) Top-K fallback (avoid overly sharp best_score*1.05 pruning)
    // 2) epsilon-greedy exploration
    // 3) starvation protection (force candidates if not chosen for a while)
    // -----------------------------
    // NOTE: tuned to be conservative. You can later expose them via
    // gflags/config.
    static constexpr double kPenalty[] = {1.0, 3.0, 10.0};
    static constexpr double kNearBestRatio = 1.05;   // original
    static constexpr size_t kTopKMin = 2;            // keep at least top-2
    static constexpr double kExploreEpsilon = 0.01;  // 1% exploration
    static constexpr uint64_t kStarveNs = 5ull * 1000 * 1000 * 1000;  // 5s

    const double w = local_weight_;

    std::unordered_map<int, double> score_map;
    bool found_device = false;

    // Keep a sorted vector for Top-K.
    std::vector<std::pair<int, double>> scored_list;
    scored_list.reserve(16);

    double best_score = std::numeric_limits<double>::infinity();
    for (size_t rank = 0; rank < Topology::DevicePriorityRanks; ++rank) {
        if (rank == Topology::DevicePriorityRanks - 1 && !allow_cross_numa_ &&
            found_device) {
            continue;
        }

        for (int dev_id : entry->device_list[rank]) {
            if (!devices_.count(dev_id)) continue;
            auto& dev = devices_[dev_id];
            auto& tl_dev = tl_device_info[dev_id];

            uint64_t overall_active_bytes =
                dev.diffusion_active_bytes.load(std::memory_order_relaxed) +
                dev.active_bytes.load(std::memory_order_relaxed);

            double weighted_active =
                w * static_cast<double>(tl_dev.active_bytes) +
                (1.0 - w) * static_cast<double>(overall_active_bytes) +
                static_cast<double>(length);

            double beta0_g = dev.beta0.load(std::memory_order_relaxed);
            double beta1_g = dev.beta1.load(std::memory_order_relaxed);
            double beta0 = w * tl_dev.beta0 + (1.0 - w) * beta0_g;
            double beta1 = w * tl_dev.beta1 + (1.0 - w) * beta1_g;

            // Numeric guard: avoid NaN/inf poisoning score.
            if (!std::isfinite(beta0)) beta0 = 0.0;
            if (!std::isfinite(beta1) || beta1 <= 0.0) beta1 = 1.0;

            double bw = dev.bw_gbps * 1e9 / 8;
            if (!(bw > 0.0) || !std::isfinite(bw)) bw = 1.0;

            double predicted_time = (weighted_active / bw) * beta1 + beta0;
            if (!std::isfinite(predicted_time) || predicted_time < 0.0)
                predicted_time = std::numeric_limits<double>::infinity();

            double score = kPenalty[rank] * predicted_time;
            score_map[dev_id] = score;

            scored_list.emplace_back(dev_id, score);
            best_score = std::min(best_score, score);
            found_device = true;
        }
    }

    if (!found_device) {
        return Status::DeviceNotFound("no eligible devices for " + location);
    }

    // Sort by score then dev_id for determinism.
    std::sort(scored_list.begin(), scored_list.end(),
              [&](const auto& a, const auto& b) {
                  if (std::abs(a.second - b.second) > 1e-12)
                      return a.second < b.second;
                  return a.first < b.first;
              });

    // Build filtered set:
    // - keep all within near-best ratio
    // - plus Top-K minimum
    // - plus starvation-protected devices
    std::vector<int> filtered;
    filtered.reserve(scored_list.size());

    const uint64_t now = getCurrentTimeInNano();
    const size_t topk = std::min(kTopKMin, scored_list.size());
    const double thresh = best_score * kNearBestRatio;

    for (size_t i = 0; i < scored_list.size(); ++i) {
        int dev_id = scored_list[i].first;
        double score = scored_list[i].second;

        bool in_near_best = (score <= thresh);
        bool in_topk = (i < topk);

        // starvation protection (per-thread, since we can't add fields to
        // DeviceInfo here)
        auto& tl_dev = tl_device_info[dev_id];
        bool starved = (tl_dev.last_picked_ts != 0 &&
                        now - tl_dev.last_picked_ts > kStarveNs);

        if (in_near_best || in_topk || starved) filtered.push_back(dev_id);
    }

    if (filtered.empty()) {
        // should not happen because topk ensures at least one, but keep safe
        filtered.push_back(scored_list.front().first);
    }

    // Deterministic RR among filtered, but allow small exploration.
    thread_local size_t rr_index = 0;
    thread_local uint64_t rng = 0;
    if (rng == 0) rng = SeedOnce();

    bool explore = (Rand01(rng) < kExploreEpsilon);
    if (explore && !score_map.empty()) {
        // Explore among ALL candidates (score_map), not only filtered.
        size_t pick = RandMod(rng, score_map.size());
        auto it = score_map.begin();
        std::advance(it, pick);
        chosen_dev_id = it->first;
    } else {
        chosen_dev_id = filtered[rr_index % filtered.size()];
        rr_index++;
    }

    // Update per-thread and global active_bytes.
    auto& chosen_tl = tl_device_info[chosen_dev_id];
    chosen_tl.active_bytes += length;
    chosen_tl.last_picked_ts = now;

    if (local_weight_ < 1 - 1e-6) {
        devices_[chosen_dev_id].active_bytes.fetch_add(
            length, std::memory_order_relaxed);
    }

    return Status::OK();
}

Status DeviceQuota::release(int dev_id, uint64_t length, double latency) {
    if (!enable_quota_) return Status::OK();
    auto it = devices_.find(dev_id);
    if (it == devices_.end())
        return Status::InvalidArgument("device not found");

    auto& dev = it->second;
    auto& tl_dev = tl_device_info[dev_id];

    if (local_weight_ < 1 - 1e-6)
        dev.active_bytes.fetch_sub(length, std::memory_order_relaxed);
    // guard: avoid underflow if something goes wrong
    if (tl_dev.active_bytes >= length)
        tl_dev.active_bytes -= length;
    else
        tl_dev.active_bytes = 0;

    if (!update_quota_params_) return Status::OK();

    // -----------------------------
    // Modeling / stability improvements:
    // - EWMA smoothing of observed latency
    // - Learning rate decays with sample count
    // - Relative error uses floor denom to avoid tiny pred_time amplifying
    // noise
    // - Trust-region update clipping (step clip), NOT tight parameter clamp
    // - Wide safety clamp only for NaN/inf prevention
    // -----------------------------
    static constexpr double kBeta0Min = 0.0;
    static constexpr double kBeta0MaxSafety =
        1e-2;  // safety only (10ms), not physics
    static constexpr double kBeta1MinSafety = 1e-3;
    static constexpr double kBeta1MaxSafety = 1e3;

    static constexpr double kEwmaGamma = 0.1;  // smoothing factor
    static constexpr double kRelErrFloor =
        20e-6;  // 20us denom floor to avoid blow-up
    static constexpr double kAlphaMin = 1e-4;
    static constexpr double kAlphaMax = 0.2;

    static constexpr double kMaxBeta0Step =
        2e-4;  // max abs additive step per update
    static constexpr double kMaxBeta1RatioStep =
        0.05;  // max ±5% multiplicative change

    // latency sanity
    double obs_time = latency;
    if (!std::isfinite(obs_time) || obs_time < 0.0) obs_time = 0.0;

    double bw = dev.bw_gbps * 1e9 / 8;
    if (!(bw > 0.0) || !std::isfinite(bw)) bw = 1.0;

    double theory_time = static_cast<double>(length) / bw;
    if (!std::isfinite(theory_time) || theory_time < 0.0) theory_time = 0.0;

    // Update EWMA of observed latency (per thread)
    if (tl_dev.samples == 0) {
        tl_dev.obs_ewma = obs_time;
    } else {
        tl_dev.obs_ewma =
            (1.0 - kEwmaGamma) * tl_dev.obs_ewma + kEwmaGamma * obs_time;
    }
    tl_dev.samples++;

    // Use smoothed observation for learning
    const double obs_smooth = tl_dev.obs_ewma;

    const double w = local_weight_;

    double beta0_g = dev.beta0.load(std::memory_order_relaxed);
    double beta1_g = dev.beta1.load(std::memory_order_relaxed);
    if (!std::isfinite(beta0_g)) beta0_g = 0.0;
    if (!std::isfinite(beta1_g) || beta1_g <= 0.0) beta1_g = 1.0;

    // Current mixed beta used for prediction
    double beta0 = w * tl_dev.beta0 + (1.0 - w) * beta0_g;
    double beta1 = w * tl_dev.beta1 + (1.0 - w) * beta1_g;
    if (!std::isfinite(beta0)) beta0 = 0.0;
    if (!std::isfinite(beta1) || beta1 <= 0.0) beta1 = 1.0;

    double pred_time = beta0 + beta1 * theory_time;
    if (!std::isfinite(pred_time) || pred_time < 0.0) pred_time = 0.0;

    double err = obs_smooth - pred_time;

    // Avoid tiny pred_time amplifying noise.
    double denom = SafeMax(pred_time, kRelErrFloor);
    double rel_err = err / denom;
    if (!std::isfinite(rel_err)) rel_err = 0.0;

    // Effective alpha: decay with samples -> stable convergence.
    // alpha_eff = alpha_ / sqrt(n)
    double alpha_eff = alpha_ / std::sqrt(static_cast<double>(tl_dev.samples));
    alpha_eff = ClampFast(alpha_eff, kAlphaMin, kAlphaMax);

    // Outlier gating: if error is extremely large, reduce learning (do not
    // amplify). z ~ normalized absolute error
    double z = std::abs(err) / denom;
    if (z > 1.0) {
        alpha_eff *= 0.1;
    } else if (z > 0.5) {
        alpha_eff *= 0.3;
    }

    // Proposed deltas
    double delta0 = alpha_eff * err;  // additive offset correction
    double delta1 =
        alpha_eff *
        rel_err;  // multiplicative slope correction (in relative space)

    // Trust-region update clipping: clamp each update, not the final parameter.
    double step0_l = ClampFast(w * delta0, -kMaxBeta0Step, +kMaxBeta0Step);
    double step1_l =
        ClampFast(w * delta1, -kMaxBeta1RatioStep, +kMaxBeta1RatioStep);

    // Local (TLS) update
    {
        double new_beta0_l = tl_dev.beta0 + step0_l;
        double new_beta1_l = tl_dev.beta1 * (1.0 + step1_l);

        tl_dev.beta0 = ClampFast(new_beta0_l, kBeta0Min, kBeta0MaxSafety);
        tl_dev.beta1 = ClampFast(new_beta1_l, kBeta1MinSafety, kBeta1MaxSafety);
    }

    // Global update only if we are mixing with global (w < 1)
    if (local_weight_ < 1 - 1e-6) {
        double step0_g =
            ClampFast((1.0 - w) * delta0, -kMaxBeta0Step, +kMaxBeta0Step);
        double step1_g = ClampFast((1.0 - w) * delta1, -kMaxBeta1RatioStep,
                                   +kMaxBeta1RatioStep);

        double new_beta0_g = beta0_g + step0_g;
        double new_beta1_g = beta1_g * (1.0 + step1_g);

        dev.beta0.store(ClampFast(new_beta0_g, kBeta0Min, kBeta0MaxSafety),
                        std::memory_order_relaxed);
        dev.beta1.store(
            ClampFast(new_beta1_g, kBeta1MinSafety, kBeta1MaxSafety),
            std::memory_order_relaxed);
        if (shared_quota_) {
            thread_local uint64_t tl_last_ts = 0;
            uint64_t now = getCurrentTimeInNano();
            if (now - tl_last_ts > diffusion_interval_) {
                tl_last_ts = now;
                return shared_quota_->diffusion();
            }
        }
    }
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake