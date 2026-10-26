#pragma once
// Benchmarking harness: warmup runs, then N timed runs with CUDA events,
// reporting the median (robust to one-off slow or fast launches).
//
//   BenchResult r = benchmark([&] { my_kernel<<<grid, block>>>(...); });
//   report("my_kernel v1", r, bytes_moved, flops, dev);
//
// The lambda must only *launch* work (no cudaDeviceSynchronize, no memcpy).

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include "cuda_check.cuh"
#include "device_buffer.cuh"
#include "device_info.cuh"

struct BenchOptions {
  int warmup = 5;      // untimed runs: first-launch overhead, clocks ramping up
  int iters = 50;      // timed runs
  bool flush_l2 = false;  // evict L2 before each timed run (see README)
};

struct BenchResult {
  float median_ms = 0, min_ms = 0, max_ms = 0;
  int iters = 0;
};

template <typename LaunchFn>
BenchResult benchmark(LaunchFn&& launch, const BenchOptions& opt = {}) {
  // A buffer twice the size of L2; writing it pushes everything else out.
  std::unique_ptr<DeviceBuffer<char>> flusher;
  if (opt.flush_l2) {
    int l2 = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    flusher = std::make_unique<DeviceBuffer<char>>(2 * static_cast<size_t>(l2));
  }

  for (int i = 0; i < opt.warmup; ++i) {
    launch();
    CUDA_CHECK_LAST();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> times;
  times.reserve(opt.iters);
  for (int i = 0; i < opt.iters; ++i) {
    if (flusher) CUDA_CHECK(cudaMemsetAsync(flusher->data(), i & 0xff, flusher->bytes()));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_LAST();
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(times.begin(), times.end());
  BenchResult r;
  r.iters = opt.iters;
  r.min_ms = times.front();
  r.max_ms = times.back();
  size_t mid = times.size() / 2;
  r.median_ms = (times.size() % 2) ? times[mid] : 0.5f * (times[mid - 1] + times[mid]);
  return r;
}

inline double to_gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double to_gflops(double flops, float ms) { return flops / (ms * 1e-3) / 1e9; }

// One result line: time, bandwidth and compute, each against the GPU's peak.
// Pass flops = 0 to skip the compute column (pure data-movement kernels).
inline void report(const char* label, const BenchResult& r, double bytes, double flops,
                   const DeviceInfo& dev) {
  double gbps = to_gbps(bytes, r.median_ms);
  std::printf("%-28s median %9.4f ms  [min %.4f, max %.4f]  %8.1f GB/s (%5.1f%% peak)",
              label, r.median_ms, r.min_ms, r.max_ms, gbps, 100.0 * gbps / dev.peak_gbps);
  if (flops > 0) {
    double gf = to_gflops(flops, r.median_ms);
    std::printf("  %9.1f GFLOPS (%5.1f%% peak)", gf, 100.0 * gf / dev.peak_fp32_gflops);
  }
  std::printf("\n");
}
