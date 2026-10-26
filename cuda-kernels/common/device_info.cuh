#pragma once
// Queries the current GPU and computes its theoretical peaks, so every
// benchmark can report "% of peak" for the machine it actually ran on.

#include <cstdio>
#include <string>

#include "cuda_check.cuh"

struct DeviceInfo {
  std::string name;
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  int core_clock_mhz = 0;
  int mem_clock_mhz = 0;
  int bus_width_bits = 0;
  size_t l2_bytes = 0;
  int fp32_lanes_per_sm = 0;
  double peak_gbps = 0.0;         // theoretical DRAM bandwidth
  double peak_fp32_gflops = 0.0;  // theoretical FP32 (FMA = 2 FLOPs)

  double ridge_flop_per_byte() const { return peak_fp32_gflops / peak_gbps; }
};

// FP32 units per SM depend on the architecture and cannot be queried.
// Verify against NVIDIA's whitepaper for any GPU not listed here.
inline int fp32_lanes_per_sm(int major, int minor) {
  if (major == 7) return 64;                // Volta, Turing
  if (major == 8 && minor == 0) return 64;  // A100
  return 128;                               // GA10x (8.6), Ada (8.9), Hopper (9.0), Blackwell
}

inline DeviceInfo get_device_info(int device = 0) {
  DeviceInfo d;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  d.name = prop.name;
  d.cc_major = prop.major;
  d.cc_minor = prop.minor;
  d.sm_count = prop.multiProcessorCount;
  d.l2_bytes = prop.l2CacheSize;

  int core_khz = 0, mem_khz = 0, bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&core_khz, cudaDevAttrClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device));
  d.core_clock_mhz = core_khz / 1000;
  d.mem_clock_mhz = mem_khz / 1000;
  d.bus_width_bits = bus_bits;
  d.fp32_lanes_per_sm = fp32_lanes_per_sm(d.cc_major, d.cc_minor);

  // DDR: 2 transfers per memory clock.
  d.peak_gbps = 2.0 * mem_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  d.peak_fp32_gflops = d.sm_count * d.fp32_lanes_per_sm * 2.0 * core_khz * 1e3 / 1e9;
  return d;
}

inline void print_device_info(const DeviceInfo& d) {
  std::printf("GPU: %s (sm_%d%d), %d SMs\n", d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count);
  std::printf("  Peak DRAM bandwidth: %.0f GB/s   (mem clock %d MHz, %d-bit bus)\n",
              d.peak_gbps, d.mem_clock_mhz, d.bus_width_bits);
  std::printf("  Peak FP32:           %.1f TFLOPS (%d SMs x %d lanes x 2 x %d MHz)\n",
              d.peak_fp32_gflops / 1e3, d.sm_count, d.fp32_lanes_per_sm, d.core_clock_mhz);
  std::printf("  Ridge point:         %.0f FLOP/byte\n", d.ridge_flop_per_byte());
  std::printf("  L2 cache:            %zu MB\n", d.l2_bytes >> 20);
}
