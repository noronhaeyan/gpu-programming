// Example project: z = a*x + y. Shows the pattern every kernel in this repo follows:
//   1. make inputs on the host   2. compute a CPU reference
//   3. run the GPU kernel        4. check it against the reference
//   5. benchmark it and report against the GPU's peak
//
//   ./build/bin/vector_add            (n = 2^26)
//   ./build/bin/vector_add 1048576    (custom n)
#include <cstdlib>

#include "bench.cuh"
#include "check.hpp"

// ---- Kernel(s) -------------------------------------------------------------
// Add new versions below (v2, v3, ...) and benchmark them side by side in main().

__global__ void transpose_v1(int n, float a, const float* x, const float* y, float* z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] = a * x[i] + y[i];
}

// ---- CPU reference ---------------------------------------------------------

void transpose_cpu(int n, float a, const std::vector<float>& x, const std::vector<float>& y,
              std::vector<float>& z) {
  for (int i = 0; i < n; ++i) z[i] = a * x[i] + y[i];
}

// ---- Driver ----------------------------------------------------------------

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 26);
  const float a = 2.0f;

  DeviceInfo dev = get_device_info();
  print_device_info(dev);
  std::printf("\nn = %d (%.1f MiB per array)\n\n", n, n * sizeof(float) / 1048576.0);

  // 1-2. Inputs and reference.
  std::vector<float> h_x = random_vector(n, 1), h_y = random_vector(n, 2), h_ref(n);
  transpose_cpu(n, a, h_x, h_y, h_ref);

  DeviceBuffer<float> d_x(n), d_y(n), d_z(n);
  d_x.copy_from_host(h_x);
  d_y.copy_from_host(h_y);

  const int block = 256;
  const int grid = (n + block - 1) / block;
  auto launch_v1 = [&] { transpose_v1<<<grid, block>>>(n, a, d_x.data(), d_y.data(), d_z.data()); };

  // 3-4. Correctness first: a fast wrong kernel is worthless.
  d_z.zero();
  launch_v1();
  CUDA_CHECK_LAST();
  // a*x+y may compile to a fused multiply-add on the GPU, so allow ~1 ulp.
  if (!check_close(h_ref, d_z.to_host(), 1e-6, 1e-6)) return EXIT_FAILURE;

  // 5. Benchmark. Traffic: read x, read y, write z = 3 floats per element.
  const double bytes = 3.0 * n * sizeof(float);
  const double flops = 2.0 * n;  // one multiply + one add per element
  std::printf("\n");
  report("transpose_v1", benchmark(launch_v1), bytes, flops, dev);

  BenchOptions cold;
  cold.flush_l2 = true;
  report("transpose_v1 (L2 flushed)", benchmark(launch_v1, cold), bytes, flops, dev);
  return 0;
}
