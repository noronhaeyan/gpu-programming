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

__global__ void transpose_v1(int rows, int cols, const float* x, float* z) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;

  if (i < rows && j < cols) {
    z[rows*j + i] = x[cols*i + j];
  }
}

// ---- CPU reference ---------------------------------------------------------

void transpose_cpu(int rows, int cols, const std::vector<float>& x, std::vector<float>& z) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      z[rows*j + i] = x[cols*i + j];
    }
  }
}

// ---- Driver ----------------------------------------------------------------

int main(int argc, char** argv) {
  const int rows = (argc > 1) ? std::atoi(argv[1]) : (1 << 26);
  const int cols = (argc > 2) ? std::atoi(argv[2]) : rows;

  DeviceInfo dev = get_device_info();
  print_device_info(dev);
  std::printf("\nrow length = %d\n", rows);
  std::printf("column length = %d\n", cols);
  std::printf("Total memory =  #.1f MiB \n" , rows * cols * sizeof(float) / 1048576.0);

  // 1-2. Inputs and reference.
  std::vector<float> h_x = random_vector(rows*cols, 1), h_ref(rows*cols);
  transpose_cpu(rows, cols, h_x, h_ref);

  DeviceBuffer<float> d_x(rows*cols), d_z(rows*cols);
  d_x.copy_from_host(h_x);

  const int block = 256;
  const int grid = (n + block - 1) / block;
  auto launch_v1 = [&] { transpose_v1<<<grid, block>>>(rows, cols, d_x.data(), d_z.data()); };

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
