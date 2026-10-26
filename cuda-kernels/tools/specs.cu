// Prints the GPU's theoretical peaks. Run this first on any new machine.
//   ./build/bin/specs
#include "device_info.cuh"

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::printf("Visible GPUs: %d (this program reports device 0)\n\n", count);
  print_device_info(get_device_info(0));
  return 0;
}
