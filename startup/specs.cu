// specs.cu — build with: nvcc -o specs specs.cu
#include <cstdio>
int main() {
    int clk_khz, bus_bits, l2_bytes;
    cudaDeviceGetAttribute(&clk_khz,  cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    double peak = 2.0 * clk_khz * 1e3 * (bus_bits / 8) / 1e9;  // GB/s
    printf("Memory clock %d MHz, bus %d bits, L2 %d MB\n",
           clk_khz / 1000, bus_bits, l2_bytes >> 20);
    printf("Theoretical peak bandwidth: %.0f GB/s\n", peak);
    int sms, core_khz;
    cudaDeviceGetAttribute(&sms,      cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&core_khz, cudaDevAttrClockRate, 0);   // in kHz
    double tflops = sms * 128.0 * 2.0 * core_khz * 1e3 / 1e12;
    printf("SMs %d, boost clock %d MHz\n", sms, core_khz / 1000);
    printf("Theoretical peak FP32: %.1f TFLOPS\n", tflops);
}