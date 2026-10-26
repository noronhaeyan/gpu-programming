# CUDA kernels

Hand-written CUDA kernels, each checked against a CPU reference and benchmarked against the GPU's theoretical peak.

## Hardware baseline

| | RTX 6000 Ada (sm_89) |
|---|---|
| Peak DRAM bandwidth | 960 GB/s |
| Peak FP32 | 91.1 TFLOPS |
| Ridge point | 95 FLOP/byte |
| L2 cache | 96 MB |
| Measured streaming bandwidth (`add`, 2^26 floats, median of 10) | 804 GB/s (84% of peak) |

Kernels below the ridge point are memory-bound (judge them in GB/s); kernels above it are compute-bound (judge them in GFLOPS).

## Layout

```
CMakeLists.txt         build for every project
common/
  cuda_check.cuh       CUDA_CHECK / CUDA_CHECK_LAST error macros
  device_buffer.cuh    DeviceBuffer<T>: RAII GPU memory (no leaks, no manual cudaFree)
  device_info.cuh      queries the GPU and computes its peak GB/s and GFLOPS
  bench.cuh            benchmark(): warmup + N timed runs, median; report(): % of peak
  check.hpp            check_close() against the CPU reference; random_vector()
tools/specs.cu         prints the current GPU's peaks
01_vector_add/         example project: z = a*x + y
```

## Build and run

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5          # an idle GPU from nvidia-smi

cmake -B build                         # once (default arch: sm_89)
cmake --build build -j                 # after every change
./build/bin/specs
./build/bin/vector_add
```

Options at configure time:

- `-DCMAKE_CUDA_ARCHITECTURES=90` for another GPU (90 = H100).
- `-DSHOW_PTXAS=ON` prints registers, shared memory, and spills per kernel at build time.
- `-DCMAKE_BUILD_TYPE=Debug` for debugging (never benchmark a Debug build).

## Adding a new project

1. Copy `01_vector_add/` to a new folder, e.g. `02_transpose/`.
2. Replace the kernel, the CPU reference, and the byte and FLOP counts.
3. Add one line to `CMakeLists.txt`: `add_cuda_executable(transpose 02_transpose/transpose.cu)`.

Keep every version of a kernel (`v1`, `v2`, ...) in the file and benchmark them side by side. The table of versions is the write-up.

## Benchmarking rules

- **Check correctness before timing.** A fast wrong kernel is worthless.
- **Report the median**, not the mean. One slow or fast launch on a shared machine skews the mean.
- **Count bytes and FLOPs by hand** from the algorithm (minimum required traffic), not from what the kernel happens to do. Then a wasteful kernel shows up as low % of peak.
- **Mind the L2 cache (96 MB here).** If the data fits in L2, repeated runs read from cache and can exceed DRAM peak. Use inputs much larger than L2 to measure DRAM bandwidth, or pass `flush_l2 = true` in `BenchOptions` to evict L2 before every timed run.
- **Pin to one idle GPU** with `CUDA_VISIBLE_DEVICES`. Another job on the same card makes timings noisy.

## Profiling

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=true -o reports/<name> ./build/bin/<binary>
sudo $(which ncu) --set full --import-source yes -o reports/<name> ./build/bin/<binary>   # AWS only
```

`-lineinfo` is on for every build, so ncu maps metrics back to source lines.
