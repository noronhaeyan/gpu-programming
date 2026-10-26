#pragma once
// Error checking. Every CUDA API call returns an error code that is silently
// ignored unless you check it; wrap every call in CUDA_CHECK.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d\n  %s\n  -> %s\n", __FILE__,  \
                   __LINE__, #call, cudaGetErrorString(err_));                \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// Kernel launches return nothing. Call this right after a launch to catch
// bad launch configurations (e.g. too many threads per block).
// Errors that happen *while* the kernel runs only surface at the next
// synchronizing call, which CUDA_CHECK will then report.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
