#pragma once
// RAII wrapper for device memory: cudaMalloc in the constructor, cudaFree in
// the destructor, so GPU memory can never leak.
//
//   DeviceBuffer<float> d_x(n);
//   d_x.copy_from_host(h_x);
//   my_kernel<<<grid, block>>>(d_x.data(), n);
//   std::vector<float> result = d_x.to_host();

#include <cstddef>
#include <vector>

#include "cuda_check.cuh"

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t n) : n_(n) {
    if (n_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  // Movable, not copyable (a copy would double-free).
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), n_(other.n_) {
    other.ptr_ = nullptr;
    other.n_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = other.ptr_;
      n_ = other.n_;
      other.ptr_ = nullptr;
      other.n_ = 0;
    }
    return *this;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return n_; }
  size_t bytes() const { return n_ * sizeof(T); }

  void copy_from_host(const std::vector<T>& host) {
    if (host.size() != n_) {
      std::fprintf(stderr, "copy_from_host: size mismatch (%zu vs %zu)\n", host.size(), n_);
      std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice));
  }

  std::vector<T> to_host() const {
    std::vector<T> host(n_);
    CUDA_CHECK(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
    return host;
  }

  void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, bytes())); }

 private:
  T* ptr_ = nullptr;
  size_t n_ = 0;
};
