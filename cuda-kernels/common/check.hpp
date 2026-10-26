#pragma once
// Correctness checking against a CPU reference.
//
// GPU results rarely match the CPU bit-for-bit: the GPU may sum in a
// different order, or fuse a*b+c into one rounding (FMA). So compare with a
// tolerance: an element passes if |got - ref| <= atol + rtol * |ref|.
//
// Rough tolerances for float32:
//   elementwise ops (add, scale, transpose)   rtol 1e-6  (transpose: exact)
//   reductions / dot products over n values   rtol ~1e-5 * sqrt(n), or compare in double
//   GEMM with K-length dot products           rtol 1e-3 .. 1e-4

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

template <typename T>
bool check_close(const std::vector<T>& ref, const std::vector<T>& got, double rtol = 1e-5,
                 double atol = 1e-6, int max_print = 5) {
  if (ref.size() != got.size()) {
    std::printf("CHECK FAILED: size mismatch (ref %zu, got %zu)\n", ref.size(), got.size());
    return false;
  }
  size_t bad = 0;
  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double r = static_cast<double>(ref[i]);
    double g = static_cast<double>(got[i]);
    double abs_err = std::fabs(g - r);
    double rel_err = abs_err / std::max(std::fabs(r), 1e-30);
    if (!(abs_err <= atol + rtol * std::fabs(r))) {  // also catches NaN
      if (bad < static_cast<size_t>(max_print))
        std::printf("  mismatch at [%zu]: ref %.8g, got %.8g\n", i, r, g);
      ++bad;
    }
    max_abs = std::max(max_abs, abs_err);
    if (std::fabs(r) > atol) max_rel = std::max(max_rel, rel_err);
  }
  if (bad) {
    std::printf("CHECK FAILED: %zu / %zu mismatches (max abs err %.3g, max rel err %.3g)\n",
                bad, ref.size(), max_abs, max_rel);
    return false;
  }
  std::printf("CHECK PASSED (%zu elements, max abs err %.3g, max rel err %.3g)\n", ref.size(),
              max_abs, max_rel);
  return true;
}

// Reproducible random inputs: same seed, same data, every run.
inline std::vector<float> random_vector(size_t n, uint32_t seed = 42, float lo = -1.0f,
                                        float hi = 1.0f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}
