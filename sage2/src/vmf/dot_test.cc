// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/cblas_adapter.h>
#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

const auto* cblas = sage2_cblas_cblas();
const auto* mkl = sage2_cblas_mkl();
const auto* openblas = sage2_cblas_openblas();

ATTR_NOINLINE float sage2_dot_ps_ref(uint64_t n, const float* x,
                                     const float* y) {
  float sum = 0;
  for (uint64_t i = 0; i < n; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}

ATTR_NOINLINE float eigen_dot_ps(uint64_t n, const float* x, const float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  return (xv * yv).sum();
}

ATTR_NOINLINE float cblas_cblas_dot_ps(uint64_t n, const float* x,
                                       const float* y) {
  return cblas->cblas_sdot((int)n, x, 1, y, 1);
}

ATTR_NOINLINE float cblas_mkl_dot_ps(uint64_t n, const float* x,
                                     const float* y) {
  return mkl->cblas_sdot((int)n, x, 1, y, 1);
}

ATTR_NOINLINE float cblas_openblas_dot_ps(uint64_t n, const float* x,
                                          const float* y) {
  return openblas->cblas_sdot((int)n, x, 1, y, 1);
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  randn(engine, &x);
  randn(engine, &y);

  float dot1 = sage2_dot_ps_ref(n, x.data(), y.data());

  float dot2 = sage2_dot_ps(n, x.data(), y.data());
  CHECK_EQUAL(dot1, dot2);

  float dot3 = eigen_dot_ps(n, x.data(), y.data());
  CHECK_EQUAL(dot1, dot3);

  if (cblas->cblas_sdot) {
    float dot4 = cblas_cblas_dot_ps(n, x.data(), y.data());
    CHECK_EQUAL(dot1, dot4);
  }

  if (mkl->cblas_sdot) {
    float dot5 = cblas_mkl_dot_ps(n, x.data(), y.data());
    CHECK_EQUAL(dot1, dot5);
  }

  if (openblas->cblas_sdot) {
    float dot6 = cblas_openblas_dot_ps(n, x.data(), y.data());
    CHECK_EQUAL(dot1, dot6);
  }
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  randn(engine, &x);
  randn(engine, &y);

  BeginTimer();
  float sum = 0;
  for (int i = 0; i < m; ++i) {
    sum += func(n, x.data(), y.data());
  }
  EndTimer();
  REFER(sum);
  return GetGFLOPS(2.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[6] = {0};
  gflops[0] = Benchmark(sage2_dot_ps_ref, n, m);
  gflops[1] = Benchmark(sage2_dot_ps, n, m);
  gflops[2] = Benchmark(eigen_dot_ps, n, m);
  if (cblas->cblas_sdot) {
    gflops[3] = Benchmark(cblas_cblas_dot_ps, n, m);
  }
  if (mkl->cblas_sdot) {
    gflops[4] = Benchmark(cblas_mkl_dot_ps, n, m);
  }
  if (openblas->cblas_sdot) {
    gflops[5] = Benchmark(cblas_openblas_dot_ps, n, m);
  }
  PrintContent(n, gflops);
}

int main(int argc, char** argv) {
  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    for (int i = 1; i < 1000; ++i) {
      Test(i);
    }
  }

  if (action & 2) {
    PrintHeader1(6, "dot");
    PrintHeader2(6, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen", cblas->vendor, mkl->vendor,
                 openblas->vendor);
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
