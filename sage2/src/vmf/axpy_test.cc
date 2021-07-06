// Copyright 2019 the deepx authors.
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

ATTR_NOINLINE void sage2_axpy_ps_ref(uint64_t n, float alpha, const float* x,
                                     float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

ATTR_NOINLINE void eigen_axpy_ps(uint64_t n, float alpha, const float* x,
                                 float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv += alpha * xv;
}

ATTR_NOINLINE void cblas_cblas_axpy_ps(uint64_t n, float alpha, const float* x,
                                       float* y) {
  cblas->cblas_saxpy((int)n, alpha, x, 1, y, 1);
}

ATTR_NOINLINE void cblas_mkl_axpy_ps(uint64_t n, float alpha, const float* x,
                                     float* y) {
  mkl->cblas_saxpy((int)n, alpha, x, 1, y, 1);
}

ATTR_NOINLINE void cblas_openblas_axpy_ps(uint64_t n, float alpha,
                                          const float* x, float* y) {
  openblas->cblas_saxpy((int)n, alpha, x, 1, y, 1);
}

void Test(int n) {
  std::default_random_engine engine;
  float alpha = 0;
  std::vector<float> x(n);
  std::vector<float> y1(n);
  std::vector<float> y2(n);
  std::vector<float> y3(n);
  std::vector<float> y4(n);
  std::vector<float> y5(n);
  std::vector<float> y6(n);
  randn(engine, &alpha);
  randn(engine, &x);
  randn(engine, &y1);
  y2 = y1;
  y3 = y1;
  y4 = y1;
  y5 = y1;
  y6 = y1;

  sage2_axpy_ps_ref(n, alpha, x.data(), y1.data());

  sage2_axpy_ps(n, alpha, x.data(), y2.data());
  CHECK_EQUAL(y1, y2);

  eigen_axpy_ps(n, alpha, x.data(), y3.data());
  CHECK_EQUAL(y1, y3);

  if (cblas->cblas_saxpy) {
    cblas_cblas_axpy_ps(n, alpha, x.data(), y4.data());
    CHECK_EQUAL(y1, y4);
  }

  if (mkl->cblas_saxpy) {
    cblas_mkl_axpy_ps(n, alpha, x.data(), y5.data());
    CHECK_EQUAL(y1, y5);
  }

  if (openblas->cblas_saxpy) {
    cblas_openblas_axpy_ps(n, alpha, x.data(), y6.data());
    CHECK_EQUAL(y1, y6);
  }
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  float alpha = 0;
  std::vector<float> x(n);
  std::vector<float> y(n);
  randn(engine, &alpha);
  randn(engine, &x);
  randn(engine, &y);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, alpha, x.data(), y.data());
  }
  EndTimer();
  return GetGFLOPS(2.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[6] = {0};
  gflops[0] = Benchmark(sage2_axpy_ps_ref, n, m);
  gflops[1] = Benchmark(sage2_axpy_ps, n, m);
  gflops[2] = Benchmark(eigen_axpy_ps, n, m);
  if (cblas->cblas_saxpy) {
    gflops[3] = Benchmark(cblas_cblas_axpy_ps, n, m);
  }
  if (mkl->cblas_saxpy) {
    gflops[4] = Benchmark(cblas_mkl_axpy_ps, n, m);
  }
  if (openblas->cblas_saxpy) {
    gflops[5] = Benchmark(cblas_openblas_axpy_ps, n, m);
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
    PrintHeader1(6, "axpy");
    PrintHeader2(6, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen", cblas->vendor, mkl->vendor,
                 openblas->vendor);
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
