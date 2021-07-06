// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/cblas_adapter.h>
#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

const auto* mkl = sage2_cblas_mkl();

ATTR_NOINLINE void libc_sqrt_ps(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = sqrtf(x[i]);
  }
}

ATTR_NOINLINE void eigen_sqrt_ps(uint64_t n, const float* x, float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv.sqrt();
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y1(n);
  std::vector<float> y2(n);
  std::vector<float> y3(n);
  std::vector<float> y4(n);
  rand(engine, &x, 0, 100);

  libc_sqrt_ps(n, x.data(), y1.data());

  sage2_sqrt_ps(n, x.data(), y2.data());
  CHECK_EQUAL(y1, y2);

  eigen_sqrt_ps(n, x.data(), y3.data());
  CHECK_EQUAL(y1, y3);

  if (mkl->vsSqrt) {
    mkl->vsSqrt(n, x.data(), y4.data());
    CHECK_EQUAL(y1, y4);
  }
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  rand(engine, &x, 0, 100);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, x.data(), y.data());
  }
  EndTimer();
  return GetGFLOPS(1.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[4] = {0};
  gflops[0] = Benchmark(libc_sqrt_ps, n, m);
  gflops[1] = Benchmark(sage2_sqrt_ps, n, m);
  gflops[2] = Benchmark(eigen_sqrt_ps, n, m);
  if (mkl->vsSqrt) {
    gflops[3] = Benchmark(mkl->vsSqrt, n, m);
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
    PrintHeader1(4, "sqrt");
    PrintHeader2(4, "GFLOPS");
    PrintHeader3("libc", "opt", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
