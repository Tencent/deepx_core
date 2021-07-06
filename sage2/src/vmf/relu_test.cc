// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_relu_ps_ref(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

ATTR_NOINLINE void eigen_relu_ps(uint64_t n, const float* x, float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = (xv > 0).select(xv, eigen_arrayxf_view_t::Zero(n));
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y1(n);
  std::vector<float> y2(n);
  std::vector<float> y3(n);
  randn(engine, &x);

  sage2_relu_ps_ref(n, x.data(), y1.data());

  sage2_relu_ps(n, x.data(), y2.data());
  CHECK_EQUAL(y1, y2);

  eigen_relu_ps(n, x.data(), y3.data());
  CHECK_EQUAL(y1, y3);
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  randn(engine, &x);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, x.data(), y.data());
  }
  EndTimer();
  return GetGFLOPS(1.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[3] = {0};
  gflops[0] = Benchmark(sage2_relu_ps_ref, n, m);
  gflops[1] = Benchmark(sage2_relu_ps, n, m);
  gflops[2] = Benchmark(eigen_relu_ps, n, m);
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
    PrintHeader1(3, "relu");
    PrintHeader2(3, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
