// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE float sage2_max_ps_ref(uint64_t n, const float* x) {
  float max = x[0];
  for (uint64_t i = 1; i < n; ++i) {
    if (x[i] > max) {
      max = x[i];
    }
  }
  return max;
}

ATTR_NOINLINE float eigen_max_ps(uint64_t n, const float* x) {
  auto xv = make_eigen_arrayxf_view(x, n);
  return xv.maxCoeff();
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  randn(engine, &x);

  float max1 = sage2_max_ps_ref(n, x.data());

  float max2 = sage2_max_ps(n, x.data());
  CHECK_EQUAL(max1, max2);

  float max3 = eigen_max_ps(n, x.data());
  CHECK_EQUAL(max1, max3);
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  randn(engine, &x);

  BeginTimer();
  float max = 0;
  for (int i = 0; i < m; ++i) {
    max += func(n, x.data());
  }
  EndTimer();
  REFER(max);
  return GetGFLOPS(1.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[3] = {0};
  gflops[0] = Benchmark(sage2_max_ps_ref, n, m);
  gflops[1] = Benchmark(sage2_max_ps, n, m);
  gflops[2] = Benchmark(eigen_max_ps, n, m);
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
    PrintHeader1(3, "max");
    PrintHeader2(3, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
