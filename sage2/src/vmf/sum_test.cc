// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE float sage2_sum_ps_ref(uint64_t n, const float* x) {
  float sum = 0;
  for (uint64_t i = 0; i < n; ++i) {
    sum += x[i];
  }
  return sum;
}

ATTR_NOINLINE float eigen_sum_ps(uint64_t n, const float* x) {
  auto xv = make_eigen_arrayxf_view(x, n);
  return xv.sum();
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  randn(engine, &x);

  float sum1 = sage2_sum_ps_ref(n, x.data());

  float sum2 = sage2_sum_ps(n, x.data());
  CHECK_EQUAL(sum1, sum2);

  float sum3 = eigen_sum_ps(n, x.data());
  CHECK_EQUAL(sum1, sum3);
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  randn(engine, &x);

  BeginTimer();
  float sum = 0;
  for (int i = 0; i < m; ++i) {
    sum += func(n, x.data());
  }
  EndTimer();
  REFER(sum);
  return GetGFLOPS(1.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[3] = {0};
  gflops[0] = Benchmark(sage2_sum_ps_ref, n, m);
  gflops[1] = Benchmark(sage2_sum_ps, n, m);
  gflops[2] = Benchmark(eigen_sum_ps, n, m);
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
    PrintHeader1(3, "sum");
    PrintHeader2(3, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
