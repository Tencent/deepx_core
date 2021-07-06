// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include "benchmark.h"
#include "internal_macro.h"

ATTR_NOINLINE float sage2_euclidean_distance_ps_ref(uint64_t n, const float* x,
                                                    const float* y) {
  float sum = 0;
  for (uint64_t i = 0; i < n; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sqrtf(sum);
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  randn(engine, &x);
  randn(engine, &y);

  float ed1 = sage2_euclidean_distance_ps_ref(n, x.data(), y.data());

  float ed2 = sage2_euclidean_distance_ps(n, x.data(), y.data());
  CHECK_EQUAL(ed1, ed2);
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
  return GetGFLOPS(3.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[2] = {0};
  gflops[0] = Benchmark(sage2_euclidean_distance_ps_ref, n, m);
  gflops[1] = Benchmark(sage2_euclidean_distance_ps, n, m);
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
    PrintHeader1(2, "euclidean_distance");
    PrintHeader2(2, "GFLOPS");
    PrintHeader3("ref", "opt");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
