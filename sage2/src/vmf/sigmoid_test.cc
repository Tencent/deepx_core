// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE float libc_sigmoid_ss(float x) {
  float expx = expf(x);
  return expx / (expx + 1);
}

ATTR_NOINLINE void libc_sigmoid_ps(uint64_t n, const float* x, float* y) {
  float expx;
  for (uint64_t i = 0; i < n; ++i) {
    expx = expf(x[i]);
    y[i] = expx / (expx + 1);
  }
}

ATTR_NOINLINE void eigen_sigmoid_ps(uint64_t n, const float* x, float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  auto expx = xv.exp();
  yv = expx / (expx + 1);
}

void Test() {
  std::default_random_engine engine;
  float x;
  float y1, y2, y3;

  for (int i = 0; i < 10000; ++i) {
    rand(engine, &x, -80, 80);

    y1 = libc_sigmoid_ss(x);

    y2 = sage2_sigmoid_ss(x);
    CHECK_EQUAL(y1, y2);

    eigen_sigmoid_ps(1, &x, &y3);
    CHECK_EQUAL(y1, y3);  // NOLINT
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y1(n);
  std::vector<float> y2(n);
  std::vector<float> y3(n);
  rand(engine, &x, -80, 80);

  libc_sigmoid_ps(n, x.data(), y1.data());

  sage2_sigmoid_ps(n, x.data(), y2.data());
  CHECK_EQUAL(y1, y2);

  eigen_sigmoid_ps(n, x.data(), y3.data());
  CHECK_EQUAL(y1, y3);
}

void TestSpecial() {
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(FLOAT_QUIET_NAN));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(FLOAT_SIGNALING_NAN));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(FLOAT_INF));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(FLOAT_NINF));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(100.0f));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(10.0f));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(1.0f));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(-100.0f));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(-10.0f));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(-1.0f));
  CHECK_NOT_NAN_INF(sage2_sigmoid_ss(0.0f));

  std::vector<float> x = {FLOAT_QUIET_NAN,
                          FLOAT_SIGNALING_NAN,
                          FLOAT_INF,
                          FLOAT_NINF,
                          100.0f,
                          10.0f,
                          1.0f,
                          -100.0f,
                          -10.0f,
                          -1.0f,
                          0.0f};
  std::vector<float> y(x.size());
  sage2_sigmoid_ps(x.size(), x.data(), y.data());
  CHECK_NOT_NAN_INF(y);
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  rand(engine, &x, -80, 80);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, x.data(), y.data());
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark(int n) {
  int m = 20000000 / n;
  double mops[3] = {0};
  mops[0] = Benchmark(libc_sigmoid_ps, n, m);
  mops[1] = Benchmark(sage2_sigmoid_ps, n, m);
  mops[2] = Benchmark(eigen_sigmoid_ps, n, m);
  PrintContent(n, mops);
}

int main(int argc, char** argv) {
  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    Test();
    for (int i = 1; i < 1000; ++i) {
      Test(i);
    }
    TestSpecial();
  }

  if (action & 2) {
    PrintHeader1(3, "sigmoid");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("libc", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
