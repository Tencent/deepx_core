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

ATTR_NOINLINE float libc_tanh_ss(float x) { return tanhf(x); }

ATTR_NOINLINE void libc_tanh_ps(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = tanhf(x[i]);
  }
}

ATTR_NOINLINE float sage2_tanh_ss_ref(float x) {
  static const float EXP_HI = 88.02969193111305f;
  static const float EXP_LO = -88.02969193111305f;
  static const float EXP_INV_LOG2E = 1.44269504088896341f;
  static const float EXP_P0 = 1.3537703155e-02f;
  static const float EXP_P1 = 5.2170695889e-02f;
  static const float EXP_P2 = 2.4121210200e-01f;
  static const float EXP_P3 = 6.9307905933e-01f;
  static const float EXP_P4 = 1.0000001462e+00f;
  static const union {
    float f;
    uint32_t u;
  } EXP_ONE = {1.0f};

  float a, fa, b, c;
  union {
    float f;
    uint32_t u;
  } pow2n;

  x = x + x;

  x = (x > EXP_HI) ? EXP_HI : x;
  x = (x < EXP_LO) ? EXP_LO : x;

  a = x * EXP_INV_LOG2E;
  fa = floorf(a);
  b = a - fa;

  c = EXP_P0;
  c = c * b + EXP_P1;
  c = c * b + EXP_P2;
  c = c * b + EXP_P3;
  c = c * b + EXP_P4;

  pow2n.u = (int)fa;
  pow2n.u = pow2n.u << 23;
  pow2n.u = pow2n.u + EXP_ONE.u;
  c = c * pow2n.f;
  return (c - 1) / (c + 1);
}

ATTR_NOINLINE void sage2_tanh_ps_ref(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = sage2_tanh_ss_ref(x[i]);
  }
}

ATTR_NOINLINE void eigen_tanh_ps(uint64_t n, const float* x, float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv.tanh();
}

void Test() {
  std::default_random_engine engine;
  float x;
  float y1, y2, y3, y4, y5;

  for (int i = 0; i < 10000; ++i) {
    rand(engine, &x, -80, 80);

    y1 = libc_tanh_ss(x);

    y2 = sage2_tanh_ss_ref(x);
    CHECK_EQUAL(y1, y2);

    y3 = sage2_tanh_ss(x);
    CHECK_EQUAL(y1, y3);

    eigen_tanh_ps(1, &x, &y4);
    CHECK_EQUAL(y1, y4);  // NOLINT

    if (mkl->vsTanh) {
      mkl->vsTanh(1, &x, &y5);
      CHECK_EQUAL(y1, y5);
    }
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y1(n);
  std::vector<float> y2(n);
  std::vector<float> y3(n);
  std::vector<float> y4(n);
  std::vector<float> y5(n);
  rand(engine, &x, -80, 80);

  libc_tanh_ps(n, x.data(), y1.data());

  sage2_tanh_ps_ref(n, x.data(), y2.data());
  CHECK_EQUAL(y1, y2);

  sage2_tanh_ps(n, x.data(), y3.data());
  CHECK_EQUAL(y1, y3);

  eigen_tanh_ps(n, x.data(), y4.data());
  CHECK_EQUAL(y1, y4);

  if (mkl->vsTanh) {
    mkl->vsTanh(n, x.data(), y5.data());
    CHECK_EQUAL(y1, y5);
  }
}

void TestSpecial() {
  CHECK_NOT_NAN_INF(sage2_tanh_ss(FLOAT_QUIET_NAN));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(FLOAT_SIGNALING_NAN));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(FLOAT_INF));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(FLOAT_NINF));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(100.0f));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(10.0f));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(1.0f));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(-100.0f));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(-10.0f));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(-1.0f));
  CHECK_NOT_NAN_INF(sage2_tanh_ss(0.0f));

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
  sage2_log_ps(x.size(), x.data(), y.data());
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
  double mops[5] = {0};
  mops[0] = Benchmark(libc_tanh_ps, n, m);
  mops[1] = Benchmark(sage2_tanh_ps_ref, n, m);
  mops[2] = Benchmark(sage2_tanh_ps, n, m);
  mops[3] = Benchmark(eigen_tanh_ps, n, m);
  if (mkl->vsTanh) {
    mops[4] = Benchmark(mkl->vsTanh, n, m);
  }
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
    PrintHeader1(5, "tanh");
    PrintHeader2(5, "Mop/s");
    PrintHeader3("libc", "ref", "opt", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
