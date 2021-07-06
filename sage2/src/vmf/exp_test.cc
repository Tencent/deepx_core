// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//
// Reference:
// http://software-lisc.fbk.eu/avx_mathfun
// https://stackoverflow.com/questions/47025373/fastest-implementation-of-exponential-function-using-sse
//

#include <sage2/cblas_adapter.h>
#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

const auto* mkl = sage2_cblas_mkl();

ATTR_NOINLINE float libc_exp_ss(float x) { return expf(x); }

ATTR_NOINLINE void libc_exp_ps(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = expf(x[i]);
  }
}

ATTR_NOINLINE float sage2_exp_ss_ref1(float x) {
  static const float EXP_HI = 88.02969193111305f;
  static const float EXP_LO = -88.02969193111305f;
  static const float EXP_INV_LOG2E = 1.44269504088896341f;
  static const float EXP_C0 = -0.693359375f;
  static const float EXP_C1 = 2.12194440e-4f;
  static const float EXP_P0 = 2.7565422393e-04f;
  static const float EXP_P1 = 1.3038713518e-03f;
  static const float EXP_P2 = 8.3795212816e-03f;
  static const float EXP_P3 = 4.1653515712e-02f;
  static const float EXP_P4 = 1.6666851064e-01f;
  static const float EXP_P5 = 4.9999990238e-01f;
  static const union {
    float f;
    uint32_t u;
  } EXP_ONE = {1.0f};

  float a, b, c;
  union {
    float f;
    uint32_t u;
  } pow2n;

  x = (x > EXP_HI) ? EXP_HI : x;
  x = (x < EXP_LO) ? EXP_LO : x;

  a = x * EXP_INV_LOG2E;
  a = floorf(a);

  x = x + a * EXP_C0;
  x = x + a * EXP_C1;
  b = x * x;

  c = EXP_P0;
  c = c * x + EXP_P1;
  c = c * x + EXP_P2;
  c = c * x + EXP_P3;
  c = c * x + EXP_P4;
  c = c * x + EXP_P5;
  c = c * b + x;
  c = c + EXP_ONE.f;

  pow2n.u = (int)a;
  pow2n.u = pow2n.u << 23;
  pow2n.u = pow2n.u + EXP_ONE.u;
  c = c * pow2n.f;
  return c;
}

ATTR_NOINLINE void sage2_exp_ps_ref1(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = sage2_exp_ss_ref1(x[i]);
  }
}

ATTR_NOINLINE float sage2_exp_ss_ref2(float x) {
  static const float EXP_HI = 88.02969193111305f;
  static const float EXP_LO = -88.02969193111305f;
  static const float EXP_INV_LOG2E = 1.44269504088896341f;
  static const float EXP_P0 = 8.2886153525e-14f;
  static const float EXP_P1 = 7.7822959126e-02f;
  static const float EXP_P2 = 2.2586729288e-01f;
  static const float EXP_P3 = 6.9617327373e-01f;
  static const float EXP_P4 = 9.9986347636e-01f;
  static const union {
    float f;
    uint32_t u;
  } EXP_ONE = {1.0f};

  float a, fa, b, c;
  union {
    float f;
    uint32_t u;
  } pow2n;

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
  return c;
}

ATTR_NOINLINE void sage2_exp_ps_ref2(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = sage2_exp_ss_ref2(x[i]);
  }
}

ATTR_NOINLINE void eigen_exp_ps(uint64_t n, const float* x, float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv.exp();
}

void Test() {
  std::default_random_engine engine;
  float x;
  float y1, y2, y3, y4, y5, y6, y7;

  for (int i = 0; i < 10000; ++i) {
    rand(engine, &x, -80, 80);

    y1 = libc_exp_ss(x);

    y2 = sage2_exp_ss_ref1(x);
    CHECK_EQUAL(y1, y2);

    y3 = sage2_exp_ss_ref2(x);
    CHECK_EQUAL(y1, y3);

    y4 = sage2_exp_ss1(x);
    CHECK_EQUAL(y1, y4);

    y5 = sage2_exp_ss2(x);
    CHECK_EQUAL(y1, y5);

    eigen_exp_ps(1, &x, &y6);
    CHECK_EQUAL(y1, y6);  // NOLINT

    if (mkl->vsExp) {
      mkl->vsExp(1, &x, &y7);
      CHECK_EQUAL(y1, y7);
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
  std::vector<float> y6(n);
  std::vector<float> y7(n);
  rand(engine, &x, -80, 80);

  libc_exp_ps(n, x.data(), y1.data());

  sage2_exp_ps_ref1(n, x.data(), y2.data());
  CHECK_EQUAL(y1, y2);

  sage2_exp_ps_ref2(n, x.data(), y3.data());
  CHECK_EQUAL(y1, y3);

  sage2_exp_ps1(n, x.data(), y4.data());
  CHECK_EQUAL(y1, y4);

  sage2_exp_ps2(n, x.data(), y5.data());
  CHECK_EQUAL(y1, y5);

  eigen_exp_ps(n, x.data(), y6.data());
  CHECK_EQUAL(y1, y6);

  if (mkl->vsExp) {
    mkl->vsExp(n, x.data(), y7.data());
    CHECK_EQUAL(y1, y7);
  }
}

void TestSpecial() {
  CHECK_NOT_NAN_INF(sage2_exp_ss(FLOAT_QUIET_NAN));
  CHECK_NOT_NAN_INF(sage2_exp_ss(FLOAT_SIGNALING_NAN));
  CHECK_NOT_NAN_INF(sage2_exp_ss(FLOAT_INF));
  CHECK_NOT_NAN_INF(sage2_exp_ss(FLOAT_NINF));
  CHECK_NOT_NAN_INF(sage2_exp_ss(100.0f));
  CHECK_NOT_NAN_INF(sage2_exp_ss(10.0f));
  CHECK_NOT_NAN_INF(sage2_exp_ss(1.0f));
  CHECK_NOT_NAN_INF(sage2_exp_ss(-100.0f));
  CHECK_NOT_NAN_INF(sage2_exp_ss(-10.0f));
  CHECK_NOT_NAN_INF(sage2_exp_ss(-1.0f));
  CHECK_NOT_NAN_INF(sage2_exp_ss(0.0f));

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
  sage2_exp_ps(x.size(), x.data(), y.data());
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
  double mops[7] = {0};
  mops[0] = Benchmark(libc_exp_ps, n, m);
  mops[1] = Benchmark(sage2_exp_ps_ref1, n, m);
  mops[2] = Benchmark(sage2_exp_ps_ref2, n, m);
  mops[3] = Benchmark(sage2_exp_ps1, n, m);
  mops[4] = Benchmark(sage2_exp_ps2, n, m);
  mops[5] = Benchmark(eigen_exp_ps, n, m);
  if (mkl->vsExp) {
    mops[6] = Benchmark(mkl->vsExp, n, m);
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
    PrintHeader1(7, "exp");
    PrintHeader2(7, "Mop/s");
    PrintHeader3("libc", "ref1", "ref2", "opt1", "opt2", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
