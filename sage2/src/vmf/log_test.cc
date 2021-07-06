// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//
// Reference:
// http://software-lisc.fbk.eu/avx_mathfun
//

#include <sage2/cblas_adapter.h>
#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

const auto* mkl = sage2_cblas_mkl();

ATTR_NOINLINE float libc_log_ss(float x) { return logf(x); }

ATTR_NOINLINE void libc_log_ps(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = logf(x[i]);
  }
}

ATTR_NOINLINE float sage2_log_ss_ref(float x) {
  static const float LOG_LO = 1.0e-6f;
  static const float LOG_C0 = 0.7071067811865476f;
  static const float LOG_C1 = 0.6931471824645996f;
  static const float LOG_P0 = +7.0376836292e-2f;
  static const float LOG_P1 = -1.1514610310e-1f;
  static const float LOG_P2 = +1.1676998740e-1f;
  static const float LOG_P3 = -1.2420140846e-1f;
  static const float LOG_P4 = +1.4249322787e-1f;
  static const float LOG_P5 = -1.6668057665e-1f;
  static const float LOG_P6 = +2.0000714765e-1f;
  static const float LOG_P7 = -2.4999993993e-1f;
  static const float LOG_P8 = +3.3333331174e-1f;
  static const uint32_t LOG_INV_MANTISSA_MASK = ~0x7f800000;
  static const uint32_t LOG_EXP_OFFSET = -127;
  static const float LOG_ONE = 1.0f;
  static const union {
    float f;
    uint32_t u;
  } LOG_0P5 = {0.5f};

  union {
    float f;
    uint32_t u;
  } a;
  float dx, e, de, y, z;

  x = (x < LOG_LO) ? LOG_LO : x;

  a.f = x;
  a.u = a.u >> 23;
  a.u = a.u + LOG_EXP_OFFSET;
  a.f = (float)(int)a.u;
  a.f = a.f + LOG_ONE;
  e = a.f;

  a.f = x;
  a.u = a.u & LOG_INV_MANTISSA_MASK;
  a.u = a.u | LOG_0P5.u;
  x = a.f;

  if (x < LOG_C0) {
    dx = x;
    de = LOG_ONE;
  } else {
    dx = 0;
    de = 0;
  }
  x = x + dx - LOG_ONE;
  e = e - de;

  y = LOG_P0;
  y = y * x + LOG_P1;
  y = y * x + LOG_P2;
  y = y * x + LOG_P3;
  y = y * x + LOG_P4;
  y = y * x + LOG_P5;
  y = y * x + LOG_P6;
  y = y * x + LOG_P7;
  y = y * x + LOG_P8;

  z = x * x;
  y = y * x - LOG_0P5.f;
  x = e * LOG_C1 + x;
  x = y * z + x;
  return x;
}

ATTR_NOINLINE void sage2_log_ps_ref(uint64_t n, const float* x, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = sage2_log_ss_ref(x[i]);
  }
}

ATTR_NOINLINE void eigen_log_ps(uint64_t n, const float* x, float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv.log();
}

void Test() {
  std::default_random_engine engine;
  float x;
  float y1, y2, y3, y4, y5;

  for (int i = 0; i < 10000; ++i) {
    rand(engine, &x, 0.1f, 100.0f);

    y1 = libc_log_ss(x);

    y2 = sage2_log_ss_ref(x);
    CHECK_EQUAL(y1, y2);

    y3 = sage2_log_ss(x);
    CHECK_EQUAL(y1, y3);

    eigen_log_ps(1, &x, &y4);
    CHECK_EQUAL(y1, y4);  // NOLINT

    if (mkl->vsLn) {
      mkl->vsLn(1, &x, &y5);
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
  rand(engine, &x, 0.1f, 100.0f);

  libc_log_ps(n, x.data(), y1.data());

  sage2_log_ps_ref(n, x.data(), y2.data());
  CHECK_EQUAL(y1, y2);

  sage2_log_ps(n, x.data(), y3.data());
  CHECK_EQUAL(y1, y3);

  eigen_log_ps(n, x.data(), y4.data());
  CHECK_EQUAL(y1, y4);

  if (mkl->vsLn) {
    mkl->vsLn(n, x.data(), y5.data());
    CHECK_EQUAL(y1, y5);
  }
}

void TestSpecial() {
  CHECK_NOT_NAN_INF(sage2_log_ss(FLOAT_QUIET_NAN));
  CHECK_NOT_NAN_INF(sage2_log_ss(FLOAT_SIGNALING_NAN));
  CHECK_NOT_NAN_INF(sage2_log_ss(FLOAT_INF));
  CHECK_NOT_NAN_INF(sage2_log_ss(FLOAT_NINF));
  CHECK_NOT_NAN_INF(sage2_log_ss(100.0f));
  CHECK_NOT_NAN_INF(sage2_log_ss(10.0f));
  CHECK_NOT_NAN_INF(sage2_log_ss(1.0f));
  CHECK_NOT_NAN_INF(sage2_log_ss(-100.0f));
  CHECK_NOT_NAN_INF(sage2_log_ss(-10.0f));
  CHECK_NOT_NAN_INF(sage2_log_ss(-1.0f));
  CHECK_NOT_NAN_INF(sage2_log_ss(0.0f));

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
  rand(engine, &x, 0.1f, 100.0f);

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
  mops[0] = Benchmark(libc_log_ps, n, m);
  mops[1] = Benchmark(sage2_log_ps_ref, n, m);
  mops[2] = Benchmark(sage2_log_ps, n, m);
  mops[3] = Benchmark(eigen_log_ps, n, m);
  if (mkl->vsLn) {
    mops[4] = Benchmark(mkl->vsLn, n, m);
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
    PrintHeader1(5, "log");
    PrintHeader2(5, "Mop/s");
    PrintHeader3("libc", "ref", "opt", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
