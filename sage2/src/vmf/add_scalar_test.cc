// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_add_scalar_ps_ref(uint64_t n, const float* x,
                                           float alpha, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = x[i] + alpha;
  }
}

ATTR_NOINLINE void sage2_sub_scalar_ps_ref(uint64_t n, const float* x,
                                           float alpha, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = x[i] - alpha;
  }
}

ATTR_NOINLINE void sage2_mul_scalar_ps_ref(uint64_t n, const float* x,
                                           float alpha, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = x[i] * alpha;
  }
}

ATTR_NOINLINE void sage2_div_scalar_ps_ref(uint64_t n, const float* x,
                                           float alpha, float* y) {
  for (uint64_t i = 0; i < n; ++i) {
    y[i] = x[i] / alpha;
  }
}

ATTR_NOINLINE void eigen_add_scalar_ps(uint64_t n, const float* x, float alpha,
                                       float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv + alpha;
}

ATTR_NOINLINE void eigen_sub_scalar_ps(uint64_t n, const float* x, float alpha,
                                       float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv - alpha;
}

ATTR_NOINLINE void eigen_mul_scalar_ps(uint64_t n, const float* x, float alpha,
                                       float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv * alpha;
}

ATTR_NOINLINE void eigen_div_scalar_ps(uint64_t n, const float* x, float alpha,
                                       float* y) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  yv = xv / alpha;
}

struct FuncMeta {
  decltype(&sage2_add_scalar_ps) ref, opt, eigen;
};

void Test(const FuncMeta& meta, int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  float alpha = 0;
  std::vector<float> y1(n);
  std::vector<float> y2(n);
  std::vector<float> y3(n);
  randn(engine, &x);
  if (meta.ref == &sage2_div_scalar_ps_ref) {
    rand(engine, &alpha, 0.1f, 100.0f);
  } else {
    randn(engine, &alpha);
  }

  meta.ref(n, x.data(), alpha, y1.data());

  meta.opt(n, x.data(), alpha, y2.data());
  CHECK_EQUAL(y1, y2);

  meta.eigen(n, x.data(), alpha, y3.data());
  CHECK_EQUAL(y1, y3);
}

void Benchmark(const FuncMeta& meta, int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  float alpha = 0;
  std::vector<float> y(n);
  randn(engine, &x);
  if (meta.ref == &sage2_div_scalar_ps_ref) {
    rand(engine, &alpha, 0.1f, 100.0f);
  } else {
    randn(engine, &alpha);
  }

  int m = 100000000 / n;
  double gflops[3] = {0};

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    meta.ref(n, x.data(), alpha, y.data());
  }
  EndTimer();
  gflops[0] = GetGFLOPS(1.0 * m * n);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    meta.opt(n, x.data(), alpha, y.data());
  }
  EndTimer();
  gflops[1] = GetGFLOPS(1.0 * m * n);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    meta.eigen(n, x.data(), alpha, y.data());
  }
  EndTimer();
  gflops[2] = GetGFLOPS(1.0 * m * n);

  char buf[64];
  snprintf(buf, sizeof(buf), "%8d", n);
  PrintContent(buf, gflops);
}

int main(int argc, char** argv) {
  FuncMeta add_meta = {&sage2_add_scalar_ps_ref, &sage2_add_scalar_ps,
                       &eigen_add_scalar_ps};
  FuncMeta sub_meta = {&sage2_sub_scalar_ps_ref, &sage2_sub_scalar_ps,
                       &eigen_sub_scalar_ps};
  FuncMeta mul_meta = {&sage2_mul_scalar_ps_ref, &sage2_mul_scalar_ps,
                       &eigen_mul_scalar_ps};
  FuncMeta div_meta = {&sage2_div_scalar_ps_ref, &sage2_div_scalar_ps,
                       &eigen_div_scalar_ps};

  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    for (int i = 1; i < 1000; ++i) {
      Test(add_meta, i);
    }
    for (int i = 1; i < 1000; ++i) {
      Test(sub_meta, i);
    }
    for (int i = 1; i < 1000; ++i) {
      Test(mul_meta, i);
    }
    for (int i = 1; i < 1000; ++i) {
      Test(div_meta, i);
    }
  }

  if (action & 2) {
    PrintHeader1(3, "add_scalar");
    PrintHeader2(3, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(add_meta, n);
    }
    PrintHeader1(3, "sub_scalar");
    PrintHeader2(3, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(sub_meta, n);
    }
    PrintHeader1(3, "mul_scalar");
    PrintHeader2(3, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(mul_meta, n);
    }
    PrintHeader1(3, "div_scalar");
    PrintHeader2(3, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(div_meta, n);
    }
  }
  return 0;
}
