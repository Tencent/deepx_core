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

ATTR_NOINLINE void sage2_add_ps_ref(uint64_t n, const float* x, const float* y,
                                    float* z) {
  for (uint64_t i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

ATTR_NOINLINE void sage2_sub_ps_ref(uint64_t n, const float* x, const float* y,
                                    float* z) {
  for (uint64_t i = 0; i < n; ++i) {
    z[i] = x[i] - y[i];
  }
}

ATTR_NOINLINE void sage2_mul_ps_ref(uint64_t n, const float* x, const float* y,
                                    float* z) {
  for (uint64_t i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

ATTR_NOINLINE void sage2_div_ps_ref(uint64_t n, const float* x, const float* y,
                                    float* z) {
  for (uint64_t i = 0; i < n; ++i) {
    z[i] = x[i] / y[i];
  }
}

ATTR_NOINLINE void eigen_add_ps(uint64_t n, const float* x, const float* y,
                                float* z) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  auto zv = make_eigen_arrayxf_view(z, n);
  zv = xv + yv;
}

ATTR_NOINLINE void eigen_sub_ps(uint64_t n, const float* x, const float* y,
                                float* z) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  auto zv = make_eigen_arrayxf_view(z, n);
  zv = xv - yv;
}

ATTR_NOINLINE void eigen_mul_ps(uint64_t n, const float* x, const float* y,
                                float* z) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  auto zv = make_eigen_arrayxf_view(z, n);
  zv = xv * yv;
}

ATTR_NOINLINE void eigen_div_ps(uint64_t n, const float* x, const float* y,
                                float* z) {
  auto xv = make_eigen_arrayxf_view(x, n);
  auto yv = make_eigen_arrayxf_view(y, n);
  auto zv = make_eigen_arrayxf_view(z, n);
  zv = xv / yv;
}

struct FuncMeta {
  decltype(&sage2_add_ps) ref, opt, eigen;
  decltype(mkl->vsAdd) _mkl;
};

void Test(const FuncMeta& meta, int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  std::vector<float> z1(n);
  std::vector<float> z2(n);
  std::vector<float> z3(n);
  std::vector<float> z4(n);
  randn(engine, &x);
  if (meta.ref == &sage2_div_ps_ref) {
    rand(engine, &y, 0.1f, 100.0f);
  } else {
    randn(engine, &y);
  }

  meta.ref(n, x.data(), y.data(), z1.data());

  meta.opt(n, x.data(), y.data(), z2.data());
  CHECK_EQUAL(z1, z2);

  meta.eigen(n, x.data(), y.data(), z3.data());
  CHECK_EQUAL(z1, z3);

  if (meta._mkl) {
    meta._mkl(n, x.data(), y.data(), z4.data());
    CHECK_EQUAL(z1, z4);
  }
}

void Benchmark(const FuncMeta& meta, int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  std::vector<float> y(n);
  std::vector<float> z(n);
  randn(engine, &x);
  if (meta.ref == &sage2_div_ps_ref) {
    rand(engine, &y, 0.1f, 100.0f);
  } else {
    randn(engine, &y);
  }

  int m = 100000000 / n;
  double gflops[4] = {0};

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    meta.ref(n, x.data(), y.data(), z.data());
  }
  EndTimer();
  gflops[0] = GetGFLOPS(1.0 * m * n);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    meta.opt(n, x.data(), y.data(), z.data());
  }
  EndTimer();
  gflops[1] = GetGFLOPS(1.0 * m * n);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    meta.eigen(n, x.data(), y.data(), z.data());
  }
  EndTimer();
  gflops[2] = GetGFLOPS(1.0 * m * n);

  if (meta._mkl) {
    BeginTimer();
    for (int i = 0; i < m; ++i) {
      meta._mkl(n, x.data(), y.data(), z.data());
    }
    EndTimer();
    gflops[3] = GetGFLOPS(1.0 * m * n);
  }

  char buf[64];
  snprintf(buf, sizeof(buf), "%8d", n);
  PrintContent(buf, gflops);
}

int main(int argc, char** argv) {
  FuncMeta add_meta = {&sage2_add_ps_ref, &sage2_add_ps, &eigen_add_ps,
                       mkl->vsAdd};
  FuncMeta sub_meta = {&sage2_sub_ps_ref, &sage2_sub_ps, &eigen_sub_ps,
                       mkl->vsSub};
  FuncMeta mul_meta = {&sage2_mul_ps_ref, &sage2_mul_ps, &eigen_mul_ps,
                       mkl->vsMul};
  FuncMeta div_meta = {&sage2_div_ps_ref, &sage2_div_ps, &eigen_div_ps,
                       mkl->vsDiv};

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
    PrintHeader1(4, "add");
    PrintHeader2(4, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(add_meta, n);
    }
    PrintHeader1(4, "sub");
    PrintHeader2(4, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(sub_meta, n);
    }
    PrintHeader1(4, "mul");
    PrintHeader2(4, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(mul_meta, n);
    }
    PrintHeader1(4, "div");
    PrintHeader2(4, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen", mkl->vendor);
    for (int n : GetLargeN()) {
      Benchmark(div_meta, n);
    }
  }
  return 0;
}
