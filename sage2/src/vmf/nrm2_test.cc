// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/cblas_adapter.h>
#include <sage2/vmf.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

const auto* cblas = sage2_cblas_cblas();
const auto* mkl = sage2_cblas_mkl();
const auto* openblas = sage2_cblas_openblas();

ATTR_NOINLINE float sage2_nrm2_ps_ref(uint64_t n, const float* x) {
  float sum = 0;
  for (uint64_t i = 0; i < n; ++i) {
    sum += x[i] * x[i];
  }
  return sqrtf(sum);
}

ATTR_NOINLINE float eigen_nrm2_ps(uint64_t n, const float* x) {
  auto xv = make_eigen_arrayxf_view(x, n);
  return sqrtf(xv.square().sum());
}

ATTR_NOINLINE float cblas_cblas_nrm2_ps(uint64_t n, const float* x) {
  return cblas->cblas_snrm2((int)n, x, 1);
}

ATTR_NOINLINE float cblas_mkl_nrm2_ps(uint64_t n, const float* x) {
  return mkl->cblas_snrm2((int)n, x, 1);
}

ATTR_NOINLINE float cblas_openblas_nrm2_ps(uint64_t n, const float* x) {
  return openblas->cblas_snrm2((int)n, x, 1);
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> x(n);
  randn(engine, &x);

  float nrm1 = sage2_nrm2_ps_ref(n, x.data());

  float nrm2 = sage2_nrm2_ps(n, x.data());
  CHECK_EQUAL(nrm1, nrm2);

  float nrm3 = eigen_nrm2_ps(n, x.data());
  CHECK_EQUAL(nrm1, nrm3);

  if (cblas->cblas_snrm2) {
    float nrm4 = cblas_cblas_nrm2_ps(n, x.data());
    CHECK_EQUAL(nrm1, nrm4);
  }

  if (mkl->cblas_snrm2) {
    float nrm5 = cblas_mkl_nrm2_ps(n, x.data());
    CHECK_EQUAL(nrm1, nrm5);
  }

  if (openblas->cblas_snrm2) {
    float nrm6 = cblas_openblas_nrm2_ps(n, x.data());
    CHECK_EQUAL(nrm1, nrm6);
  }
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
  return GetGFLOPS(3.0 * m * n);
}

void Benchmark(int n) {
  int m = 100000000 / n;
  double gflops[6] = {0};
  gflops[0] = Benchmark(sage2_nrm2_ps_ref, n, m);
  gflops[1] = Benchmark(sage2_nrm2_ps, n, m);
  gflops[2] = Benchmark(eigen_nrm2_ps, n, m);
  if (cblas->cblas_snrm2) {
    gflops[3] = Benchmark(cblas_cblas_nrm2_ps, n, m);
  }
  if (mkl->cblas_snrm2) {
    gflops[4] = Benchmark(cblas_mkl_nrm2_ps, n, m);
  }
  if (openblas->cblas_snrm2) {
    gflops[5] = Benchmark(cblas_openblas_nrm2_ps, n, m);
  }
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
    PrintHeader1(6, "nrm2");
    PrintHeader2(6, "GFLOPS");
    PrintHeader3("ref", "opt", "eigen", cblas->vendor, mkl->vendor,
                 openblas->vendor);
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
