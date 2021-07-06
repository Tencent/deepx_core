// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_grad_clip_20_ss_ref(float* g) {
  if (*g > 20) {
    *g = 20;
  } else if (*g < -20) {
    *g = -20;
  }
}

ATTR_NOINLINE void sage2_grad_clip_20_ps_ref(uint64_t n, float* g) {
  for (uint64_t i = 0; i < n; ++i) {
    if (g[i] > 20) {
      g[i] = 20;
    } else if (g[i] < -20) {
      g[i] = -20;
    }
  }
}

ATTR_NOINLINE void eigen_grad_clip_20_ps(uint64_t n, float* g) {
  auto gv = make_eigen_arrayxf_view(g, n);
  gv = (gv > 20).select(eigen_arrayxf_view_t::Constant(n, 20), gv);
  gv = (gv < -20).select(eigen_arrayxf_view_t::Constant(n, -20), gv);
}

void Test() {
  std::default_random_engine engine;
  float g, g1, g2, g3;

  for (int i = 0; i < 10000; ++i) {
    rand(engine, &g, -80, 80);
    g1 = g;
    sage2_grad_clip_20_ss_ref(&g1);

    g2 = g;
    sage2_grad_clip_20_ss(&g2);
    CHECK_EQUAL(g1, g2);

    g3 = g;
    eigen_grad_clip_20_ps(1, &g3);
    CHECK_EQUAL(g1, g3);
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> g(n);
  std::vector<float> g1(n);
  std::vector<float> g2(n);
  std::vector<float> g3(n);

  for (int i = 0; i < 10000; ++i) {
    rand(engine, &g, -80, 80);
    g1 = g;
    sage2_grad_clip_20_ps_ref(n, g1.data());

    g2 = g;
    sage2_grad_clip_20_ps(n, g2.data());
    CHECK_EQUAL(g1, g2);

    g3 = g;
    eigen_grad_clip_20_ps(n, g3.data());
    CHECK_EQUAL(g1, g3);
  }
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> g(n * m);
  float* g_ptr;
  rand(engine, &g, -80, 80);

  BeginTimer();
  g_ptr = g.data();
  for (int i = 0; i < m; ++i) {
    func(n, g_ptr);
    g_ptr += n;
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark(int n) {
  int m = 20000000 / n;
  double mops[3] = {0};
  mops[0] = Benchmark(sage2_grad_clip_20_ps_ref, n, m);
  mops[1] = Benchmark(sage2_grad_clip_20_ps, n, m);
  mops[2] = Benchmark(eigen_grad_clip_20_ps, n, m);
  PrintContent(n, mops);
}

int main(int argc, char** argv) {
  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    Test();
    for (int n : GetN()) {
      Test(n);
    }
  }

  if (action & 2) {
    PrintHeader1(3, "grad_clip");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
