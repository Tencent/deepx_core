// Copyright 2021 the deepx authors.
// Author: simplefan (simplefan@tencent.com)
//

#include <sage2/tf_optimizer.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_tf_ada_grad_v1_update_ps_ref(
    const sage2_tf_ada_grad_v1_config_s* config, uint64_t _n, const float* g,
    float* w, float* n) {
  for (uint64_t i = 0; i < _n; ++i) {
    float new_n = n[i] + g[i] * g[i];
    float new_w = w[i] - g[i] / sqrtf(new_n) * config->alpha;
    w[i] = new_w;
    n[i] = new_n;
  }
}

ATTR_NOINLINE void eigen_tf_ada_grad_v1_update_ps(
    const sage2_tf_ada_grad_v1_config_s* config, uint64_t _n, const float* g,
    float* w, float* n) {
  auto gv = make_eigen_arrayxf_view(g, _n);
  auto wv = make_eigen_arrayxf_view(w, _n);
  auto nv = make_eigen_arrayxf_view(n, _n);
  nv += gv * gv;
  wv -= gv / nv.sqrt() * config->alpha;
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> g(n);
  std::vector<float> w1(n), n1(n);
  std::vector<float> w2(n), n2(n);
  std::vector<float> w3(n), n3(n);
  sage2_tf_ada_grad_v1_config_s config;
  config.alpha = 0.01f;

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_tf_ada_grad_v1_update_ps_ref(&config, n, g.data(), w1.data(),
                                       n1.data());

    sage2_tf_ada_grad_v1_update_ps(&config, n, g.data(), w2.data(), n2.data());
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(n1, n2);
    w2 = w1;
    n2 = n1;

    eigen_tf_ada_grad_v1_update_ps(&config, n, g.data(), w3.data(), n3.data());
    CHECK_EQUAL(w1, w3);
    CHECK_EQUAL(n1, n3);
    w3 = w1;
    n3 = n1;
  }
}

template <class Func>
double Benchmark(Func&& func, int _n, int m) {
  std::default_random_engine engine;
  std::vector<float> g(_n * m);
  float* g_ptr;
  std::vector<float> w(_n), n(_n);
  sage2_tf_ada_grad_v1_config_s config;
  randn(engine, &g);
  config.alpha = 0.01f;

  BeginTimer();
  g_ptr = g.data();
  for (int i = 0; i < m; ++i) {
    func(&config, _n, g_ptr, w.data(), n.data());
    g_ptr += _n;
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark(int n) {
  int m = 20000000 / n;
  double mops[3] = {0};
  mops[0] = Benchmark(sage2_tf_ada_grad_v1_update_ps_ref, n, m);
  mops[1] = Benchmark(sage2_tf_ada_grad_v1_update_ps, n, m);
  mops[2] = Benchmark(eigen_tf_ada_grad_v1_update_ps, n, m);
  PrintContent(n, mops);
}

int main(int argc, char** argv) {
  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    for (int n : GetN()) {
      Test(n);
    }
  }

  if (action & 2) {
    PrintHeader1(3, "tf_ada_grad_v1");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
