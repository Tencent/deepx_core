// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_ada_grad_update_ss_ref(
    const sage2_ada_grad_config_s* config, float g, float* w, float* n) {
  float new_n = *n + g * g;
  float new_w = *w - g / sqrtf(new_n + config->beta) * config->alpha;
  *w = new_w;
  *n = new_n;
}

ATTR_NOINLINE void sage2_ada_grad_update_ps_ref(
    const sage2_ada_grad_config_s* config, uint64_t _n, const float* g,
    float* w, float* n) {
  for (uint64_t i = 0; i < _n; ++i) {
    float new_n = n[i] + g[i] * g[i];
    float new_w = w[i] - g[i] / sqrtf(new_n + config->beta) * config->alpha;
    w[i] = new_w;
    n[i] = new_n;
  }
}

ATTR_NOINLINE void eigen_ada_grad_update_ps(
    const sage2_ada_grad_config_s* config, uint64_t _n, const float* g,
    float* w, float* n) {
  auto gv = make_eigen_arrayxf_view(g, _n);
  auto wv = make_eigen_arrayxf_view(w, _n);
  auto nv = make_eigen_arrayxf_view(n, _n);
  nv += gv * gv;
  wv -= gv / (nv + config->beta).sqrt() * config->alpha;
}

void Test() {
  std::default_random_engine engine;
  float g;
  float w1 = 0, n1 = 0;
  float w2 = 0, n2 = 0;
  float w3 = 0, n3 = 0;
  sage2_ada_grad_config_s config;
  sage2_ada_grad_config_s_default(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_ada_grad_update_ss_ref(&config, g, &w1, &n1);

    sage2_ada_grad_update_ss(&config, g, &w2, &n2);
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(n1, n2);
    w2 = w1;
    n2 = n1;

    eigen_ada_grad_update_ps(&config, 1, &g, &w3, &n3);
    CHECK_EQUAL(w1, w3);  // NOLINT
    CHECK_EQUAL(n1, n3);  // NOLINT
    w3 = w1;
    n3 = n1;
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> g(n);
  std::vector<float> w1(n), n1(n);
  std::vector<float> w2(n), n2(n);
  std::vector<float> w3(n), n3(n);
  sage2_ada_grad_config_s config;
  sage2_ada_grad_config_s_default(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_ada_grad_update_ps_ref(&config, n, g.data(), w1.data(), n1.data());

    sage2_ada_grad_update_ps(&config, n, g.data(), w2.data(), n2.data());
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(n1, n2);
    w2 = w1;
    n2 = n1;

    eigen_ada_grad_update_ps(&config, n, g.data(), w3.data(), n3.data());
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
  sage2_ada_grad_config_s config;
  randn(engine, &g);
  sage2_ada_grad_config_s_default(&config);

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
  mops[0] = Benchmark(sage2_ada_grad_update_ps_ref, n, m);
  mops[1] = Benchmark(sage2_ada_grad_update_ps, n, m);
  mops[2] = Benchmark(eigen_ada_grad_update_ps, n, m);
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
    PrintHeader1(3, "ada_grad");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
