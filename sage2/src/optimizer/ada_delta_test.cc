// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_ada_delta_update_ss_ref(
    const sage2_ada_delta_config_s* config, float g, float* w, float* n,
    float* deltaw) {
  float new_n = config->rho * *n + config->one_sub_rho * g * g;
  float a = sqrtf(*deltaw + config->beta) / sqrtf(new_n + config->beta) * g;
  float new_deltaw = config->rho * *deltaw + config->one_sub_rho * a * a;
  float new_w = *w - config->alpha * a;
  *w = new_w;
  *n = new_n;
  *deltaw = new_deltaw;
}

ATTR_NOINLINE void sage2_ada_delta_update_ps_ref(
    const sage2_ada_delta_config_s* config, uint64_t _n, const float* g,
    float* w, float* n, float* deltaw) {
  for (uint64_t i = 0; i < _n; ++i) {
    float new_n = config->rho * n[i] + config->one_sub_rho * g[i] * g[i];
    float a =
        sqrtf(deltaw[i] + config->beta) / sqrtf(new_n + config->beta) * g[i];
    float new_deltaw = config->rho * deltaw[i] + config->one_sub_rho * a * a;
    float new_w = w[i] - config->alpha * a;
    w[i] = new_w;
    n[i] = new_n;
    deltaw[i] = new_deltaw;
  }
}

ATTR_NOINLINE void eigen_ada_delta_update_ps(
    const sage2_ada_delta_config_s* config, uint64_t _n, const float* g,
    float* w, float* n, float* deltaw) {
  auto gv = make_eigen_arrayxf_view(g, _n);
  auto wv = make_eigen_arrayxf_view(w, _n);
  auto nv = make_eigen_arrayxf_view(n, _n);
  auto deltawv = make_eigen_arrayxf_view(deltaw, _n);
  nv *= config->rho;
  nv += config->one_sub_rho * gv * gv;
  auto a = ((deltawv + config->beta).sqrt() / (nv + config->beta).sqrt() * gv)
               .eval();
  deltawv *= config->rho;
  deltawv += (config->one_sub_rho * a * a);
  wv -= config->alpha * a;
}

void Test() {
  std::default_random_engine engine;
  float g;
  float w1 = 0, n1 = 0, deltaw1 = 0;
  float w2 = 0, n2 = 0, deltaw2 = 0;
  float w3 = 0, n3 = 0, deltaw3 = 0;
  sage2_ada_delta_config_s config;
  sage2_ada_delta_config_s_default(&config);
  sage2_ada_delta_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_ada_delta_update_ss_ref(&config, g, &w1, &n1, &deltaw1);

    sage2_ada_delta_update_ss(&config, g, &w2, &n2, &deltaw2);
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(n1, n2);
    CHECK_EQUAL(deltaw1, deltaw2);
    w2 = w1;
    n2 = n1;
    deltaw2 = deltaw1;

    eigen_ada_delta_update_ps(&config, 1, &g, &w3, &n3, &deltaw3);
    CHECK_EQUAL(w1, w3);            // NOLINT
    CHECK_EQUAL(n1, n3);            // NOLINT
    CHECK_EQUAL(deltaw1, deltaw3);  // NOLINT
    w3 = w1;
    n3 = n1;
    deltaw3 = deltaw1;
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> g(n);
  std::vector<float> w1(n), n1(n), deltaw1(n);
  std::vector<float> w2(n), n2(n), deltaw2(n);
  std::vector<float> w3(n), n3(n), deltaw3(n);
  sage2_ada_delta_config_s config;
  sage2_ada_delta_config_s_default(&config);
  sage2_ada_delta_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_ada_delta_update_ps_ref(&config, n, g.data(), w1.data(), n1.data(),
                                  deltaw1.data());

    sage2_ada_delta_update_ps(&config, n, g.data(), w2.data(), n2.data(),
                              deltaw2.data());
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(n1, n2);
    CHECK_EQUAL(deltaw1, deltaw2);
    w2 = w1;
    n2 = n1;
    deltaw2 = deltaw1;

    eigen_ada_delta_update_ps(&config, n, g.data(), w3.data(), n3.data(),
                              deltaw3.data());
    CHECK_EQUAL(w1, w3);
    CHECK_EQUAL(n1, n3);
    CHECK_EQUAL(deltaw1, deltaw3);
    w3 = w1;
    n3 = n1;
    deltaw3 = deltaw1;
  }
}

template <class Func>
double Benchmark(Func&& func, int _n, int m) {
  std::default_random_engine engine;
  std::vector<float> g(_n * m);
  float* g_ptr;
  std::vector<float> w(_n), n(_n), deltaw(_n);
  sage2_ada_delta_config_s config;
  randn(engine, &g);
  sage2_ada_delta_config_s_default(&config);
  sage2_ada_delta_config_s_init(&config);

  BeginTimer();
  g_ptr = g.data();
  for (int i = 0; i < m; ++i) {
    func(&config, _n, g_ptr, w.data(), n.data(), deltaw.data());
    g_ptr += _n;
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark(int n) {
  int m = 20000000 / n;
  double mops[3] = {0};
  mops[0] = Benchmark(sage2_ada_delta_update_ps_ref, n, m);
  mops[1] = Benchmark(sage2_ada_delta_update_ps, n, m);
  mops[2] = Benchmark(eigen_ada_delta_update_ps, n, m);
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
    PrintHeader1(3, "ada_delta");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
