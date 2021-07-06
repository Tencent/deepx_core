// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_rms_prop_update_ss_ref(
    const sage2_rms_prop_config_s* config, float g, float* w, float* v) {
  float new_v = config->rho * *v + config->one_sub_rho * g * g;
  float new_w = *w - g / sqrtf(new_v + config->beta) * config->alpha;
  *w = new_w;
  *v = new_v;
}

ATTR_NOINLINE void sage2_rms_prop_update_ps_ref(
    const sage2_rms_prop_config_s* config, uint64_t n, const float* g, float* w,
    float* v) {
  for (uint64_t i = 0; i < n; ++i) {
    float new_v = config->rho * v[i] + config->one_sub_rho * g[i] * g[i];
    float new_w = w[i] - g[i] / sqrtf(new_v + config->beta) * config->alpha;
    w[i] = new_w;
    v[i] = new_v;
  }
}

ATTR_NOINLINE void eigen_rms_prop_update_ps(
    const sage2_rms_prop_config_s* config, uint64_t n, const float* g, float* w,
    float* v) {
  auto gv = make_eigen_arrayxf_view(g, n);
  auto wv = make_eigen_arrayxf_view(w, n);
  auto vv = make_eigen_arrayxf_view(v, n);
  vv *= config->rho;
  vv += config->one_sub_rho * gv * gv;
  wv -= gv / (vv + config->beta).sqrt() * config->alpha;
}

void Test() {
  std::default_random_engine engine;
  float g;
  float w1 = 0, v1 = 0;
  float w2 = 0, v2 = 0;
  float w3 = 0, v3 = 0;
  sage2_rms_prop_config_s config;
  sage2_rms_prop_config_s_default(&config);
  sage2_rms_prop_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_rms_prop_update_ss_ref(&config, g, &w1, &v1);

    sage2_rms_prop_update_ss(&config, g, &w2, &v2);
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(v1, v2);
    w2 = w1;
    v2 = v1;

    eigen_rms_prop_update_ps(&config, 1, &g, &w3, &v3);
    CHECK_EQUAL(w1, w3);  // NOLINT
    CHECK_EQUAL(v1, v3);  // NOLINT
    w3 = w1;
    v3 = v1;
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> g(n);
  std::vector<float> w1(n), v1(n);
  std::vector<float> w2(n), v2(n);
  std::vector<float> w3(n), v3(n);
  sage2_rms_prop_config_s config;
  sage2_rms_prop_config_s_default(&config);
  sage2_rms_prop_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_rms_prop_update_ps_ref(&config, n, g.data(), w1.data(), v1.data());

    sage2_rms_prop_update_ps(&config, n, g.data(), w2.data(), v2.data());
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(v1, v2);
    w2 = w1;
    v2 = v1;

    eigen_rms_prop_update_ps(&config, n, g.data(), w3.data(), v3.data());
    CHECK_EQUAL(w1, w3);
    CHECK_EQUAL(v1, v3);
    w3 = w1;
    v3 = v1;
  }
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> g(n * m);
  float* g_ptr;
  std::vector<float> w(n), v(n);
  sage2_rms_prop_config_s config;
  randn(engine, &g);
  sage2_rms_prop_config_s_default(&config);
  sage2_rms_prop_config_s_init(&config);

  BeginTimer();
  g_ptr = g.data();
  for (int i = 0; i < m; ++i) {
    func(&config, n, g_ptr, w.data(), v.data());
    g_ptr += n;
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark(int n) {
  int m = 20000000 / n;
  double mops[3] = {0};
  mops[0] = Benchmark(sage2_rms_prop_update_ps_ref, n, m);
  mops[1] = Benchmark(sage2_rms_prop_update_ps, n, m);
  mops[2] = Benchmark(eigen_rms_prop_update_ps, n, m);
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
    PrintHeader1(3, "rms_prop");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
