// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_adam_update_ss_ref(const sage2_adam_config_s* config,
                                            float g, float* w, float* m,
                                            float* v) {
  float new_m = config->rho1 * *m + config->one_sub_rho1 * g;
  float new_v = config->rho2 * *v + config->one_sub_rho2 * g * g;
  float new_w = *w - config->rho_aux * new_m / (sqrtf(new_v) + config->beta);
  *w = new_w;
  *m = new_m;
  *v = new_v;
}

ATTR_NOINLINE void sage2_adam_update_ps_ref(const sage2_adam_config_s* config,
                                            uint64_t n, const float* g,
                                            float* w, float* m, float* v) {
  for (uint64_t i = 0; i < n; ++i) {
    float new_m = config->rho1 * m[i] + config->one_sub_rho1 * g[i];
    float new_v = config->rho2 * v[i] + config->one_sub_rho2 * g[i] * g[i];
    float new_w =
        w[i] - config->rho_aux * new_m / (sqrtf(new_v) + config->beta);
    w[i] = new_w;
    m[i] = new_m;
    v[i] = new_v;
  }
}

ATTR_NOINLINE void eigen_adam_update_ps(const sage2_adam_config_s* config,
                                        uint64_t n, const float* g, float* w,
                                        float* m, float* v) {
  auto gv = make_eigen_arrayxf_view(g, n);
  auto wv = make_eigen_arrayxf_view(w, n);
  auto mv = make_eigen_arrayxf_view(m, n);
  auto vv = make_eigen_arrayxf_view(v, n);
  mv *= config->rho1;
  mv += config->one_sub_rho1 * gv;
  vv *= config->rho2;
  vv += config->one_sub_rho2 * gv * gv;
  wv -= config->rho_aux * mv / (vv.sqrt() + config->beta);
}

void Test() {
  std::default_random_engine engine;
  float g;
  float w1 = 0, m1 = 0, v1 = 0;
  float w2 = 0, m2 = 0, v2 = 0;
  float w3 = 0, m3 = 0, v3 = 0;
  sage2_adam_config_s config;
  sage2_adam_config_s_default(&config);
  sage2_adam_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    if (i % 1000 == 0) {
      sage2_adam_config_s_prebatch(&config);
    }
    randn(engine, &g);
    sage2_adam_update_ss_ref(&config, g, &w1, &m1, &v1);

    sage2_adam_update_ss(&config, g, &w2, &m2, &v2);
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(m1, m2);
    CHECK_EQUAL(v1, v2);
    w2 = w1;
    m2 = m1;
    v2 = v1;

    eigen_adam_update_ps(&config, 1, &g, &w3, &m3, &v3);
    CHECK_EQUAL(w1, w3);  // NOLINT
    CHECK_EQUAL(m1, m3);  // NOLINT
    CHECK_EQUAL(v1, v3);  // NOLINT
    w3 = w1;
    m3 = m1;
    v3 = v1;
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> g(n);
  std::vector<float> w1(n), m1(n), v1(n);
  std::vector<float> w2(n), m2(n), v2(n);
  std::vector<float> w3(n), m3(n), v3(n);
  sage2_adam_config_s config;
  sage2_adam_config_s_default(&config);
  sage2_adam_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    if (i % 1000 == 0) {
      sage2_adam_config_s_prebatch(&config);
    }
    randn(engine, &g);
    sage2_adam_update_ps_ref(&config, n, g.data(), w1.data(), m1.data(),
                             v1.data());

    sage2_adam_update_ps(&config, n, g.data(), w2.data(), m2.data(), v2.data());
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(m1, m2);
    CHECK_EQUAL(v1, v2);
    w2 = w1;
    m2 = m1;
    v2 = v1;

    eigen_adam_update_ps(&config, n, g.data(), w3.data(), m3.data(), v3.data());
    CHECK_EQUAL(w1, w3);
    CHECK_EQUAL(m1, m3);
    CHECK_EQUAL(v1, v3);
    w3 = w1;
    m3 = m1;
    v3 = v1;
  }
}

template <class Func>
double Benchmark(Func&& func, int n, int _m) {
  std::default_random_engine engine;
  std::vector<float> g(n * _m);
  float* g_ptr;
  std::vector<float> w(n), m(n), v(n);
  sage2_adam_config_s config;
  randn(engine, &g);
  sage2_adam_config_s_default(&config);
  sage2_adam_config_s_init(&config);

  BeginTimer();
  g_ptr = g.data();
  for (int i = 0; i < _m; ++i) {
    if (i % 1000 == 0) {
      sage2_adam_config_s_prebatch(&config);
    }
    func(&config, n, g_ptr, w.data(), m.data(), v.data());
    g_ptr += n;
  }
  EndTimer();
  return 1e-6 * _m / GetSeconds();
}

void Benchmark(int n) {
  int m = 20000000 / n;
  double mops[3] = {0};
  mops[0] = Benchmark(sage2_adam_update_ps_ref, n, m);
  mops[1] = Benchmark(sage2_adam_update_ps, n, m);
  mops[2] = Benchmark(eigen_adam_update_ps, n, m);
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
    PrintHeader1(3, "adam");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
