// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>
#include "benchmark.h"
#include "eigen_wrapper.h"
#include "internal_macro.h"

using namespace sage2;  // NOLINT

ATTR_NOINLINE void sage2_gftrl_update_ss_ref(const sage2_gftrl_config_s* config,
                                             float g, float* w, float* n,
                                             float* z) {
  float new_n = *n + g * g;
  float sqrt_n = sqrtf(*n);
  float sqrt_new_n = sqrtf(new_n);
  float sigma = (sqrt_n - sqrt_new_n) * config->inv_alpha;
  float new_z = *z + g + sigma * *w;
  *z = new_z;
  *n = new_n;

  float norm2_z = fabsf(new_z);
  float threshold = config->lambda;
  if (norm2_z < threshold) {
    *w = 0;
  } else {
    float tmp = config->alpha * (threshold / norm2_z - 1);
    *w = tmp * new_z / (config->beta + sqrt_new_n);
  }
}

ATTR_NOINLINE void sage2_gftrl_update_ps_ref(const sage2_gftrl_config_s* config,
                                             uint64_t _n, const float* g,
                                             float* w, float* n, float* z) {
  for (uint64_t i = 0; i < _n; ++i) {
    float new_n = n[i] + g[i] * g[i];
    float sqrt_n = sqrtf(n[i]);
    float sqrt_new_n = sqrtf(new_n);
    float sigma = (sqrt_n - sqrt_new_n) * config->inv_alpha;
    float new_z = z[i] + g[i] + sigma * w[i];
    z[i] = new_z;
    n[i] = new_n;
  }

  float norm2_z = 0;
  for (uint64_t i = 0; i < _n; ++i) {
    norm2_z += z[i] * z[i];
  }
  norm2_z = sqrtf(norm2_z);
  float threshold = config->lambda * sqrtf((float)_n);
  if (norm2_z < threshold) {
    for (uint64_t i = 0; i < _n; ++i) {
      w[i] = 0;
    }
  } else {
    float tmp = config->alpha * (threshold / norm2_z - 1);
    for (uint64_t i = 0; i < _n; ++i) {
      w[i] = tmp * z[i] / (config->beta + sqrtf(n[i]));
    }
  }
}

ATTR_NOINLINE void eigen_gftrl_update_ps(const sage2_gftrl_config_s* config,
                                         uint64_t _n, const float* g, float* w,
                                         float* n, float* z) {
  auto gv = make_eigen_arrayxf_view(g, _n);
  auto wv = make_eigen_arrayxf_view(w, _n);
  auto nv = make_eigen_arrayxf_view(n, _n);
  auto zv = make_eigen_arrayxf_view(z, _n);
  auto sqrt_n = nv.sqrt().eval();
  nv += gv * gv;
  auto sqrt_new_n = nv.sqrt();
  zv += gv + (sqrt_n - sqrt_new_n) * config->inv_alpha * wv;

  float norm2_z = sqrtf((zv * zv).sum());
  float threshold = config->lambda * sqrtf((float)_n);
  if (norm2_z < threshold) {
    wv = eigen_arrayxf_view_t::Zero(_n);
  } else {
    float tmp = config->alpha * (threshold / norm2_z - 1);
    wv = tmp * zv / (config->beta + nv.sqrt());
  }
}

void Test() {
  std::default_random_engine engine;
  float g;
  float w1 = 0, n1 = 0, z1 = 0;
  float w2 = 0, n2 = 0, z2 = 0;
  float w3 = 0, n3 = 0, z3 = 0;
  sage2_gftrl_config_s config;
  sage2_gftrl_config_s_default(&config);
  sage2_gftrl_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_gftrl_update_ss_ref(&config, g, &w1, &n1, &z1);

    sage2_gftrl_update_ss(&config, g, &w2, &n2, &z2);
    CHECK_EQUAL(w1, w2);  // NOLINT
    CHECK_EQUAL(n1, n2);  // NOLINT
    CHECK_EQUAL(z1, z2);  // NOLINT
    w2 = w1;
    n2 = n1;
    z2 = z1;

    eigen_gftrl_update_ps(&config, 1, &g, &w3, &n3, &z3);
    CHECK_EQUAL(w1, w3);  // NOLINT
    CHECK_EQUAL(n1, n3);
    CHECK_EQUAL(z1, z3);
    w3 = w1;
    n3 = n1;
    z3 = z1;
  }
}

void Test(int n) {
  std::default_random_engine engine;
  std::vector<float> g(n);
  std::vector<float> w1(n), n1(n), z1(n);
  std::vector<float> w2(n), n2(n), z2(n);
  std::vector<float> w3(n), n3(n), z3(n);
  sage2_gftrl_config_s config;
  sage2_gftrl_config_s_default(&config);
  sage2_gftrl_config_s_init(&config);

  for (int i = 0; i < 10000; ++i) {
    randn(engine, &g);
    sage2_gftrl_update_ps_ref(&config, n, g.data(), w1.data(), n1.data(),
                              z1.data());

    sage2_gftrl_update_ps(&config, n, g.data(), w2.data(), n2.data(),
                          z2.data());
    CHECK_EQUAL(w1, w2);
    CHECK_EQUAL(n1, n2);
    CHECK_EQUAL(z1, z2);
    w2 = w1;
    n2 = n1;
    z2 = z1;

    eigen_gftrl_update_ps(&config, n, g.data(), w3.data(), n3.data(),
                          z3.data());
    CHECK_EQUAL(w1, w3);
    CHECK_EQUAL(n1, n3);
    CHECK_EQUAL(z1, z3);
    w3 = w1;
    n3 = n1;
    z3 = z1;
  }
}

template <class Func>
double Benchmark(Func&& func, int _n, int m) {
  std::default_random_engine engine;
  std::vector<float> g(_n * m);
  float* g_ptr;
  std::vector<float> w(_n), n(_n), z(_n);
  sage2_gftrl_config_s config;
  randn(engine, &g);
  sage2_gftrl_config_s_default(&config);
  sage2_gftrl_config_s_init(&config);

  BeginTimer();
  g_ptr = g.data();
  for (int i = 0; i < m; ++i) {
    func(&config, _n, g_ptr, w.data(), n.data(), z.data());
    g_ptr += _n;
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark(int n) {
  int m = 20000000 / n;
  double mops[3] = {0};
  mops[0] = Benchmark(sage2_gftrl_update_ps_ref, n, m);
  mops[1] = Benchmark(sage2_gftrl_update_ps, n, m);
  mops[2] = Benchmark(eigen_gftrl_update_ps, n, m);
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
    PrintHeader1(3, "gftrl");
    PrintHeader2(3, "Mop/s");
    PrintHeader3("ref", "opt", "eigen");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
