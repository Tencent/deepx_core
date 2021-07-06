// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/cblas_adapter.h>
#include <sage2/sgemm.h>
#include "benchmark.h"
#include "internal_sgemm.h"

const auto* cblas = sage2_cblas_cblas();
const auto* mkl = sage2_cblas_mkl();
const auto* openblas = sage2_cblas_openblas();

void Test(int m, int n, int k) {
  std::default_random_engine engine;
  std::vector<float> X(m * k);
  std::vector<float> Y(k * n);
  std::vector<float> Z1(m * n);
  std::vector<float> Z2;
  std::vector<float> Z3;
  std::vector<float> Z4;
  std::vector<float> Z5;
  std::vector<float> Z6;
  std::vector<float> Z7;
  std::vector<float> Z8;

  auto test = [m, n, k, &engine, &X, &Y, &Z1, &Z2, &Z3, &Z4, &Z5, &Z6, &Z7,
               &Z8](int layout, int transX, int transY, float alpha, float beta,
                    int ldX, int ldY, int ldZ) {
    randn(engine, &X);
    randn(engine, &Y);
    randn(engine, &Z1);
    Z2 = Z1;
    Z3 = Z1;
    Z4 = Z1;
    Z5 = Z1;
    Z6 = Z1;
    Z7 = Z1;
    Z8 = Z1;

    sage2_sgemm_ref(layout, transX, transY, m, n, k, alpha, X.data(), ldX,
                    Y.data(), ldY, beta, Z1.data(), ldZ);

    sage2_sgemm(layout, transX, transY, m, n, k, alpha, X.data(), ldX, Y.data(),
                ldY, beta, Z2.data(), ldZ);
    CHECK_EQUAL(Z1, Z2);

    void* jit = sage2_sgemm_jit_init(layout, transX, transY, m, n, k, alpha,
                                     ldX, ldY, beta, ldZ);
    sage2_sgemm_t jit_func = sage2_sgemm_jit_get(jit);
    jit_func(jit, X.data(), Y.data(), Z3.data());
    sage2_sgemm_jit_uninit(jit);
    CHECK_EQUAL(Z1, Z3);

    sage2_sgemm_eigen(layout, transX, transY, m, n, k, alpha, X.data(), ldX,
                      Y.data(), ldY, beta, Z4.data(), ldZ);
    CHECK_EQUAL(Z1, Z4);

    if (cblas->cblas_sgemm) {
      cblas->cblas_sgemm(layout, transX, transY, m, n, k, alpha, X.data(), ldX,
                         Y.data(), ldY, beta, Z5.data(), ldZ);
      CHECK_EQUAL(Z1, Z5);
    }

    if (mkl->cblas_sgemm) {
      mkl->cblas_sgemm(layout, transX, transY, m, n, k, alpha, X.data(), ldX,
                       Y.data(), ldY, beta, Z6.data(), ldZ);
      CHECK_EQUAL(Z1, Z6);
    }

    if (mkl->mkl_cblas_jit_create_sgemm) {
      void* jit = nullptr;
      (void)mkl->mkl_cblas_jit_create_sgemm(&jit, layout, transX, transY, m, n,
                                            k, alpha, ldX, ldY, beta, ldZ);
      auto* jit_func = mkl->mkl_jit_get_sgemm_ptr(jit);
      jit_func(jit, X.data(), Y.data(), Z7.data());
      (void)mkl->mkl_jit_destroy(jit);
      CHECK_EQUAL(Z1, Z7);
    }

    if (openblas->cblas_sgemm) {
      openblas->cblas_sgemm(layout, transX, transY, m, n, k, alpha, X.data(),
                            ldX, Y.data(), ldY, beta, Z8.data(), ldZ);
      CHECK_EQUAL(Z1, Z8);
    }
  };

  test(101, 111, 111, 0, 0, k, n, n);
  test(101, 111, 111, 0, 1, k, n, n);
  test(101, 111, 111, 0, 0.5f, k, n, n);
  test(101, 111, 111, 1, 0, k, n, n);
  test(101, 112, 111, 1, 1, m, n, n);
  test(101, 111, 112, 1, 0.5f, k, k, n);
  test(101, 112, 112, 1, 0.3f, m, k, n);
  test(101, 111, 111, 0.5f, 0, k, n, n);
  test(101, 112, 111, 0.5f, 1, m, n, n);
  test(101, 111, 112, 0.5f, 0.5f, k, k, n);
  test(101, 112, 112, 0.5f, 0.3f, m, k, n);

  test(102, 111, 111, 0, 0, m, k, m);
  test(102, 111, 111, 0, 1, m, k, m);
  test(102, 111, 111, 0, 0.5f, m, k, m);
  test(102, 111, 111, 1, 0, m, k, m);
  test(102, 112, 111, 1, 1, k, k, m);
  test(102, 111, 112, 1, 0.5f, m, n, m);
  test(102, 112, 112, 1, 0.3f, k, n, m);
  test(102, 111, 111, 0.5f, 0, m, k, m);
  test(102, 112, 111, 0.5f, 1, k, k, m);
  test(102, 111, 112, 0.5f, 0.5f, m, n, m);
  test(102, 112, 112, 0.5f, 0.3f, k, n, m);

  X.resize(m * k * 2);
  Y.resize(k * n * 2);
  Z1.resize(m * n * 2);

  test(101, 111, 111, 0, 0, k * 2, n * 2, n * 2);
  test(101, 111, 111, 0, 1, k * 2, n * 2, n * 2);
  test(101, 111, 111, 0, 0.5f, k * 2, n * 2, n * 2);
  test(101, 111, 111, 1, 0, k * 2, n * 2, n * 2);
  test(101, 112, 111, 1, 1, m * 2, n * 2, n * 2);
  test(101, 111, 112, 1, 0.5f, k * 2, k * 2, n * 2);
  test(101, 112, 112, 1, 0.3f, m * 2, k * 2, n * 2);
  test(101, 111, 111, 0.5f, 0, k * 2, n * 2, n * 2);
  test(101, 112, 111, 0.5f, 1, m * 2, n * 2, n * 2);
  test(101, 111, 112, 0.5f, 0.5f, k * 2, k * 2, n * 2);
  test(101, 112, 112, 0.5f, 0.3f, m * 2, k * 2, n * 2);

  test(102, 111, 111, 0, 0, m * 2, k * 2, m * 2);
  test(102, 111, 111, 0, 1, m * 2, k * 2, m * 2);
  test(102, 111, 111, 0, 0.5f, m * 2, k * 2, m * 2);
  test(102, 111, 111, 1, 0, m * 2, k * 2, m * 2);
  test(102, 112, 111, 1, 1, k * 2, k * 2, m * 2);
  test(102, 111, 112, 1, 0.5f, m * 2, n * 2, m * 2);
  test(102, 112, 112, 1, 0.3f, k * 2, n * 2, m * 2);
  test(102, 111, 111, 0.5f, 0, m * 2, k * 2, m * 2);
  test(102, 112, 111, 0.5f, 1, k * 2, k * 2, m * 2);
  test(102, 111, 112, 0.5f, 0.5f, m * 2, n * 2, m * 2);
  test(102, 112, 112, 0.5f, 0.3f, k * 2, n * 2, m * 2);
}

template <class Func>
double Benchmark(Func&& func, int _m, int m, int n, int k, int transX,
                 int transY) {
  std::default_random_engine engine;
  float alpha = 1, beta = 0;
  std::vector<float> X(m * k);
  std::vector<float> Y(k * n);
  std::vector<float> Z(m * n);
  int ldX = transX ? m : k;
  int ldY = transY ? k : n;
  int ldZ = n;
  transX = transX ? 112 : 111;
  transY = transY ? 112 : 111;
  randn(engine, &X);
  randn(engine, &Y);

  BeginTimer();
  for (int i = 0; i < _m; ++i) {
    func(101, transX, transY, m, n, k, alpha, X.data(), ldX, Y.data(), ldY,
         beta, Z.data(), ldZ);
  }
  EndTimer();
  return GetGFLOPS(2.0 * _m * m * n * k);
}

double BenchmarkJit(int _m, int m, int n, int k, int transX, int transY) {
  std::default_random_engine engine;
  float alpha = 1, beta = 0;
  std::vector<float> X(m * k);
  std::vector<float> Y(k * n);
  std::vector<float> Z(m * n);
  int ldX = transX ? m : k;
  int ldY = transY ? k : n;
  int ldZ = n;
  transX = transX ? 112 : 111;
  transY = transY ? 112 : 111;
  randn(engine, &X);
  randn(engine, &Y);

  void* jit = sage2_sgemm_jit_init(101, transX, transY, m, n, k, alpha, ldX,
                                   ldY, beta, ldZ);
  sage2_sgemm_t jit_func = sage2_sgemm_jit_get(jit);
  BeginTimer();
  for (int i = 0; i < _m; ++i) {
    jit_func(jit, X.data(), Y.data(), Z.data());
  }
  EndTimer();
  sage2_sgemm_jit_uninit(jit);
  return GetGFLOPS(2.0 * _m * m * n * k);
}

double BenchmarkMKLJit(int _m, int m, int n, int k, int transX, int transY) {
  std::default_random_engine engine;
  float alpha = 1, beta = 0;
  std::vector<float> X(m * k);
  std::vector<float> Y(k * n);
  std::vector<float> Z(m * n);
  int ldX = transX ? m : k;
  int ldY = transY ? k : n;
  int ldZ = n;
  transX = transX ? 112 : 111;
  transY = transY ? 112 : 111;
  randn(engine, &X);
  randn(engine, &Y);

  void* jit = nullptr;
  (void)mkl->mkl_cblas_jit_create_sgemm(&jit, 101, transX, transY, m, n, k,
                                        alpha, ldX, ldY, beta, ldZ);
  auto* jit_func = mkl->mkl_jit_get_sgemm_ptr(jit);
  BeginTimer();
  for (int i = 0; i < _m; ++i) {
    jit_func(jit, X.data(), Y.data(), Z.data());
  }
  EndTimer();
  (void)mkl->mkl_jit_destroy(jit);
  return GetGFLOPS(2.0 * _m * m * n * k);
}

void Benchmark(int m, int n, int k, int transX, int transY) {
  int _m = (int)(10000000000.0 / m / n / k);
  if (_m > 1000) {
    _m = 1000;
  }
  if (_m < 10) {
    _m = 10;
  }

  double gflops[7] = {0};
  gflops[0] = Benchmark(sage2_sgemm, _m, m, n, k, transX, transY);
  gflops[1] = BenchmarkJit(_m, m, n, k, transX, transY);
  gflops[2] = Benchmark(sage2_sgemm_eigen, _m, m, n, k, transX, transY);
  if (cblas->cblas_sgemm) {
    gflops[3] = Benchmark(cblas->cblas_sgemm, _m, m, n, k, transX, transY);
  }
  if (mkl->cblas_sgemm) {
    gflops[4] = Benchmark(mkl->cblas_sgemm, _m, m, n, k, transX, transY);
  }
  if (mkl->mkl_cblas_jit_create_sgemm) {
    gflops[5] = BenchmarkMKLJit(_m, m, n, k, transX, transY);
  }
  if (openblas->cblas_sgemm) {
    gflops[6] = Benchmark(openblas->cblas_sgemm, _m, m, n, k, transX, transY);
  }

  char buf[64];
  snprintf(buf, sizeof(buf), "%8d %8d %8d %8d %8d", m, n, k, transX, transY);
  PrintContent(buf, gflops);
}

int main(int argc, char** argv) {
  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    Test(1, 1, 1);
    auto i_list = {32, 64, 128, 10, 50, 100};
    for (int i : i_list) {
      Test(1, 1, i);
    }
    for (int i : i_list) {
      Test(i, i + 1, 1);
    }
    for (int i : i_list) {
      Test(1, i, i + 1);
    }
    for (int i : i_list) {
      Test(i, 1, i + 1);
    }
    for (int i : i_list) {
      Test(i, i + 1, i + 2);
    }
    // matrices of random shape
    std::default_random_engine engine;
    std::uniform_int_distribution<int> dist(1, 128);
    for (int i = 0; i < 10; ++i) {
      int m = dist(engine);
      int n = dist(engine);
      int k = dist(engine);
      Test(1, 1, k);
      Test(m, n, 1);
      Test(1, n, k);
      Test(m, 1, k);
      Test(m, n, k);
    }
  }

  if (action & 2) {
    char buf[64];
    size_t prefix_size =  // NOLINT
        (size_t)snprintf(buf, sizeof(buf), "%8s %8s %8s %8s %8s", "m", "n", "k",
                         "transX", "transY");
    PrintHeader1(prefix_size, 7, "sgemm");
    PrintHeader2(buf, 7, "GFLOPS");
    PrintHeader3(prefix_size, "opt", "jit", "eigen", cblas->vendor, mkl->vendor,
                 "mkl_jit", openblas->vendor);
    if (argc >= 5) {
      int m = std::stoi(argv[2]);
      int n = std::stoi(argv[3]);
      int k = std::stoi(argv[4]);
      for (int transX = 0; transX <= 1; ++transX) {
        for (int transY = 0; transY <= 1; ++transY) {
          Benchmark(m, n, k, transX, transY);
        }
      }
    } else {
      Benchmark(1, 1, 1, 0, 0);
      auto i_list = {32, 64, 128, 256, 512};
      auto j_list = {4, 8, 16, 32, 64, 128, 256, 512};
      for (int i : i_list) {
        Benchmark(1, 1, i, 0, 0);
      }
      for (int i : i_list) {
        Benchmark(i, i, 1, 0, 0);
      }
      for (int i : i_list) {
        Benchmark(1, i, i, 0, 0);
      }
      for (int i : i_list) {
        Benchmark(i, 1, i, 0, 0);
      }
      for (int i : i_list) {
        for (int j : j_list) {
          Benchmark(i, j, j * 2, 0, 0);
          Benchmark(i, j, j * 2, 0, 1);
          Benchmark(i, j, j * 2, 1, 0);
          Benchmark(i, j, j * 2, 1, 1);
        }
      }
    }
  }
  return 0;
}
