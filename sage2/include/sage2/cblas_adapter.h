// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_CBLAS_ADAPTER_H_
#define SAGE2_CBLAS_ADAPTER_H_

#include <sage2/macro.h>

// NOLINTNEXTLINE
typedef void (*sage2_mkl_cblas_jit_sgemm_t)(void* jit, float* A, float* B,
                                            float* C);
// NOLINTNEXTLINE
typedef struct _sage2_cblas {
  void* handle;
  const char* so;
  const char* vendor;

  // cblas
  void (*cblas_saxpy)(int N, float alpha, const float* X, int incX, float* Y,
                      int incY);
  void (*cblas_saxpby)(int N, float alpha, const float* X, int incX, float beta,
                       float* Y, int incY);
  float (*cblas_sdot)(int N, const float* X, int incX, const float* Y,
                      int incY);
  float (*cblas_snrm2)(int N, const float* X, int incX);
  float (*cblas_sasum)(int N, const float* X, int incX);
  void (*cblas_sscal)(int N, float alpha, float* X, int incX);
  void (*cblas_sgemm)(int Layout, int TransA, int TransB, int M, int N, int K,
                      float alpha, const float* A, int ldA, const float* B,
                      int ldB, float beta, float* C, int ldC);

  // MKL only sgemm batch
  void (*cblas_sgemm_batch)(int Layout, const int* TransA_Array,
                            const int* TransB_Array, const int* M_Array,
                            const int* N_Array, const int* K_Array,
                            const float* alpha_Array, const float** A_Array,
                            const int* ldA_Array, const float** B_Array,
                            const int* ldB_Array, const float* beta_Array,
                            float** C_Array, const int* ldC_Array,
                            int group_count, const int* group_size);

  // MKL only sgemm jit
  int (*mkl_cblas_jit_create_sgemm)(void** jit, int Layout, int TransA,
                                    int TransB, int M, int N, int K,
                                    const float alpha, int ldA, int ldB,
                                    const float beta, int ldC);
  sage2_mkl_cblas_jit_sgemm_t (*mkl_jit_get_sgemm_ptr)(const void* jit);
  int (*mkl_jit_destroy)(void* jit);

  // MKL only vml
  void (*vsAdd)(int n, const float* a, const float* b, float* r);
  void (*vsSub)(int n, const float* a, const float* b, float* r);
  void (*vsMul)(int n, const float* a, const float* b, float* r);
  void (*vsDiv)(int n, const float* a, const float* b, float* r);
  void (*vsExp)(int n, const float* a, float* r);
  void (*vsLn)(int n, const float* a, float* r);
  void (*vsSqrt)(int n, const float* a, float* r);
  void (*vsTanh)(int n, const float* a, float* r);
} sage2_cblas;

SAGE2_C_API const sage2_cblas* sage2_cblas_cblas();
SAGE2_C_API const sage2_cblas* sage2_cblas_mkl();
SAGE2_C_API const sage2_cblas* sage2_cblas_openblas();

#endif  // SAGE2_CBLAS_ADAPTER_H_
