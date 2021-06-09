// Copyright 2021 the deepx authors.
// Author: Xingfei Li (xingfeili@tencent.com)
//

#pragma once
#include <deepx_core/tensor/ll_math.h>
#include <deepx_core/tensor/ll_tensor.h>
#include <deepx_core/tensor/sparse_row_matrix.h>
#include <deepx_core/tensor/tensor.h>

namespace deepx_core {

/************************************************************************/
/* LLWePSOptimizer */
/************************************************************************/
template <typename T, typename I>
class LLWePSOptimizer : public LLOptimizer<T, I> {
 private:
  using base_t = LLOptimizer<T, I>;

 public:
  using float_t = T;
  using int_t = I;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using tsr_t = Tensor<float_t>;
  using srm_t = SparseRowMatrix<float_t, int_t>;
  using ll_math_t = LLMath<float_t>;

  using base_t::Clip;
  using base_t::ClipArray;
  using base_t::get_data;
  using base_t::get_total_dim;
  using base_t::Init;
  using base_t::PostBatch;
  using base_t::PreBatch;

  using base_t::GRAD_CLIP_THRESHOLD;
  using base_t::SMOOTH;

 public:
  /************************************************************************/
  /* scalar */
  /************************************************************************/
  template <class Config>
  static void UpdateScalar(const Config& config, float_t g, float_t w,
                           ptr_t d) noexcept;

  template <class Config>
  static void UpdateScalar(const Config& config, float_t g, float_t w, ptr_t d,
                           ptr_t a) noexcept;

  template <class Config>
  static void UpdateScalar(const Config& config, float_t g, float_t w, ptr_t d,
                           ptr_t a, ptr_t b) noexcept;

 public:
  /************************************************************************/
  /* array */
  /************************************************************************/
  template <class Config>
  static void UpdateArray(const Config& config, int n, cptr_t g, cptr_t w,
                          ptr_t d) noexcept {
    for (int i = 0; i < n; ++i) {
      UpdateScalar(config, g[i], w[i], &d[i]);
    }
  }

  template <class Config>
  static void UpdateArray(const Config& config, int n, cptr_t g, cptr_t w,
                          ptr_t d, ptr_t a) noexcept {
    for (int i = 0; i < n; ++i) {
      UpdateScalar(config, g[i], w[i], &d[i], &a[i]);
    }
  }

  template <class Config>
  static void UpdateArray(const Config& config, int n, cptr_t g, cptr_t w,
                          ptr_t d, ptr_t a, ptr_t b) noexcept {
    for (int i = 0; i < n; ++i) {
      UpdateScalar(config, g[i], w[i], &d[i], &a[i], &b[i]);
    }
  }

 public:
  /************************************************************************/
  /* grad tsr, param tsr */
  /************************************************************************/
  template <class Config>
  static void UpdateTSR2TSR(const Config& config, const tsr_t& G,
                            const tsr_t& W, tsr_t* D) noexcept {
    UpdateArray(config, get_total_dim(G), get_data(G), get_data(W),
                get_data(D));
  }

  template <class Config>
  static void UpdateTSR2TSR(const Config& config, const tsr_t& G,
                            const tsr_t& W, tsr_t* D, tsr_t* A) noexcept {
    UpdateArray(config, get_total_dim(G), get_data(G), get_data(W), get_data(D),
                get_data(A));
  }

  template <class Config>
  static void UpdateTSR2TSR(const Config& config, const tsr_t& G,
                            const tsr_t& W, tsr_t* D, tsr_t* A,
                            tsr_t* B) noexcept {
    UpdateArray(config, get_total_dim(G), get_data(G), get_data(W), get_data(D),
                get_data(A), get_data(B));
  }

 public:
  /************************************************************************/
  /* grad srm, param tsr */
  /************************************************************************/
  template <class Config>
  static void UpdateSRM2TSR(const Config& config, const srm_t& G,
                            const tsr_t& W, tsr_t* D) noexcept {
    DXASSERT_RANK2(*D);
    int n = D->dim(1);
    DXASSERT(G.col() == n);
    DXASSERT_SAME_SHAPE(W, *D);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, *(get_data(W) + i), get_data(D) + i);
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, get_data(W) + i * n, get_data(D) + i * n);
      }
    }
  }

  template <class Config>
  static void UpdateSRM2TSR(const Config& config, const srm_t& G,
                            const tsr_t& W, tsr_t* D, tsr_t* A) noexcept {
    DXASSERT_RANK2(*D);
    int n = D->dim(1);
    DXASSERT(G.col() == n);
    DXASSERT_SAME_SHAPE(W, *D, *A);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, *(get_data(W) + i), get_data(D) + i,
                     get_data(A) + i);
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, get_data(W) + i * n, get_data(D) + i * n,
                    get_data(A) + i * n);
      }
    }
  }

  template <class Config>
  static void UpdateSRM2TSR(const Config& config, const srm_t& G,
                            const tsr_t& W, tsr_t* D, tsr_t* A,
                            tsr_t* B) noexcept {
    DXASSERT_RANK2(*D);
    int n = D->dim(1);
    DXASSERT(G.col() == n);
    DXASSERT_SAME_SHAPE(W, *D, *A, *B);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, *(get_data(W) + i), get_data(D) + i,
                     get_data(A) + i, get_data(B) + i);
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, get_data(W) + i * n, get_data(D) + i * n,
                    get_data(A) + i * n, get_data(B) + i * n);
      }
    }
  }

 public:
  /************************************************************************/
  /* grad srm, param srm */
  /************************************************************************/
  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G,
                            const srm_t& W, srm_t* D) {
    int n = G.col();
    DXASSERT(W.col() == n);
    DXASSERT(D->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, *W.get_row_no_init(i), D->get_row_no_init(i));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W.get_row_no_init(i), D->get_row_no_init(i));
      }
    }
  }

  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G,
                            const srm_t& W, srm_t* D, srm_t* A) {
    int n = G.col();
    DXASSERT(W.col() == n);
    DXASSERT(D->col() == n);
    DXASSERT(A->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, *W.get_row_no_init(i), D->get_row_no_init(i),
                     A->get_row_no_init(i));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W.get_row_no_init(i), D->get_row_no_init(i),
                    A->get_row_no_init(i));
      }
    }
  }

  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G,
                            const srm_t& W, srm_t* D, srm_t* A, srm_t* B) {
    int n = G.col();
    DXASSERT(W.col() == n);
    DXASSERT(D->col() == n);
    DXASSERT(A->col() == n);
    DXASSERT(B->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, *W.get_row_no_init(i), D->get_row_no_init(i),
                     A->get_row_no_init(i), B->get_row_no_init(i));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W.get_row_no_init(i), D->get_row_no_init(i),
                    A->get_row_no_init(i), B->get_row_no_init(i));
      }
    }
  }

 public:
  /************************************************************************/
  /* sgd */
  /************************************************************************/
  using typename base_t::SGDConfig;

  static void UpdateScalar(const SGDConfig& config, float_t g, float_t /*w*/,
                           ptr_t d) noexcept {
    *d = -config.real_alpha * g;
  }

  static void UpdateArray(const SGDConfig& config, int n, cptr_t G,
                          cptr_t /*W*/, ptr_t D) noexcept {
    ll_math_t::mul_scalar(n, G, -config.real_alpha, D);
  }

 public:
  /************************************************************************/
  /* ada delta */
  /************************************************************************/
  using typename base_t::AdaDeltaConfig;

  static void UpdateScalar(const AdaDeltaConfig& config, float_t g,
                           float_t /*w*/, ptr_t d, ptr_t n,
                           ptr_t deltaw) noexcept {
    float_t new_n = config.rho * *n + config.one_sub_rho * g * g;
    float_t a =
        std::sqrt(*deltaw + config.beta) / std::sqrt(new_n + config.beta) * g;
    float_t new_deltaw = config.rho * *deltaw + config.one_sub_rho * a * a;
    *d = -config.alpha * a;
    *n = new_n;
    *deltaw = new_deltaw;
  }

 public:
  /************************************************************************/
  /* ada grad */
  /************************************************************************/
  using typename base_t::AdaGradConfig;

  static void UpdateScalar(const AdaGradConfig& config, float_t g,
                           float_t /*w*/, ptr_t d, ptr_t n) noexcept {
    float_t new_n = *n + g * g;
    *d = -g / std::sqrt(new_n + config.beta) * config.alpha;
    *n = new_n;
  }

 public:
  /************************************************************************/
  /* adam */
  /************************************************************************/
  using typename base_t::AdamConfig;

  static void UpdateScalar(const AdamConfig& config, float_t g, float_t /*w*/,
                           ptr_t d, ptr_t m, ptr_t v) noexcept {
    float_t new_m = config.rho1 * *m + config.one_sub_rho1 * g;
    float_t new_v = config.rho2 * *v + config.one_sub_rho2 * g * g;
    *d = -config.rho_aux * new_m / (std::sqrt(new_v) + config.beta);
    *m = new_m;
    *v = new_v;
  }

 public:
  /************************************************************************/
  /* momentum */
  /************************************************************************/
  using typename base_t::MomentumConfig;

  static void UpdateScalar(const MomentumConfig& config, float_t g,
                           float_t /*w*/, ptr_t d, ptr_t v) noexcept {
    float_t new_v = config.rho * *v + g;
    *d = -config.alpha * new_v;
    *v = new_v;
  }

 public:
  /************************************************************************/
  /* rmsprop */
  /************************************************************************/
  using typename base_t::RMSPropConfig;

  static void UpdateScalar(const RMSPropConfig& config, float_t g,
                           float_t /*w*/, ptr_t d, ptr_t v) noexcept {
    float_t new_v = config.rho * *v + config.one_sub_rho * g * g;
    *d = -g / std::sqrt(new_v + config.beta) * config.alpha;
    *v = new_v;
  }
};

}  // namespace deepx_core
