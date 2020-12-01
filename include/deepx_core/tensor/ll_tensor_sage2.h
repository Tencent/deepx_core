// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>

namespace deepx_core {

/************************************************************************/
/* sage2 implementations */
/************************************************************************/
template <>
inline void LLOptimizer<float, uint64_t>::ClipArray(int n, ptr_t g) noexcept {
  sage2_grad_clip_20_ps(n, g);
}

template <>
inline void LLOptimizer<float, uint64_t>::Init(
    AdaDeltaConfig* config) noexcept {
  sage2_ada_delta_config_s_init((sage2_ada_delta_config_s*)config);
}

template <>
template <>
inline void LLOptimizer<float, uint64_t>::UpdateArray(
    const AdaDeltaConfig& config, int _n, cptr_t g, ptr_t w, ptr_t n,
    ptr_t deltaw) noexcept {
  static_assert(sizeof(sage2_ada_delta_config_s) == sizeof(AdaDeltaConfig), "");
  sage2_ada_delta_update_ps((const sage2_ada_delta_config_s*)&config, _n, g, w,
                            n, deltaw);
}

template <>
template <>
inline void LLOptimizer<float, uint64_t>::UpdateArray(
    const AdaGradConfig& config, int _n, cptr_t g, ptr_t w, ptr_t n) noexcept {
  static_assert(sizeof(sage2_ada_grad_config_s) == sizeof(AdaGradConfig), "");
  sage2_ada_grad_update_ps((const sage2_ada_grad_config_s*)&config, _n, g, w,
                           n);
}

template <>
inline void LLOptimizer<float, uint64_t>::Init(AdamConfig* config) noexcept {
  sage2_adam_config_s_init((sage2_adam_config_s*)config);
}

template <>
inline void LLOptimizer<float, uint64_t>::PreBatch(
    AdamConfig* config) noexcept {
  sage2_adam_config_s_prebatch((sage2_adam_config_s*)config);
}

template <>
template <>
inline void LLOptimizer<float, uint64_t>::UpdateArray(const AdamConfig& config,
                                                      int n, cptr_t g, ptr_t w,
                                                      ptr_t m,
                                                      ptr_t v) noexcept {
  static_assert(sizeof(sage2_adam_config_s) == sizeof(AdamConfig), "");
  sage2_adam_update_ps((const sage2_adam_config_s*)&config, n, g, w, m, v);
}

template <>
inline void LLOptimizer<float, uint64_t>::Init(FTRLConfig* config) noexcept {
  sage2_ftrl_config_s_init((sage2_ftrl_config_s*)config);
}

template <>
template <>
inline void LLOptimizer<float, uint64_t>::UpdateArray(const FTRLConfig& config,
                                                      int _n, cptr_t g, ptr_t w,
                                                      ptr_t n,
                                                      ptr_t z) noexcept {
  static_assert(sizeof(sage2_ftrl_config_s) == sizeof(FTRLConfig), "");
  sage2_ftrl_update_ps((const sage2_ftrl_config_s*)&config, _n, g, w, n, z);
}

template <>
inline void LLOptimizer<float, uint64_t>::Init(GFTRLConfig* config) noexcept {
  sage2_gftrl_config_s_init((sage2_gftrl_config_s*)config);
}

template <>
inline void LLOptimizer<float, uint64_t>::UpdateArray(const GFTRLConfig& config,
                                                      int _n, cptr_t g, ptr_t w,
                                                      ptr_t n,
                                                      ptr_t z) noexcept {
  static_assert(sizeof(sage2_gftrl_config_s) == sizeof(GFTRLConfig), "");
  sage2_gftrl_update_ps((const sage2_gftrl_config_s*)&config, _n, g, w, n, z);
}

template <>
template <>
inline void LLOptimizer<float, uint64_t>::UpdateArray(
    const MomentumConfig& config, int n, cptr_t g, ptr_t w, ptr_t v) noexcept {
  static_assert(sizeof(sage2_momentum_config_s) == sizeof(MomentumConfig), "");
  sage2_momentum_update_ps((const sage2_momentum_config_s*)&config, n, g, w, v);
}

template <>
inline void LLOptimizer<float, uint64_t>::Init(RMSPropConfig* config) noexcept {
  sage2_rms_prop_config_s_init((sage2_rms_prop_config_s*)config);
}

template <>
template <>
inline void LLOptimizer<float, uint64_t>::UpdateArray(
    const RMSPropConfig& config, int n, cptr_t g, ptr_t w, ptr_t v) noexcept {
  static_assert(sizeof(sage2_rms_prop_config_s) == sizeof(RMSPropConfig), "");
  sage2_rms_prop_update_ps((const sage2_rms_prop_config_s*)&config, n, g, w, v);
}

}  // namespace deepx_core
