// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_OPTIMIZER_H_
#define SAGE2_OPTIMIZER_H_

#include <sage2/macro.h>
#include <stdint.h>  // NOLINT

/************************************************************************/
/* ada delta */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_ada_delta_config_s {
  float rho;          // constant
  float alpha;        // constant
  float beta;         // constant
  float one_sub_rho;  // internal
} sage2_ada_delta_config_s;
// Get a default ada delta config.
SAGE2_C_API void sage2_ada_delta_config_s_default(
    sage2_ada_delta_config_s* config);
// Initialize the ada delta config.
SAGE2_C_API void sage2_ada_delta_config_s_init(
    sage2_ada_delta_config_s* config);
// Perform an ada delta update on scalar.
SAGE2_C_API void sage2_ada_delta_update_ss(
    const sage2_ada_delta_config_s* config, float g, float* w, float* n,
    float* deltaw);
// Perform an ada delta update on vector.
SAGE2_C_API void sage2_ada_delta_update_ps(
    const sage2_ada_delta_config_s* config, uint64_t _n, const float* g,
    float* w, float* n, float* deltaw);

/************************************************************************/
/* ada grad */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_ada_grad_config_s {
  float alpha;  // constant
  float beta;   // constant
} sage2_ada_grad_config_s;
// Get a default ada grad config.
SAGE2_C_API void sage2_ada_grad_config_s_default(
    sage2_ada_grad_config_s* config);
// Perform an ada grad update on scalar.
SAGE2_C_API void sage2_ada_grad_update_ss(const sage2_ada_grad_config_s* config,
                                          float g, float* w, float* n);
// Perform an ada grad update on vector.
SAGE2_C_API void sage2_ada_grad_update_ps(const sage2_ada_grad_config_s* config,
                                          uint64_t _n, const float* g, float* w,
                                          float* n);

/************************************************************************/
/* adam */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_adam_config_s {
  float rho1;          // constant
  float rho2;          // constant
  float alpha;         // constant
  float beta;          // constant
  float rho1t;         // mutable
  float rho2t;         // mutable
  float one_sub_rho1;  // internal
  float one_sub_rho2;  // internal
  float rho_aux;       // internal
} sage2_adam_config_s;
// Get a default adam config.
SAGE2_C_API void sage2_adam_config_s_default(sage2_adam_config_s* config);
// Initialize the adam config.
SAGE2_C_API void sage2_adam_config_s_init(sage2_adam_config_s* config);
// Pre-batch adjust the adam config.
SAGE2_C_API void sage2_adam_config_s_prebatch(sage2_adam_config_s* config);
// Perform an adam update on scalar.
SAGE2_C_API void sage2_adam_update_ss(const sage2_adam_config_s* config,
                                      float g, float* w, float* m, float* v);
// Perform an adam update on vector.
SAGE2_C_API void sage2_adam_update_ps(const sage2_adam_config_s* config,
                                      uint64_t n, const float* g, float* w,
                                      float* m, float* v);

/************************************************************************/
/* ftrl */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_ftrl_config_s {
  float alpha;      // constant
  float beta;       // constant
  float l1;         // constant
  float l2;         // constant
  float inv_alpha;  // internal
} sage2_ftrl_config_s;
// Get a default ftrl config.
SAGE2_C_API void sage2_ftrl_config_s_default(sage2_ftrl_config_s* config);
// Initialize the ftrl config.
SAGE2_C_API void sage2_ftrl_config_s_init(sage2_ftrl_config_s* config);
// Perform a ftrl update on scalar.
SAGE2_C_API void sage2_ftrl_update_ss(const sage2_ftrl_config_s* config,
                                      float g, float* w, float* n, float* z);
// Perform a ftrl update on vector.
SAGE2_C_API void sage2_ftrl_update_ps(const sage2_ftrl_config_s* config,
                                      uint64_t _n, const float* g, float* w,
                                      float* n, float* z);

/************************************************************************/
/* gftrl */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_gftrl_config_s {
  float alpha;      // constant
  float beta;       // constant
  float lambda;     // constant
  float inv_alpha;  // internal
} sage2_gftrl_config_s;
// Get a default gftrl config.
SAGE2_C_API void sage2_gftrl_config_s_default(sage2_gftrl_config_s* config);
// Initialize the gftrl config.
SAGE2_C_API void sage2_gftrl_config_s_init(sage2_gftrl_config_s* config);
// Perform a gftrl update on scalar.
SAGE2_C_API void sage2_gftrl_update_ss(const sage2_gftrl_config_s* config,
                                       float g, float* w, float* n, float* z);
// Perform a gftrl update on vector.
SAGE2_C_API void sage2_gftrl_update_ps(const sage2_gftrl_config_s* config,
                                       uint64_t _n, const float* g, float* w,
                                       float* n, float* z);

/************************************************************************/
/* momentum */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_momentum_config_s {
  float rho;    // constant
  float alpha;  // constant
} sage2_momentum_config_s;
// Get a default momentum config.
SAGE2_C_API void sage2_momentum_config_s_default(
    sage2_momentum_config_s* config);
// Perform a momentum update on scalar.
SAGE2_C_API void sage2_momentum_update_ss(const sage2_momentum_config_s* config,
                                          float g, float* w, float* v);
// Perform a momentum update on vector.
SAGE2_C_API void sage2_momentum_update_ps(const sage2_momentum_config_s* config,
                                          uint64_t _n, const float* g, float* w,
                                          float* v);

/************************************************************************/
/* rms prop */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_rms_prop_config_s {
  float rho;          // constant
  float alpha;        // constant
  float beta;         // constant
  float one_sub_rho;  // internal
} sage2_rms_prop_config_s;
// Get a default rms prop config.
SAGE2_C_API void sage2_rms_prop_config_s_default(
    sage2_rms_prop_config_s* config);
// Initialize the rms prop config.
SAGE2_C_API void sage2_rms_prop_config_s_init(sage2_rms_prop_config_s* config);
// Perform a rms prop update on scalar.
SAGE2_C_API void sage2_rms_prop_update_ss(const sage2_rms_prop_config_s* config,
                                          float g, float* w, float* v);
// Perform a rms prop update on vector.
SAGE2_C_API void sage2_rms_prop_update_ps(const sage2_rms_prop_config_s* config,
                                          uint64_t _n, const float* g, float* w,
                                          float* v);

/************************************************************************/
/* grad_clip */
/************************************************************************/
// Clip the grad into range [-20, 20].
SAGE2_C_API void sage2_grad_clip_20_ss(float* g);
SAGE2_C_API void sage2_grad_clip_20_ps(uint64_t n, float* g);

#endif  // SAGE2_OPTIMIZER_H_
