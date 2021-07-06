// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_TF_OPTIMIZER_H_
#define SAGE2_TF_OPTIMIZER_H_

#include <sage2/macro.h>
#include <sage2/optimizer.h>
#include <stdint.h>  // NOLINT

/************************************************************************/
/* TensorFlow ada grad v1 */
/* It can accelerate tensorflow::functor::ApplyAdagrad */
/* when update_slots is true. */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_tf_ada_grad_v1_config_s {
  float alpha;  // constant
} sage2_tf_ada_grad_v1_config_s;
SAGE2_C_API void sage2_tf_ada_grad_v1_update_ps(
    const sage2_tf_ada_grad_v1_config_s* config, uint64_t _n, const float* g,
    float* w, float* n);

/************************************************************************/
/* TensorFlow ada grad v2 */
/* It can accelerate tensorflow::functor::ApplyAdagradV2 */
/* when update_slots is true. */
/************************************************************************/
#define sage2_tf_ada_grad_v2_config_s sage2_ada_grad_config_s
#define sage2_tf_ada_grad_v2_update_ps sage2_ada_grad_update_ps

/************************************************************************/
/* TensorFlow adam */
/* It can accelerate tensorflow::functor::ApplyAdam */
/************************************************************************/
#define sage2_tf_adam_config_s sage2_adam_config_s
#define sage2_tf_adam_update_ps sage2_adam_update_ps

/************************************************************************/
/* TensorFlow ftrl v1 */
/* It can accelerate tensorflow::functor::ApplyFtrl */
/* when learning_rate_power is -0.5. */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_tf_ftrl_v1_config_s {
  float alpha;
  float l1;
  float l2;
  float inv_alpha;
} sage2_tf_ftrl_v1_config_s;
SAGE2_C_API void sage2_tf_ftrl_v1_update_ps(
    const sage2_tf_ftrl_v1_config_s* config, uint64_t _n, const float* g,
    float* w, float* n, float* z);

#endif  // SAGE2_TF_OPTIMIZER_H_
