// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>

void sage2_ada_grad_config_s_default(sage2_ada_grad_config_s* config) {
  config->alpha = 0.01f;
  config->beta = 1e-5f;
}
