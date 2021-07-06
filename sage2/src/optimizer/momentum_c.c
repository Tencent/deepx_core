// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>

void sage2_momentum_config_s_default(sage2_momentum_config_s* config) {
  config->rho = 0.5f;
  config->alpha = 0.1f;
}
