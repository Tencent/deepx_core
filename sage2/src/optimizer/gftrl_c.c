// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>

void sage2_gftrl_config_s_default(sage2_gftrl_config_s* config) {
  config->alpha = 0.1f;
  config->beta = 0.01f;
  config->lambda = 1e-4f;
  config->inv_alpha = 1 / config->alpha;
}

void sage2_gftrl_config_s_init(sage2_gftrl_config_s* config) {
  config->inv_alpha = 1 / config->alpha;
}
