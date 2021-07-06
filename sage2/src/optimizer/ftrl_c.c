// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>

void sage2_ftrl_config_s_default(sage2_ftrl_config_s* config) {
  config->alpha = 0.01f;
  config->beta = 1;
  config->l1 = 1;
  config->l2 = 0;
  config->inv_alpha = 1 / config->alpha;
}

void sage2_ftrl_config_s_init(sage2_ftrl_config_s* config) {
  config->inv_alpha = 1 / config->alpha;
}
