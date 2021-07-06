// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>

void sage2_ada_delta_config_s_default(sage2_ada_delta_config_s* config) {
  config->rho = 0.95f;
  config->alpha = 1;
  config->beta = 1e-5f;
  config->one_sub_rho = 1 - config->rho;
}

void sage2_ada_delta_config_s_init(sage2_ada_delta_config_s* config) {
  config->one_sub_rho = 1 - config->rho;
}
