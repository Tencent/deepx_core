// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <math.h>
#include <sage2/optimizer.h>

void sage2_adam_config_s_default(sage2_adam_config_s* config) {
  config->rho1 = 0.9f;
  config->rho2 = 0.999f;
  config->alpha = 0.001f;
  config->beta = 1e-5f;
  config->rho1t = 1;
  config->rho2t = 1;
  config->one_sub_rho1 = 1 - config->rho1;
  config->one_sub_rho2 = 1 - config->rho2;
  config->rho_aux = 0;
}

void sage2_adam_config_s_init(sage2_adam_config_s* config) {
  config->rho1t = 1;
  config->rho2t = 1;
  config->one_sub_rho1 = 1 - config->rho1;
  config->one_sub_rho2 = 1 - config->rho2;
  config->rho_aux = 0;
}

void sage2_adam_config_s_prebatch(sage2_adam_config_s* config) {
  config->rho1t *= config->rho1;
  config->rho2t *= config->rho2;
  config->rho_aux =
      sqrtf(1 - config->rho2t) / (1 - config->rho1t) * config->alpha;
}
