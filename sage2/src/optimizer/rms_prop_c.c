// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/optimizer.h>

void sage2_rms_prop_config_s_default(sage2_rms_prop_config_s* config) {
  config->rho = 0.5f;
  config->alpha = 0.1f;
  config->beta = 1e-5f;
  config->one_sub_rho = 1 - config->rho;
}

void sage2_rms_prop_config_s_init(sage2_rms_prop_config_s* config) {
  config->one_sub_rho = 1 - config->rho;
}
