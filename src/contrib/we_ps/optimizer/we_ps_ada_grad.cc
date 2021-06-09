// Copyright 2021 the deepx authors.
// Author: Xingfei  Li (xingfeili@tencent.com)
//

#include <deepx_core/contrib/we_ps/optimizer/we_ps_optimizer_impl.h>

namespace deepx_core {

class WePSAdaGradOptimizer : public WePSOptimizerBase1 {
 private:
  ll_we_ps_optimizer_t::AdaGradConfig config_;

 public:
  DEFINE_WE_PS_OPTIMIZER_LIKE(WePSAdaGradOptimizer);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (k == "alpha") {
      config_.alpha = (float_t)std::stod(v);
      if (config_.alpha <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "beta") {
      config_.beta = (float_t)std::stod(v);
      if (config_.beta <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else {
      DXERROR("Unexpected config: %s=%s.", k.c_str(), v.c_str());
      return false;
    }
    return true;
  }

  bool PostInitConfig() override {
    ll_we_ps_optimizer_t::Init(&config_);
    return true;
  }

  void PreUpdate() override { ll_we_ps_optimizer_t::PreBatch(&config_); }

  void PostUpdate() override { ll_we_ps_optimizer_t::PostBatch(&config_); }

  void UpdateTSR2TSR(const std::string& /*name*/, const tsr_t& G,
                     const tsr_t& W, tsr_t* D,
                     WePSOptimizerTSRSlot* slot) const override {
    ll_we_ps_optimizer_t::UpdateTSR2TSR(config_, G, W, D, &slot->O[0]);
  }

  void UpdateSRM2TSR(const std::string& /*name*/, const srm_t& G,
                     const tsr_t& W, tsr_t* D,
                     WePSOptimizerTSRSlot* slot) const override {
    ll_we_ps_optimizer_t::UpdateSRM2TSR(config_, G, W, D, &slot->O[0]);
  }

  void UpdateSRM2SRM(const std::string& /*name*/, const srm_t& G,
                     const srm_t& W, srm_t* D,
                     WePSOptimizerSRMSlot* slot) const override {
    ll_we_ps_optimizer_t::UpdateSRM2SRM(config_, G, W, D, &slot->O[0]);
  }
};

WE_PS_OPTIMIZER_REGISTER(WePSAdaGradOptimizer, "WePSAdaGradOptimizer");
WE_PS_OPTIMIZER_REGISTER(WePSAdaGradOptimizer, "ada_grad");
WE_PS_OPTIMIZER_REGISTER(WePSAdaGradOptimizer, "adagrad");

}  // namespace deepx_core
