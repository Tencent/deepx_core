// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/optimizer_impl.h>

namespace deepx_core {

class AdaDeltaOptimizer : public OptimizerBase2 {
 private:
  ll_optimizer_t::AdaDeltaConfig config_;

 public:
  DEFINE_OPTIMIZER_LIKE(AdaDeltaOptimizer);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (k == "rho") {
      config_.rho = (float_t)std::stod(v);
      if (config_.rho <= 0 || config_.rho > 1) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "alpha") {
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
    ll_optimizer_t::Init(&config_);
    return true;
  }

  void WriteConfigLegacy(OutputStream& os) const override {
    os << config_.rho << config_.alpha << config_.beta;
  }

  void ReadConfigLegacy(InputStream& is) override {
    is >> config_.rho >> config_.alpha >> config_.beta;
    config_.one_sub_rho = 1 - config_.rho;
  }

  void CopyConfigLegacy(const Optimizer& other) override {
    config_ = ((const AdaDeltaOptimizer&)other).config_;
  }

  void PreUpdate() override { ll_optimizer_t::PreBatch(&config_); }

  void PostUpdate() override { ll_optimizer_t::PostBatch(&config_); }

  void UpdateTSR2TSR(const std::string& /*name*/, const tsr_t& G, tsr_t* W,
                     OptimizerTSRSlot* slot) const override {
    ll_optimizer_t::UpdateTSR2TSR(config_, G, W, &slot->O[0], &slot->O[1]);
  }

  void UpdateSRM2TSR(const std::string& /*name*/, const srm_t& G, tsr_t* W,
                     OptimizerTSRSlot* slot) const override {
    ll_optimizer_t::UpdateSRM2TSR(config_, G, W, &slot->O[0], &slot->O[1]);
  }

  void UpdateSRM2SRM(const std::string& /*name*/, const srm_t& G, srm_t* W,
                     OptimizerSRMSlot* slot) const override {
    if (use_lock_) {
      ll_optimizer_t::UpdateSRM2SRM(config_, G, W, &slot->O[0], &slot->O[1],
                                    slot->Wlock.get(), slot->Olock[0].get(),
                                    slot->Olock[1].get());
    } else {
      ll_optimizer_t::UpdateSRM2SRM(config_, G, W, &slot->O[0], &slot->O[1]);
    }
  }
};

OPTIMIZER_REGISTER(AdaDeltaOptimizer, "AdaDeltaOptimizer");
OPTIMIZER_REGISTER(AdaDeltaOptimizer, "ada_delta");
OPTIMIZER_REGISTER(AdaDeltaOptimizer, "adadelta");

}  // namespace deepx_core
