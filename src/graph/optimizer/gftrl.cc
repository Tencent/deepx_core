// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/optimizer_impl.h>

namespace deepx_core {

class GFTRLOptimizer : public OptimizerBase2 {
 private:
  ll_optimizer_t::GFTRLConfig config_;

 public:
  DEFINE_OPTIMIZER_LIKE(GFTRLOptimizer);

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
    } else if (k == "lambda") {
      config_.lambda = (float_t)std::stod(v);
      if (config_.lambda <= 0) {
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
    os << config_.alpha << config_.beta << config_.lambda;
  }

  void ReadConfigLegacy(InputStream& is) override {
    is >> config_.alpha >> config_.beta >> config_.lambda;
    config_.inv_alpha = 1 / config_.alpha;
  }

  void CopyConfigLegacy(const Optimizer& other) override {
    config_ = ((const GFTRLOptimizer&)other).config_;
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

OPTIMIZER_REGISTER(GFTRLOptimizer, "GFTRLOptimizer");
OPTIMIZER_REGISTER(GFTRLOptimizer, "gftrl");

}  // namespace deepx_core
