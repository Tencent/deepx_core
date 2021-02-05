// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/optimizer_impl.h>

namespace deepx_core {

class SGDOptimizer : public OptimizerBase0 {
 private:
  ll_optimizer_t::SGDConfig config_;

 public:
  DEFINE_OPTIMIZER_LIKE(SGDOptimizer);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (k == "alpha") {
      config_.alpha = (float_t)std::stod(v);
      if (config_.alpha <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "min_alpha") {
      config_.min_alpha = (float_t)std::stod(v);
      if (config_.min_alpha <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "batch_decay") {
      config_.batch_decay = std::stoi(v);
      if (config_.batch_decay < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "batch_decay_rate") {
      config_.batch_decay_rate = (float_t)std::stod(v);
      if (config_.batch_decay_rate <= 0 || config_.batch_decay_rate > 1) {
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
    os << config_.alpha << config_.min_alpha << config_.batch_decay
       << config_.batch_decay_rate << config_.real_batch << config_.real_alpha;
  }

  void ReadConfigLegacy(InputStream& is) override {
    is >> config_.alpha >> config_.min_alpha >> config_.batch_decay >>
        config_.batch_decay_rate >> config_.real_batch >> config_.real_alpha;
  }

  void CopyConfigLegacy(const Optimizer& other) override {
    config_ = ((const SGDOptimizer&)other).config_;
  }

  void PreUpdate() override { ll_optimizer_t::PreBatch(&config_); }

  void PostUpdate() override { ll_optimizer_t::PostBatch(&config_); }

  void UpdateTSR2TSR(const std::string& /*name*/, const tsr_t& G, tsr_t* W,
                     OptimizerTSRSlot* /*slot*/) const override {
    ll_optimizer_t::UpdateTSR2TSR(config_, G, W);
  }

  void UpdateSRM2TSR(const std::string& /*name*/, const srm_t& G, tsr_t* W,
                     OptimizerTSRSlot* /*slot*/) const override {
    ll_optimizer_t::UpdateSRM2TSR(config_, G, W);
  }

  void UpdateSRM2SRM(const std::string& /*name*/, const srm_t& G, srm_t* W,
                     OptimizerSRMSlot* slot) const override {
    if (use_lock_) {
      ll_optimizer_t::UpdateSRM2SRM(config_, G, W, slot->Wlock.get());
    } else {
      ll_optimizer_t::UpdateSRM2SRM(config_, G, W);
    }
  }
};

OPTIMIZER_REGISTER(SGDOptimizer, "SGDOptimizer");
OPTIMIZER_REGISTER(SGDOptimizer, "sgd");

}  // namespace deepx_core
