// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/optimizer_impl.h>

namespace deepx_core {

class Hybrid2Optimizer : public OptimizerImpl {
 private:
  ll_optimizer_t::AdaGradConfig ada_grad_config_;
  ll_optimizer_t::GFTRLConfig gftrl_config_;

 public:
  DEFINE_OPTIMIZER_LIKE(Hybrid2Optimizer);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (k == "ada_grad_alpha") {
      ada_grad_config_.alpha = (float_t)std::stod(v);
      if (ada_grad_config_.alpha <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "ada_grad_beta") {
      ada_grad_config_.beta = (float_t)std::stod(v);
      if (ada_grad_config_.beta <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "gftrl_alpha") {
      gftrl_config_.alpha = (float_t)std::stod(v);
      if (gftrl_config_.alpha <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "gftrl_beta") {
      gftrl_config_.beta = (float_t)std::stod(v);
      if (gftrl_config_.beta <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "gftrl_lambda") {
      gftrl_config_.lambda = (float_t)std::stod(v);
      if (gftrl_config_.lambda <= 0) {
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
    ll_optimizer_t::Init(&ada_grad_config_);
    ll_optimizer_t::Init(&gftrl_config_);
    return true;
  }

  void WriteConfigLegacy(OutputStream& os) const override {
    int version = 0;
    os << version;
    os << ada_grad_config_.alpha << ada_grad_config_.beta;
    os << gftrl_config_.alpha << gftrl_config_.beta << gftrl_config_.lambda;
  }

  void ReadConfigLegacy(InputStream& is) override {
    int version;
    is >> version;
    if (!is) {
      DXERROR("Failed to read config.");
      return;
    }
    if (version > 0) {
      DXERROR("Couldn't handle a higher version: %d.", version);
      is.set_bad();
      return;
    }
    is >> ada_grad_config_.alpha >> ada_grad_config_.beta;
    is >> gftrl_config_.alpha >> gftrl_config_.beta >> gftrl_config_.lambda;
    gftrl_config_.inv_alpha = 1 / gftrl_config_.alpha;
  }

  void CopyConfigLegacy(const Optimizer& other) override {
    ada_grad_config_ = ((const Hybrid2Optimizer&)other).ada_grad_config_;
    gftrl_config_ = ((const Hybrid2Optimizer&)other).gftrl_config_;
  }

  void InitParamTSR(const std::string& /*name*/, const tsr_t& W,
                    OptimizerTSRSlot* slot) const override {
    slot->O.resize(1);
    slot->O[0].resize(W.shape());
    slot->O[0].zeros();
  }

  void InitParamSRM(const std::string& /*name*/, const srm_t& W,
                    OptimizerSRMSlot* slot) const override {
    slot->O.resize(2);
    slot->O[0].clear();
    slot->O[0].set_col(W.col());
    slot->O[0].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
    slot->O[1].clear();
    slot->O[1].set_col(W.col());
    slot->O[1].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  }

  void PreUpdate() override {
    ll_optimizer_t::PreBatch(&ada_grad_config_);
    ll_optimizer_t::PreBatch(&gftrl_config_);
  }

  void PostUpdate() override {
    ll_optimizer_t::PostBatch(&ada_grad_config_);
    ll_optimizer_t::PostBatch(&gftrl_config_);
  }

  void UpdateTSR2TSR(const std::string& /*name*/, const tsr_t& G, tsr_t* W,
                     OptimizerTSRSlot* slot) const override {
    ll_optimizer_t::UpdateTSR2TSR(ada_grad_config_, G, W, &slot->O[0]);
  }

  void UpdateSRM2TSR(const std::string& /*name*/, const srm_t& G, tsr_t* W,
                     OptimizerTSRSlot* slot) const override {
    ll_optimizer_t::UpdateSRM2TSR(ada_grad_config_, G, W, &slot->O[0]);
  }

  void UpdateSRM2SRM(const std::string& /*name*/, const srm_t& G, srm_t* W,
                     OptimizerSRMSlot* slot) const override {
    if (use_lock_) {
      ll_optimizer_t::UpdateSRM2SRM(gftrl_config_, G, W, &slot->O[0],
                                    &slot->O[1], slot->Wlock.get(),
                                    slot->Olock[0].get(), slot->Olock[1].get());
    } else {
      ll_optimizer_t::UpdateSRM2SRM(gftrl_config_, G, W, &slot->O[0],
                                    &slot->O[1]);
    }
  }
};

OPTIMIZER_REGISTER(Hybrid2Optimizer, "Hybrid2Optimizer");
OPTIMIZER_REGISTER(Hybrid2Optimizer, "hybrid2");

}  // namespace deepx_core
