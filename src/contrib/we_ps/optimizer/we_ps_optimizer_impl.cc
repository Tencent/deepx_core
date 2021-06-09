// Copyright 2021 the deepx authors.
// Author: Xingfei Li (xingfeili@tencent.com)
//

#include <deepx_core/contrib/we_ps/optimizer/we_ps_optimizer_impl.h>

namespace deepx_core {

/************************************************************************/
/* WePSOptimizerImpl */
/************************************************************************/
void WePSOptimizerImpl::Init(const Graph* graph, TensorMap* param) {
  graph_ = graph;
  param_ = param;
}

bool WePSOptimizerImpl::InitConfig(const AnyMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const auto& v = entry.second.to_ref<std::string>();
    if (k == "cache_size") {
      InitCache((size_t)std::stoull(v));
      continue;
    }
    if (!InitConfigKV(k, v)) {
      return false;
    }
  }

  if (!PostInitConfig()) {
    return false;
  }

  AnyMapToStringMap(config, &config_);
  return true;
}

bool WePSOptimizerImpl::InitConfig(const StringMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const std::string& v = entry.second;
    if (k == "cache_size") {
      InitCache((size_t)std::stoull(v));
      continue;
    }
    if (!InitConfigKV(k, v)) {
      return false;
    }
  }

  if (!PostInitConfig()) {
    return false;
  }

  config_ = config;
  return true;
}

bool WePSOptimizerImpl::InitParam() {
  DXINFO("Initializing WePSOptimizer...");
  for (const auto& entry : *param_) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      DXINFO("Initializing TSR %s...", name.c_str());
      const auto& W = Wany.unsafe_to_ref<tsr_t>();
      InitParamTSR(name, W, &tsr_slot_map_[name]);
    } else if (Wany.is<srm_t>()) {
      DXINFO("Initializing SRM %s...", name.c_str());
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      InitParamSRM(name, W, &srm_slot_map_[name]);
    }
  }
  DXINFO("Done.");
  return true;
}

bool WePSOptimizerImpl::Write(OutputStream& os) const {
  int version = 0;
  os << version;
  os << config_ << tsr_slot_map_;
  // no 'srm_slot_map_'
  if (!os) {
    DXERROR("Failed to write WePSOptimizer.");
    return false;
  }
  return true;
}

bool WePSOptimizerImpl::Read(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read WePSOptimizer.");
    return false;
  }

  if (version > 0) {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }

  is >> config_ >> tsr_slot_map_;
  if (!is) {
    DXERROR("Failed to read WePSOptimizer.");
    return false;
  }
  for (const auto& entry : *param_) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      DXINFO("Initializing SRM %s...", name.c_str());
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      InitParamSRM(name, W, &srm_slot_map_[name]);
    }
  }

  if (!InitConfig(config_)) {
    return false;
  }
  return true;
}

void WePSOptimizerImpl::Update(TensorMap* grad, TensorMap* delta_param) {
  PreUpdate();
  for (auto& entry : *param_) {
    const std::string& name = entry.first;
    auto it = grad->find(name);
    if (it == grad->end()) {
      continue;
    }

    auto it1 = delta_param->find(name);
    if (it1 == delta_param->end()) {
      continue;
    }

    Any& Wany = entry.second;
    Any& Gany = it->second;
    Any& Dany = it1->second;

    Update(name, &Gany, &Wany, &Dany);
  }
  PostUpdate();

  if (++update_times_ == 10000) {  // magic number
    UpdateCache();
    update_times_ = 0;
  }
}

void WePSOptimizerImpl::ForEachSRM(
    const std::function<void(const std::string&, srm_t*)>& func) {
  for (auto& entry : srm_slot_map_) {
    const std::string& name = entry.first;
    WePSOptimizerSRMSlot& slot = entry.second;
    for (auto& W : slot.O) {
      func(name, &W);
    }
  }
}

void WePSOptimizerImpl::Update(const std::string& name, Any* Gany, Any* Wany,
                               Any* Dany) {
  if (Wany->is<tsr_t>()) {
    auto& tsr_slot = tsr_slot_map_[name];
    auto& W = Wany->unsafe_to_ref<tsr_t>();
    auto& D = Dany->unsafe_to_ref<tsr_t>();
    if (Gany->is<tsr_t>()) {
      auto& G = Gany->unsafe_to_ref<tsr_t>();
      ll_we_ps_optimizer_t::Clip(&G);
      UpdateTSR2TSR(name, G, W, &D, &tsr_slot);
    } else if (Gany->is<srm_t>()) {
      auto& G = Gany->unsafe_to_ref<srm_t>();
      ll_we_ps_optimizer_t::Clip(&G);
      UpdateSRM2TSR(name, G, W, &D, &tsr_slot);
    }
  } else if (Wany->is<srm_t>()) {
    auto& srm_slot = srm_slot_map_[name];
    auto& W = Wany->unsafe_to_ref<srm_t>();
    auto& D = Dany->unsafe_to_ref<srm_t>();
    if (Gany->is<srm_t>()) {
      auto& G = Gany->unsafe_to_ref<srm_t>();
      for (auto& entry : G) {
        active_ids_.emplace(entry.first);
      }
      ll_we_ps_optimizer_t::Clip(&G);
      UpdateSRM2SRM(name, G, W, &D, &srm_slot);
    }
  }
}

void WePSOptimizerImpl::InitCache(size_t cache_size) {
  cache_.set_evict_callback(
      [this](const srm_t::key_type& key, const bool& /*value*/) {
        evicted_ids_.emplace(key);
      });
  cache_.init(cache_size);
}

void WePSOptimizerImpl::UpdateCache() {
  evicted_ids_.clear();
  for (int_t id : active_ids_) {
    cache_.get_or_insert(id);
  }
  active_ids_.clear();
  DXINFO("Cache hit rate is: %.2f.", cache_.hit_rate());
  if (!evicted_ids_.empty()) {
    auto filter = [this](const std::string& /*name*/, srm_t* W) {
      W->remove_if([this](const srm_t::value_type& entry) {
        return evicted_ids_.count(entry.first) > 0;
      });
    };
    ForEachSRM(filter);
  }
}

/************************************************************************/
/* WePSOptimizerBase1 */
/************************************************************************/
void WePSOptimizerBase1::InitParamTSR(const std::string& /*name*/,
                                      const tsr_t& W,
                                      WePSOptimizerTSRSlot* slot) const {
  slot->O.resize(1);
  slot->O[0].resize(W.shape());
  slot->O[0].zeros();
}

void WePSOptimizerBase1::InitParamSRM(const std::string& /*name*/,
                                      const srm_t& W,
                                      WePSOptimizerSRMSlot* slot) const {
  slot->O.resize(1);
  slot->O[0].clear();
  slot->O[0].set_col(W.col());
  slot->O[0].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
}

/************************************************************************/
/* WePSOptimizerBase2 */
/************************************************************************/
void WePSOptimizerBase2::InitParamTSR(const std::string& /*name*/,
                                      const tsr_t& W,
                                      WePSOptimizerTSRSlot* slot) const {
  slot->O.resize(2);
  slot->O[0].resize(W.shape());
  slot->O[0].zeros();
  slot->O[1].resize(W.shape());
  slot->O[1].zeros();
}

void WePSOptimizerBase2::InitParamSRM(const std::string& /*name*/,
                                      const srm_t& W,
                                      WePSOptimizerSRMSlot* slot) const {
  slot->O.resize(2);
  slot->O[0].clear();
  slot->O[0].set_col(W.col());
  slot->O[0].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  slot->O[1].clear();
  slot->O[1].set_col(W.col());
  slot->O[1].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
}

/************************************************************************/
/* WePSOptimizer functions */
/************************************************************************/
std::unique_ptr<WePSOptimizer> NewWePSOptimizer(const std::string& name) {
  std::unique_ptr<WePSOptimizer> optimizer(WE_PS_OPTIMIZER_NEW(name));
  if (!optimizer) {
    DXERROR("Invalid WePSOptimizer name: %s.", name.c_str());
    DXERROR("WePSOptimizer name can be:");
    for (const std::string& _name : WE_PS_OPTIMIZER_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return optimizer;
}

bool SaveWePSOptimizer(const std::string& file,
                       const WePSOptimizer& optimizer) {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  DXINFO("Saving WePSOptimizer to %s...", file.c_str());
  std::string name = optimizer.class_name();
  os << name;
  if (!os) {
    DXERROR("Failed to write WePSOptimizer.");
    return false;
  }

  if (!optimizer.Write(os)) {
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool LoadWePSOptimizer(const std::string& file, WePSOptimizer* optimizer) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  DXINFO("Loading WePSOptimizer from %s...", file.c_str());
  std::string name;
  is >> name;
  if (!is) {
    DXERROR("Failed to read WePSOptimizer.");
    return false;
  }

  if (!optimizer->Read(is)) {
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool LoadWePSOptimizerName(const std::string& file, std::string* name) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  DXINFO("Loading WePSOptimizer name from %s...", file.c_str());
  is >> *name;
  if (!is) {
    DXERROR("Failed to read WePSOptimizer name.");
    return false;
  }
  return true;
}

}  // namespace deepx_core
