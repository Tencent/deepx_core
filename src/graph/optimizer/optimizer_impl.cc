// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/optimizer_impl.h>

namespace deepx_core {

/************************************************************************/
/* OptimizerImpl */
/************************************************************************/
void OptimizerImpl::Init(const Graph* graph, TensorMap* param) {
  graph_ = graph;
  param_ = param;
}

bool OptimizerImpl::InitConfig(const AnyMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const auto& v = entry.second.to_ref<std::string>();
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

bool OptimizerImpl::InitConfig(const StringMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const std::string& v = entry.second;
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

bool OptimizerImpl::InitParam() {
  DXINFO("Initializing optimizer...");
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

void OptimizerImpl::InitLock(AnyMap* param_lock) {
  use_lock_ = 1;
  for (auto& entry : srm_slot_map_) {
    const std::string& name = entry.first;
    OptimizerSRMSlot& slot = entry.second;
    slot.Wlock = param_lock->unsafe_get<std::shared_ptr<ReadWriteLock>>(name);
    slot.Olock.resize(slot.O.size());
    for (size_t i = 0; i < slot.O.size(); ++i) {
      slot.Olock[i].reset(new ReadWriteLock);
    }
  }
}

bool OptimizerImpl::Write(OutputStream& os) const {
  int version = 0;
  os << version;
  os << config_ << tsr_slot_map_ << srm_slot_map_;
  if (!os) {
    DXERROR("Failed to write optimizer.");
    return false;
  }
  return true;
}

bool OptimizerImpl::Read(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read optimizer.");
    return false;
  }

  if (version > 0) {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }

  is >> config_ >> tsr_slot_map_ >> srm_slot_map_;
  if (!is) {
    DXERROR("Failed to read optimizer.");
    return false;
  }

  if (!InitConfig(config_)) {
    return false;
  }
  return true;
}

void OptimizerImpl::Merge(Optimizer* other, const Shard* shard) {
  DXINFO("Merging optimizer slot...");
  auto merge_tsr_slot = [](const std::string& name,
                           OptimizerTSRSlot& local_slot,
                           OptimizerTSRSlot& remote_slot) {
    DXINFO("Merging TSR %s...", name.c_str());
    for (size_t i = 0; i < local_slot.O.size(); ++i) {
      local_slot.O[i] = std::move(remote_slot.O[i]);
    }
  };
  auto merge_srm_slot = [shard](const std::string& name,
                                OptimizerSRMSlot& local_slot,
                                OptimizerSRMSlot& remote_slot) {
    DXINFO("Merging SRM %s...", name.c_str());
    for (size_t i = 0; i < local_slot.O.size(); ++i) {
      local_slot.O[i].set_col(remote_slot.O[i].col());
      local_slot.O[i].merge_if(std::move(remote_slot.O[i]),
                               [shard](const srm_t::value_type& entry) {
                                 return !shard || shard->HasSRM(entry.first);
                               });
    }
  };

  for (auto& entry : ((OptimizerImpl*)other)->tsr_slot_map_) {
    const std::string& name = entry.first;
    OptimizerTSRSlot& remote_slot = entry.second;
    auto it = tsr_slot_map_.find(name);
    if (it != tsr_slot_map_.end()) {
      OptimizerTSRSlot& local_slot = it->second;
      merge_tsr_slot(name, local_slot, remote_slot);
    } else if (!shard || shard->HasTSR(name)) {
      OptimizerTSRSlot& local_slot = tsr_slot_map_[name];
      local_slot.O.resize(remote_slot.O.size());
      merge_tsr_slot(name, local_slot, remote_slot);
    }
  }

  for (auto& entry : ((OptimizerImpl*)other)->srm_slot_map_) {
    const std::string& name = entry.first;
    OptimizerSRMSlot& remote_slot = entry.second;
    OptimizerSRMSlot& local_slot = srm_slot_map_[name];
    local_slot.O.resize(remote_slot.O.size());
    merge_srm_slot(name, local_slot, remote_slot);
  }
  DXINFO("Done.");
}

void OptimizerImpl::Warmup(Optimizer* other) {
  DXINFO("Warming up optimizer...");
  auto reduce_tsr = [](const std::string& name, tsr_t& local_W,
                       tsr_t& remote_W) {
    DXINFO("Warming up TSR %s...", name.c_str());
    local_W = std::move(remote_W);
  };
  auto reduce_srm = [](const std::string& name, srm_t& local_W,
                       srm_t& remote_W) {
    DXINFO("Warming up SRM %s...", name.c_str());
    local_W.merge(std::move(remote_W));
  };
  Reduce(other, reduce_tsr, reduce_srm);
  DXINFO("Done.");
}

void OptimizerImpl::Update(TensorMap* grad) {
  PreUpdate();

  for (auto& entry : *param_) {
    const std::string& name = entry.first;
    auto it = grad->find(name);
    if (it == grad->end()) {
      continue;
    }

    Any& Wany = entry.second;
    Any& Gany = it->second;
    UpdateParam(name, &Gany, &Wany);
  }

  PostUpdate();
}

void OptimizerImpl::ForEachSRM(
    const std::function<void(const std::string&, srm_t*)>& func) {
  for (auto& entry : srm_slot_map_) {
    const std::string& name = entry.first;
    OptimizerSRMSlot& slot = entry.second;
    for (auto& W : slot.O) {
      func(name, &W);
    }
  }
}

void OptimizerImpl::UpdateParam(const std::string& name, Any* Gany, Any* Wany) {
  if (Wany->is<tsr_t>()) {
    auto& W = Wany->unsafe_to_ref<tsr_t>();
    if (Gany->is<tsr_t>()) {
      auto& G = Gany->unsafe_to_ref<tsr_t>();
      ll_optimizer_t::Clip(&G);
      UpdateTSR2TSR(name, G, &W, &tsr_slot_map_[name]);
    } else if (Gany->is<srm_t>()) {
      auto& G = Gany->unsafe_to_ref<srm_t>();
      ll_optimizer_t::Clip(&G);
      UpdateSRM2TSR(name, G, &W, &tsr_slot_map_[name]);
    }
  } else if (Wany->is<srm_t>()) {
    auto& W = Wany->unsafe_to_ref<srm_t>();
    if (Gany->is<srm_t>()) {
      auto& G = Gany->unsafe_to_ref<srm_t>();
      ll_optimizer_t::Clip(&G);
      UpdateSRM2SRM(name, G, &W, &srm_slot_map_[name]);
    }
  }
}

/************************************************************************/
/* OptimizerBase1 */
/************************************************************************/
void OptimizerBase1::InitParamTSR(const std::string& /*name*/, const tsr_t& W,
                                  OptimizerTSRSlot* slot) const {
  slot->O.resize(1);
  slot->O[0].resize(W.shape());
  slot->O[0].zeros();
}

void OptimizerBase1::InitParamSRM(const std::string& /*name*/, const srm_t& W,
                                  OptimizerSRMSlot* slot) const {
  slot->O.resize(1);
  slot->O[0].clear();
  slot->O[0].set_col(W.col());
  slot->O[0].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
}

/************************************************************************/
/* OptimizerBase2 */
/************************************************************************/
void OptimizerBase2::InitParamTSR(const std::string& /*name*/, const tsr_t& W,
                                  OptimizerTSRSlot* slot) const {
  slot->O.resize(2);
  slot->O[0].resize(W.shape());
  slot->O[0].zeros();
  slot->O[1].resize(W.shape());
  slot->O[1].zeros();
}

void OptimizerBase2::InitParamSRM(const std::string& /*name*/, const srm_t& W,
                                  OptimizerSRMSlot* slot) const {
  slot->O.resize(2);
  slot->O[0].clear();
  slot->O[0].set_col(W.col());
  slot->O[0].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  slot->O[1].clear();
  slot->O[1].set_col(W.col());
  slot->O[1].set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
}

/************************************************************************/
/* Optimizer functions */
/************************************************************************/
std::unique_ptr<Optimizer> NewOptimizer(const std::string& name) {
  std::unique_ptr<Optimizer> optimizer(OPTIMIZER_NEW(name));
  if (!optimizer) {
    DXERROR("Invalid optimizer name: %s.", name.c_str());
    DXERROR("Optimizer name can be:");
    for (const std::string& _name : OPTIMIZER_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return optimizer;
}

bool SaveOptimizer(const std::string& file, const Optimizer& optimizer) {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  DXINFO("Saving optimizer to %s...", file.c_str());
  std::string name = optimizer.class_name();
  os << name;
  if (!os) {
    DXERROR("Failed to write optimizer.");
    return false;
  }

  if (!optimizer.Write(os)) {
    return false;
  }

  DXINFO("Done.");
  return true;
}

std::unique_ptr<Optimizer> LoadOptimizer(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return nullptr;
  }

  DXINFO("Loading optimizer from %s...", file.c_str());
  std::string name;
  is >> name;
  if (!is) {
    DXERROR("Failed to read optimizer.");
    return nullptr;
  }

  std::unique_ptr<Optimizer> optimizer = NewOptimizer(name);
  if (!optimizer) {
    return nullptr;
  }

  if (!optimizer->Read(is)) {
    return nullptr;
  }

  DXINFO("Done.");
  return optimizer;
}

}  // namespace deepx_core
