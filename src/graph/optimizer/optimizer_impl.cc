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

bool OptimizerImpl::WriteLegacy(OutputStream& os) const {
  int version = 1;
  os << version;
  WriteConfigLegacy(os);
  os << tsr_slot_map_ << srm_slot_map_;
  if (!os) {
    DXERROR("Failed to write optimizer.");
    return false;
  }
  return true;
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

bool OptimizerImpl::ReadLegacy(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read optimizer.");
    return false;
  }

  if (version == 0) {
    std::unordered_map<std::string, OptimizerSRMSlot> svp_slot_map;
    ReadConfigLegacy(is);
    is >> tsr_slot_map_;
    ReadOptimizerSRPSlotMap(is, srm_slot_map_);
    ReadOptimizerSVPSlotMap(is, svp_slot_map);
    if (!is) {
      DXERROR("Failed to read optimizer.");
      return false;
    }
    for (auto& entry : svp_slot_map) {
      srm_slot_map_.emplace(std::move(entry));
    }
    return true;
  } else if (version == 1) {
    ReadConfigLegacy(is);
    is >> tsr_slot_map_ >> srm_slot_map_;
    if (!is) {
      DXERROR("Failed to read optimizer.");
      return false;
    }
    return true;
  } else {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }
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

bool OptimizerImpl::MergeLegacy(Optimizer* other, const Shard* shard,
                                int shard_id) {
  DXINFO("Merging optimizer...");
  auto config_reduce_func = [this, other](StringMap& /*local_config*/,
                                          StringMap& /*remote_config*/) {
    DXINFO("Merging config...");
    CopyConfigLegacy(*other);
  };
  auto tsr_reduce_func = [](const std::string& name, tsr_t& local_W,
                            tsr_t& remote_W) {
    DXINFO("Merging TSR %s...", name.c_str());
    local_W = std::move(remote_W);
  };
  auto srm_reduce_func = [shard, shard_id](const std::string& name,
                                           srm_t& local_W, srm_t& remote_W) {
    DXINFO("Merging SRM %s...", name.c_str());
    local_W.merge_if(
        std::move(remote_W), [shard, shard_id](const srm_t::value_type& entry) {
          return shard == nullptr || shard->HasSRM(shard_id, entry.first);
        });
  };
  if (!Reduce(other, config_reduce_func, tsr_reduce_func, srm_reduce_func,
              shard, shard_id)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool OptimizerImpl::Merge(Optimizer* other, const Shard* shard, int shard_id) {
  DXINFO("Merging optimizer...");
  auto config_reduce_func = [this](StringMap& /*local_config*/,
                                   StringMap& remote_config) {
    DXINFO("Merging config...");
    (void)InitConfig(remote_config);
  };
  auto tsr_reduce_func = [](const std::string& name, tsr_t& local_W,
                            tsr_t& remote_W) {
    DXINFO("Merging TSR %s...", name.c_str());
    local_W = std::move(remote_W);
  };
  auto srm_reduce_func = [shard, shard_id](const std::string& name,
                                           srm_t& local_W, srm_t& remote_W) {
    DXINFO("Merging SRM %s...", name.c_str());
    local_W.merge_if(
        std::move(remote_W), [shard, shard_id](const srm_t::value_type& entry) {
          return shard == nullptr || shard->HasSRM(shard_id, entry.first);
        });
  };
  if (!Reduce(other, config_reduce_func, tsr_reduce_func, srm_reduce_func,
              shard, shard_id)) {
    return false;
  }
  DXINFO("Done.");
  return true;
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
      if (W.same_shape(G)) {
        ll_optimizer_t::Clip(&G);
        UpdateTSR2TSR(name, G, &W, &tsr_slot_map_[name]);
      }
    } else if (Gany->is<srm_t>()) {
      auto& G = Gany->unsafe_to_ref<srm_t>();
      if (W.dim(1) == G.col()) {
        ll_optimizer_t::Clip(&G);
        UpdateSRM2TSR(name, G, &W, &tsr_slot_map_[name]);
      }
    }
  } else if (Wany->is<srm_t>()) {
    auto& W = Wany->unsafe_to_ref<srm_t>();
    if (Gany->is<srm_t>()) {
      auto& G = Gany->unsafe_to_ref<srm_t>();
      if (W.col() == G.col()) {
        ll_optimizer_t::Clip(&G);
        UpdateSRM2SRM(name, G, &W, &srm_slot_map_[name]);
      }
    }
  }
}

bool OptimizerImpl::Reduce(Optimizer* other,
                           const config_reduce_func_t& config_reduce_func,
                           const tsr_reduce_func_t& tsr_reduce_func,
                           const srm_reduce_func_t& srm_reduce_func,
                           const Shard* shard, int shard_id) {
  if (std::string(class_name()) != other->class_name()) {
    DXERROR("Inconsistent class name: %s vs %s.", class_name(),
            other->class_name());
    return false;
  }

  config_reduce_func(config_, ((OptimizerImpl*)other)->config_);

  for (auto& entry : ((OptimizerImpl*)other)->tsr_slot_map_) {
    const std::string& name = entry.first;
    auto it = tsr_slot_map_.find(name);
    if (it == tsr_slot_map_.end()) {
      continue;
    }

    OptimizerTSRSlot& local_slot = it->second;
    OptimizerTSRSlot& remote_slot = entry.second;
    if (local_slot.O.size() == remote_slot.O.size() &&
        (shard == nullptr || shard->HasTSR(shard_id, name))) {
      for (size_t i = 0; i < local_slot.O.size(); ++i) {
        if (local_slot.O[i].same_shape(remote_slot.O[i])) {
          tsr_reduce_func(name, local_slot.O[i], remote_slot.O[i]);
        }
      }
    }
  }

  for (auto& entry : ((OptimizerImpl*)other)->srm_slot_map_) {
    const std::string& name = entry.first;
    auto it = srm_slot_map_.find(name);
    if (it == srm_slot_map_.end()) {
      continue;
    }

    OptimizerSRMSlot& local_slot = it->second;
    OptimizerSRMSlot& remote_slot = entry.second;
    if (local_slot.O.size() == remote_slot.O.size()) {
      for (size_t i = 0; i < local_slot.O.size(); ++i) {
        if (local_slot.O[i].col() == remote_slot.O[i].col()) {
          srm_reduce_func(name, local_slot.O[i], remote_slot.O[i]);
        }
      }
    }
  }
  return true;
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

bool SaveOptimizerLegacy(const std::string& file, const Optimizer& optimizer) {
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

  if (!optimizer.WriteLegacy(os)) {
    return false;
  }

  DXINFO("Done.");
  return true;
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

std::unique_ptr<Optimizer> LoadOptimizerLegacy(const std::string& file) {
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

  std::unique_ptr<Optimizer> optimizer(NewOptimizer(name));
  if (!optimizer) {
    return nullptr;
  }

  if (!optimizer->ReadLegacy(is)) {
    return nullptr;
  }

  DXINFO("Done.");
  return optimizer;
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

  std::unique_ptr<Optimizer> optimizer(NewOptimizer(name));
  if (!optimizer) {
    return nullptr;
  }

  if (!optimizer->Read(is)) {
    return nullptr;
  }

  DXINFO("Done.");
  return optimizer;
}

bool LoadOptimizerName(const std::string& file, std::string* name) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  DXINFO("Loading optimizer name from %s...", file.c_str());
  is >> *name;
  if (!is) {
    DXERROR("Failed to read optimizer name.");
    return false;
  }
  return true;
}

}  // namespace deepx_core
