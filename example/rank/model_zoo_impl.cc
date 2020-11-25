// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "model_zoo_impl.h"

namespace deepx_core {

/************************************************************************/
/* ModelZooImpl */
/************************************************************************/
bool ModelZooImpl::InitConfig(const AnyMap& config) {
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
  return true;
}

bool ModelZooImpl::InitConfig(const StringMap& config) {
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
  return true;
}

bool ModelZooImpl::InitConfigKV(const std::string& k, const std::string& v) {
  if (k == "config" || k == "group_config") {
    if (!GuessGroupConfig(v, &items_, nullptr, k.c_str())) {
      return false;
    }
    item_is_fm_ = IsFMGroupConfig(items_) ? 1 : 0;
    item_m_ = (int)items_.size();
    if (item_is_fm_) {
      item_k_ = items_.front().embedding_col;
    } else {
      item_k_ = 0;
    }
    item_mk_ = GetTotalEmbeddingCol(items_);
  } else if (k == "w" || k == "has_w") {
    has_w_ = std::stoi(v);
  } else if (k == "sparse") {
    sparse_ = std::stoi(v);
  } else {
    return false;
  }
  return true;
}

/************************************************************************/
/* ModelZoo functions */
/************************************************************************/
std::unique_ptr<ModelZoo> NewModelZoo(const std::string& name) {
  std::unique_ptr<ModelZoo> model_zoo(MODEL_ZOO_NEW(name));
  if (!model_zoo) {
    DXERROR("Invalid model name: %s.", name.c_str());
    DXERROR("Model name can be:");
    for (const std::string& _name : MODEL_ZOO_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return model_zoo;
}

}  // namespace deepx_core
