// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/model_shard.h>
#include <utility>

namespace deepx_core {

std::string ModelShard::GetSuffixLegacy(const Shard* shard, int shard_id) {
  return std::to_string(shard_id) + "." + std::to_string(shard->shard_size()) +
         ".-2.1";
}

std::string ModelShard::GetSuffix(const Shard* shard, int shard_id) {
  return shard->shard_mode() == 0 ? "" : "." + std::to_string(shard_id);
}

std::string ModelShard::GetModelFileLegacy(const std::string& dir,
                                           const Shard* shard, int shard_id) {
  return dir + "/param.bin." + GetSuffixLegacy(shard, shard_id);
}

std::string ModelShard::GetModelFile(const std::string& dir, const Shard* shard,
                                     int shard_id) {
  return dir + "/model.bin" + GetSuffix(shard, shard_id);
}

std::string ModelShard::GetTextModelFile(const std::string& dir,
                                         const Shard* shard, int shard_id) {
  return dir + "/model.txt" + GetSuffix(shard, shard_id);
}

std::string ModelShard::GetFeatureKVModelFile(const std::string& dir,
                                              const Shard* shard,
                                              int shard_id) {
  return dir + "/model.feature_kv" + GetSuffix(shard, shard_id);
}

std::string ModelShard::GetOptimizerFileLegacy(const std::string& dir,
                                               const Shard* shard,
                                               int shard_id) {
  return dir + "/optimizer.bin." + GetSuffixLegacy(shard, shard_id);
}

std::string ModelShard::GetOptimizerFile(const std::string& dir,
                                         const Shard* shard, int shard_id) {
  return dir + "/optimizer.bin" + GetSuffix(shard, shard_id);
}

std::string ModelShard::GetTSStoreFileLegacy(const std::string& dir,
                                             const Shard* shard, int shard_id) {
  return dir + "/ts.bin." + GetSuffixLegacy(shard, shard_id);
}

std::string ModelShard::GetTSStoreFile(const std::string& dir,
                                       const Shard* shard, int shard_id) {
  return dir + "/ts_store.bin" + GetSuffix(shard, shard_id);
}

std::string ModelShard::GetFreqStoreFileLegacy(const std::string& dir,
                                               const Shard* shard,
                                               int shard_id) {
  return dir + "/freq.bin." + GetSuffixLegacy(shard, shard_id);
}

std::string ModelShard::GetFreqStoreFile(const std::string& dir,
                                         const Shard* shard, int shard_id) {
  return dir + "/freq_store.bin" + GetSuffix(shard, shard_id);
}

std::string ModelShard::GetSuccessFileLegacy(const std::string& dir,
                                             const Shard* shard, int shard_id) {
  return dir + "/SUCCESS_" + GetSuffixLegacy(shard, shard_id);
}

std::string ModelShard::GetSuccessFile(const std::string& dir,
                                       const Shard* shard, int shard_id) {
  return dir + "/SUCCESS_" + GetSuffix(shard, shard_id);
}

std::string ModelShard::GetModelFileLegacy(const std::string& dir) const {
  return GetModelFileLegacy(dir, shard_, shard_id_);
}

std::string ModelShard::GetModelFile(const std::string& dir) const {
  return GetModelFile(dir, shard_, shard_id_);
}

std::string ModelShard::GetTextModelFile(const std::string& dir) const {
  return GetTextModelFile(dir, shard_, shard_id_);
}

std::string ModelShard::GetFeatureKVModelFile(const std::string& dir) const {
  return GetFeatureKVModelFile(dir, shard_, shard_id_);
}

std::string ModelShard::GetOptimizerFileLegacy(const std::string& dir) const {
  return GetOptimizerFileLegacy(dir, shard_, shard_id_);
}

std::string ModelShard::GetOptimizerFile(const std::string& dir) const {
  return GetOptimizerFile(dir, shard_, shard_id_);
}

std::string ModelShard::GetTSStoreFileLegacy(const std::string& dir) const {
  return GetTSStoreFileLegacy(dir, shard_, shard_id_);
}

std::string ModelShard::GetTSStoreFile(const std::string& dir) const {
  return GetTSStoreFile(dir, shard_, shard_id_);
}

std::string ModelShard::GetFreqStoreFileLegacy(const std::string& dir) const {
  return GetFreqStoreFileLegacy(dir, shard_, shard_id_);
}

std::string ModelShard::GetFreqStoreFile(const std::string& dir) const {
  return GetFreqStoreFile(dir, shard_, shard_id_);
}

std::string ModelShard::GetSuccessFileLegacy(const std::string& dir) const {
  return GetSuccessFileLegacy(dir, shard_, shard_id_);
}

std::string ModelShard::GetSuccessFile(const std::string& dir) const {
  return GetSuccessFile(dir, shard_, shard_id_);
}

void ModelShard::InitShard(const Shard* shard, int shard_id) noexcept {
  shard_ = shard;
  shard_id_ = shard_id;
}

void ModelShard::InitGraph(const Graph* graph) noexcept { graph_ = graph; }

bool ModelShard::InitModelPlaceholder() {
  model_.reset(new Model);
  model_->Init(graph_);
  return model_->InitParamPlaceholder();
}

bool ModelShard::InitModel() {
  model_.reset(new Model);
  model_->Init(graph_);
  return model_->InitParam(engine_, shard_, shard_id_);
}

bool ModelShard::InitOptimizer(const std::string& optimizer,
                               const std::string& optimizer_config) {
  optimizer_ = NewOptimizer(optimizer);
  if (!optimizer_) {
    return false;
  }
  optimizer_->Init(graph_, model_->mutable_param());
  if (!optimizer_->InitParam()) {
    return false;
  }
  return InitOptimizerConfig(optimizer_config);
}

bool ModelShard::InitOptimizerConfig(const std::string& optimizer_config) {
  StringMap config;
  if (!ParseConfig(optimizer_config, &config)) {
    DXERROR("Failed to parse optimizer config: %s.", optimizer_config.c_str());
    return false;
  }
  return optimizer_->InitConfig(config);
}

bool ModelShard::InitTSStore(ts_t now, ts_t expire_threshold) {
  ts_store_.reset(new TSStore);
  ts_store_->set_now(now);
  ts_store_->set_expire_threshold(expire_threshold);
  ts_store_->Init(model_->mutable_param());
  return ts_store_->InitParam();
}

bool ModelShard::InitFreqStore(freq_t freq_filter_threshold) {
  freq_store_.reset(new FreqStore);
  freq_store_->set_freq_filter_threshold(freq_filter_threshold);
  freq_store_->Init(model_->mutable_param());
  return freq_store_->InitParam();
}

bool ModelShard::InitOLStore(freq_t update_threshold,
                             float_t distance_threshold) {
  ol_store_.reset(new OLStore);
  ol_store_->set_update_threshold(update_threshold);
  ol_store_->set_distance_threshold(distance_threshold);
  ol_store_->Init(graph_, model_->mutable_param());
  return ol_store_->InitParam();
}

bool ModelShard::InitLock() {
  if (ol_store_) {
    DXERROR("OLStore does not support InitLock.");
    return false;
  }

  model_->InitLock();
  if (optimizer_) {
    optimizer_->InitLock(model_->mutable_param_lock());
  }
  if (ts_store_) {
    ts_store_->InitLock();
  }
  if (freq_store_) {
    freq_store_->InitLock();
  }
  return true;
}

bool ModelShard::SaveModelLegacy(const std::string& dir) const {
  return model_->SaveLegacy(GetModelFileLegacy(dir));
}

bool ModelShard::SaveModel(const std::string& dir) const {
  return model_->Save(GetModelFile(dir));
}

bool ModelShard::SaveTextModel(const std::string& dir) const {
  return model_->SaveText(GetTextModelFile(dir));
}

bool ModelShard::SaveFeatureKVModelLegacy(
    const std::string& dir, int feature_kv_protocol_version) const {
  return model_->SaveFeatureKV(GetModelFileLegacy(dir),
                               feature_kv_protocol_version);
}

bool ModelShard::SaveFeatureKVModel(const std::string& dir,
                                    int feature_kv_protocol_version) const {
  return model_->SaveFeatureKV(GetFeatureKVModelFile(dir),
                               feature_kv_protocol_version);
}

bool ModelShard::SaveOLFeatureKVModelLegacy(
    const std::string& dir, int feature_kv_protocol_version) const {
  return ol_store_->SaveFeatureKVModel(GetModelFileLegacy(dir),
                                       feature_kv_protocol_version);
}

bool ModelShard::SaveOLFeatureKVModel(const std::string& dir,
                                      int feature_kv_protocol_version) const {
  return ol_store_->SaveFeatureKVModel(GetFeatureKVModelFile(dir),
                                       feature_kv_protocol_version);
}

bool ModelShard::SaveOptimizerLegacy(const std::string& dir) const {
  return deepx_core::SaveOptimizerLegacy(GetOptimizerFileLegacy(dir),
                                         *optimizer_);
}

bool ModelShard::SaveOptimizer(const std::string& dir) const {
  return deepx_core::SaveOptimizer(GetOptimizerFile(dir), *optimizer_);
}

bool ModelShard::SaveTSStoreLegacy(const std::string& dir) const {
  return ts_store_->SaveLegacy(GetTSStoreFileLegacy(dir));
}

bool ModelShard::SaveTSStore(const std::string& dir) const {
  return ts_store_->Save(GetTSStoreFile(dir));
}

bool ModelShard::SaveFreqStoreLegacy(const std::string& dir) const {
  return freq_store_->Save(GetFreqStoreFileLegacy(dir));
}

bool ModelShard::SaveFreqStore(const std::string& dir) const {
  return freq_store_->Save(GetFreqStoreFile(dir));
}

bool ModelShard::SaveSuccessLegacy(const std::string& dir) const {
  std::string file = GetSuccessFileLegacy(dir);
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving SUCCESS to %s...", file.c_str());
  DXINFO("Done.");
  return true;
}

bool ModelShard::SaveSuccess(const std::string& dir) const {
  std::string file = GetSuccessFile(dir);
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving SUCCESS to %s...", file.c_str());
  DXINFO("Done.");
  return true;
}

int ModelShard::GetShardStatusLegacy(const std::string& dir,
                                     Shard* remote_shard) const {
  if (!LoadShardLegacy(dir, remote_shard)) {
    return -1;
  }

  if (shard_->shard_mode() != remote_shard->shard_mode()) {
    DXERROR("Inconsistent shard mode: %d vs %d.", shard_->shard_mode(),
            remote_shard->shard_mode());
    return -1;
  }

  if (shard_->shard_size() != remote_shard->shard_size() ||
      shard_->shard_func_name() != remote_shard->shard_func_name()) {
    return 0;
  }
  return 1;
}

int ModelShard::GetShardStatus(const std::string& dir,
                               Shard* remote_shard) const {
  if (!LoadShard(dir, remote_shard)) {
    return -1;
  }

  if (shard_->shard_mode() != remote_shard->shard_mode()) {
    DXERROR("Inconsistent shard mode: %d vs %d.", shard_->shard_mode(),
            remote_shard->shard_mode());
    return -1;
  }

  if (shard_->shard_size() != remote_shard->shard_size() ||
      shard_->shard_func_name() != remote_shard->shard_func_name()) {
    return 0;
  }
  return 1;
}

bool ModelShard::LoadModelLegacy(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  model_.reset(new Model);
  model_->Init(graph_);

  if (status == 0) {
    if (!model_->InitParam(engine_, shard_, shard_id_)) {
      return false;
    }

    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      Model remote_model;
      remote_model.Init(graph_);
      if (!remote_model.LoadLegacy(GetModelFileLegacy(dir, &remote_shard, i))) {
        return false;
      }
      model_->Merge(&remote_model, shard_, shard_id_);
    }
    return true;
  } else {
    return model_->LoadLegacy(GetModelFileLegacy(dir));
  }
}

bool ModelShard::LoadModel(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  model_.reset(new Model);
  model_->Init(graph_);

  if (status == 0) {
    if (!model_->InitParam(engine_, shard_, shard_id_)) {
      return false;
    }

    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      Model remote_model;
      remote_model.Init(graph_);
      if (!remote_model.Load(GetModelFile(dir, &remote_shard, i))) {
        return false;
      }
      model_->Merge(&remote_model, shard_, shard_id_);
    }
    return true;
  } else {
    return model_->Load(GetModelFile(dir));
  }
}

bool ModelShard::LoadOptimizerLegacy(const std::string& dir,
                                     const std::string& optimizer_config) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    std::string name;
    if (!LoadOptimizerName(GetOptimizerFileLegacy(dir, &remote_shard, 0),
                           &name)) {
      return false;
    }
    optimizer_ = NewOptimizer(name);
    if (!optimizer_) {
      return false;
    }
    optimizer_->Init(graph_, model_->mutable_param());
    if (!optimizer_->InitParam()) {
      return false;
    }

    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      std::unique_ptr<Optimizer> remote_optimizer(
          deepx_core::LoadOptimizerLegacy(
              GetOptimizerFileLegacy(dir, &remote_shard, i)));
      if (!remote_optimizer) {
        return false;
      }
      remote_optimizer->Init(graph_, model_->mutable_param());
      if (!optimizer_->MergeLegacy(remote_optimizer.get(), shard_, shard_id_)) {
        return false;
      }
    }
  } else {
    optimizer_ = deepx_core::LoadOptimizerLegacy(GetOptimizerFileLegacy(dir));
    if (!optimizer_) {
      return false;
    }
    optimizer_->Init(graph_, model_->mutable_param());
  }

  if (!optimizer_config.empty()) {
    return InitOptimizerConfig(optimizer_config);
  }
  return true;
}

bool ModelShard::LoadOptimizer(const std::string& dir,
                               const std::string& optimizer_config) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    std::string name;
    if (!LoadOptimizerName(GetOptimizerFile(dir, &remote_shard, 0), &name)) {
      return false;
    }
    optimizer_ = NewOptimizer(name);
    if (!optimizer_) {
      return false;
    }
    optimizer_->Init(graph_, model_->mutable_param());
    if (!optimizer_->InitParam()) {
      return false;
    }

    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      std::unique_ptr<Optimizer> remote_optimizer(
          deepx_core::LoadOptimizer(GetOptimizerFile(dir, &remote_shard, i)));
      if (!remote_optimizer) {
        return false;
      }
      remote_optimizer->Init(graph_, model_->mutable_param());
      if (!optimizer_->Merge(remote_optimizer.get(), shard_, shard_id_)) {
        return false;
      }
    }
  } else {
    optimizer_ = deepx_core::LoadOptimizer(GetOptimizerFile(dir));
    if (!optimizer_) {
      return false;
    }
    optimizer_->Init(graph_, model_->mutable_param());
  }

  if (!optimizer_config.empty()) {
    return InitOptimizerConfig(optimizer_config);
  }
  return true;
}

bool ModelShard::LoadTSStoreLegacy(const std::string& dir, ts_t now,
                                   ts_t expire_threshold) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  ts_store_.reset(new TSStore);
  ts_store_->set_now(now);
  ts_store_->set_expire_threshold(expire_threshold);
  ts_store_->Init(model_->mutable_param());

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      TSStore remote_ts_store;
      remote_ts_store.set_now(now);
      remote_ts_store.set_expire_threshold(expire_threshold);
      remote_ts_store.Init(model_->mutable_param());
      if (!remote_ts_store.LoadLegacy(
              GetTSStoreFileLegacy(dir, &remote_shard, i))) {
        return false;
      }
      ts_store_->Merge(&remote_ts_store, shard_, shard_id_);
    }
    return true;
  } else {
    return ts_store_->LoadLegacy(GetTSStoreFileLegacy(dir));
  }
}

bool ModelShard::LoadTSStore(const std::string& dir, ts_t now,
                             ts_t expire_threshold) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  ts_store_.reset(new TSStore);
  ts_store_->set_now(now);
  ts_store_->set_expire_threshold(expire_threshold);
  ts_store_->Init(model_->mutable_param());

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      TSStore remote_ts_store;
      remote_ts_store.set_now(now);
      remote_ts_store.set_expire_threshold(expire_threshold);
      remote_ts_store.Init(model_->mutable_param());
      if (!remote_ts_store.Load(GetTSStoreFile(dir, &remote_shard, i))) {
        return false;
      }
      ts_store_->Merge(&remote_ts_store, shard_, shard_id_);
    }
    return true;
  } else {
    return ts_store_->Load(GetTSStoreFile(dir));
  }
}

bool ModelShard::LoadFreqStoreLegacy(const std::string& dir,
                                     freq_t freq_filter_threshold) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  freq_store_.reset(new FreqStore);
  freq_store_->set_freq_filter_threshold(freq_filter_threshold);
  freq_store_->Init(model_->mutable_param());

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      FreqStore remote_freq_store;
      remote_freq_store.set_freq_filter_threshold(freq_filter_threshold);
      remote_freq_store.Init(model_->mutable_param());
      if (!remote_freq_store.Load(
              GetFreqStoreFileLegacy(dir, &remote_shard, i))) {
        return false;
      }
      freq_store_->Merge(&remote_freq_store, shard_, shard_id_);
    }
    return true;
  } else {
    return freq_store_->Load(GetFreqStoreFileLegacy(dir));
  }
}

bool ModelShard::LoadFreqStore(const std::string& dir,
                               freq_t freq_filter_threshold) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  freq_store_.reset(new FreqStore);
  freq_store_->set_freq_filter_threshold(freq_filter_threshold);
  freq_store_->Init(model_->mutable_param());

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      FreqStore remote_freq_store;
      remote_freq_store.set_freq_filter_threshold(freq_filter_threshold);
      remote_freq_store.Init(model_->mutable_param());
      if (!remote_freq_store.Load(GetFreqStoreFile(dir, &remote_shard, i))) {
        return false;
      }
      freq_store_->Merge(&remote_freq_store, shard_, shard_id_);
    }
    return true;
  } else {
    return freq_store_->Load(GetFreqStoreFile(dir));
  }
}

bool ModelShard::WarmupModelLegacy(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      Model remote_model;
      remote_model.Init(graph_);
      if (!remote_model.LoadLegacy(GetModelFileLegacy(dir, &remote_shard, i))) {
        return false;
      }
      model_->Merge(&remote_model, shard_, shard_id_);
    }
  } else {
    Model remote_model;
    remote_model.Init(graph_);
    if (!remote_model.LoadLegacy(GetModelFileLegacy(dir))) {
      return false;
    }
    model_->Merge(&remote_model);
  }
  return true;
}

bool ModelShard::WarmupModel(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      Model remote_model;
      remote_model.Init(graph_);
      if (!remote_model.Load(GetModelFile(dir, &remote_shard, i))) {
        return false;
      }
      model_->Merge(&remote_model, shard_, shard_id_);
    }
  } else {
    Model remote_model;
    remote_model.Init(graph_);
    if (!remote_model.Load(GetModelFile(dir))) {
      return false;
    }
    model_->Merge(&remote_model);
  }
  return true;
}

bool ModelShard::WarmupOptimizerLegacy(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      std::unique_ptr<Optimizer> remote_optimizer(
          deepx_core::LoadOptimizerLegacy(
              GetOptimizerFileLegacy(dir, &remote_shard, i)));
      if (!remote_optimizer) {
        return false;
      }
      remote_optimizer->Init(graph_, model_->mutable_param());
      if (!optimizer_->MergeLegacy(remote_optimizer.get(), shard_, shard_id_)) {
        return false;
      }
    }
  } else {
    std::unique_ptr<Optimizer> remote_optimizer(
        deepx_core::LoadOptimizerLegacy(GetOptimizerFileLegacy(dir)));
    if (!remote_optimizer) {
      return false;
    }
    remote_optimizer->Init(graph_, model_->mutable_param());
    if (!optimizer_->MergeLegacy(remote_optimizer.get())) {
      return false;
    }
  }
  return true;
}

bool ModelShard::WarmupOptimizer(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      std::unique_ptr<Optimizer> remote_optimizer(
          deepx_core::LoadOptimizer(GetOptimizerFile(dir, &remote_shard, i)));
      if (!remote_optimizer) {
        return false;
      }
      remote_optimizer->Init(graph_, model_->mutable_param());
      if (!optimizer_->Merge(remote_optimizer.get(), shard_, shard_id_)) {
        return false;
      }
    }
  } else {
    std::unique_ptr<Optimizer> remote_optimizer(
        deepx_core::LoadOptimizer(GetOptimizerFile(dir)));
    if (!remote_optimizer) {
      return false;
    }
    remote_optimizer->Init(graph_, model_->mutable_param());
    if (!optimizer_->Merge(remote_optimizer.get())) {
      return false;
    }
  }
  return true;
}

bool ModelShard::WarmupTSStoreLegacy(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      TSStore remote_ts_store;
      remote_ts_store.set_now(ts_store_->now());
      remote_ts_store.set_expire_threshold(ts_store_->expire_threshold());
      remote_ts_store.Init(model_->mutable_param());
      if (!remote_ts_store.LoadLegacy(
              GetTSStoreFileLegacy(dir, &remote_shard, i))) {
        return false;
      }
      ts_store_->Merge(&remote_ts_store, shard_, shard_id_);
    }
  } else {
    TSStore remote_ts_store;
    remote_ts_store.set_now(ts_store_->now());
    remote_ts_store.set_expire_threshold(ts_store_->expire_threshold());
    remote_ts_store.Init(model_->mutable_param());
    if (!remote_ts_store.LoadLegacy(GetTSStoreFileLegacy(dir))) {
      return false;
    }
    ts_store_->Merge(&remote_ts_store);
  }
  return true;
}

bool ModelShard::WarmupTSStore(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      TSStore remote_ts_store;
      remote_ts_store.set_now(ts_store_->now());
      remote_ts_store.set_expire_threshold(ts_store_->expire_threshold());
      remote_ts_store.Init(model_->mutable_param());
      if (!remote_ts_store.Load(GetTSStoreFile(dir, &remote_shard, i))) {
        return false;
      }
      ts_store_->Merge(&remote_ts_store, shard_, shard_id_);
    }
  } else {
    TSStore remote_ts_store;
    remote_ts_store.set_now(ts_store_->now());
    remote_ts_store.set_expire_threshold(ts_store_->expire_threshold());
    remote_ts_store.Init(model_->mutable_param());
    if (!remote_ts_store.Load(GetTSStoreFile(dir))) {
      return false;
    }
    ts_store_->Merge(&remote_ts_store);
  }
  return true;
}

bool ModelShard::WarmupFreqStoreLegacy(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatusLegacy(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      FreqStore remote_freq_store;
      remote_freq_store.set_freq_filter_threshold(
          freq_store_->freq_filter_threshold());
      remote_freq_store.Init(model_->mutable_param());
      if (!remote_freq_store.Load(
              GetFreqStoreFileLegacy(dir, &remote_shard, i))) {
        return false;
      }
      freq_store_->Merge(&remote_freq_store, shard_, shard_id_);
    }
  } else {
    FreqStore remote_freq_store;
    remote_freq_store.set_freq_filter_threshold(
        freq_store_->freq_filter_threshold());
    remote_freq_store.Init(model_->mutable_param());
    if (!remote_freq_store.Load(GetFreqStoreFileLegacy(dir))) {
      return false;
    }
    freq_store_->Merge(&remote_freq_store);
  }
  return true;
}

bool ModelShard::WarmupFreqStore(const std::string& dir) {
  Shard remote_shard;
  int status = GetShardStatus(dir, &remote_shard);
  if (status == -1) {
    return false;
  }

  if (status == 0) {
    for (int i = 0; i < remote_shard.shard_size(); ++i) {
      FreqStore remote_freq_store;
      remote_freq_store.set_freq_filter_threshold(
          freq_store_->freq_filter_threshold());
      remote_freq_store.Init(model_->mutable_param());
      if (!remote_freq_store.Load(GetFreqStoreFile(dir, &remote_shard, i))) {
        return false;
      }
      freq_store_->Merge(&remote_freq_store, shard_, shard_id_);
    }
  } else {
    FreqStore remote_freq_store;
    remote_freq_store.set_freq_filter_threshold(
        freq_store_->freq_filter_threshold());
    remote_freq_store.Init(model_->mutable_param());
    if (!remote_freq_store.Load(GetFreqStoreFile(dir))) {
      return false;
    }
    freq_store_->Merge(&remote_freq_store);
  }
  return true;
}

void ModelShard::Pull(PullRequest* pull_request, TensorMap* param) {
  if (freq_store_ && pull_request->is_train) {
    freq_store_->Filter(pull_request);
  }
  model_->Pull(engine_, *pull_request, param);
}

void ModelShard::Push(TensorMap* grad, TensorMap* overwritten_param) {
  if (!grad->empty()) {
    if (ol_store_) {
      ol_store_->Update(grad);
    }
    if (freq_store_) {
      freq_store_->Filter(grad);
    }
    if (ts_store_) {
      ts_store_->Update(grad);
    }
    optimizer_->Update(grad);
  }

  if (overwritten_param && !overwritten_param->empty()) {
    if (ol_store_) {
      ol_store_->Update(overwritten_param);
    }
    model_->Update(overwritten_param);
  }
}

void ModelShard::ExpireTSStore() {
  auto expired = ts_store_->Expire();
  auto filter = [&expired](const std::string& name, srm_t* W) {
    size_t prev_size = W->size();
    W->remove_if([&expired](const srm_t::value_type& entry) {
      return expired.count(entry.first) > 0;
    });
    DXINFO("SRM %s has %zu entries expired, %zu entries remained.",
           name.c_str(), prev_size - W->size(), W->size());
  };
  model_->ForEachSRM(filter);
  optimizer_->ForEachSRM(filter);
  if (freq_store_) {
    freq_store_->RemoveIf([&expired](const id_freq_map_t::value_type& entry) {
      return expired.count(entry.first) > 0;
    });
  }
}

bool ModelShard::InitThreadPool() {
  thread_pool_.reset(new ThreadPool);
  return true;
}

void ModelShard::StartThreadPool() { thread_pool_->start(1); }

void ModelShard::StopThreadPool() { thread_pool_->stop(); }

void ModelShard::AsyncPull(PullRequest* pull_request, TensorMap* param,
                           const std::function<void()>& completion_handler) {
  thread_pool_->post([this, pull_request, param, completion_handler]() {
    Pull(pull_request, param);
    completion_handler();
  });
}

void ModelShard::AsyncPush(TensorMap* grad, TensorMap* overwritten_param,
                           const std::function<void()>& completion_handler) {
  thread_pool_->post([this, grad, overwritten_param, completion_handler]() {
    Push(grad, overwritten_param);
    completion_handler();
  });
}

void ModelShard::SplitPullRequest(const PullRequest& full_pull_request,
                                  std::vector<PullRequest>* pull_requests,
                                  std::vector<id_set_t*>* aux) const {
  DXASSERT(shard_->shard_mode() == 1);
  int shard_size = shard_->shard_size();
  DXASSERT((int)pull_requests->size() == shard_size);
  DXASSERT((int)aux->size() == shard_size);

  for (PullRequest& pull_request : *pull_requests) {
    pull_request.clear();
    pull_request.is_train = full_pull_request.is_train;
  }

  for (const std::string& name : full_pull_request.tsr_set) {
    int shard_id = shard_->GetTSRShardId(name);
    (*pull_requests)[shard_id].tsr_set.emplace(name);
  }

  for (const auto& entry : full_pull_request.srm_map) {
    const std::string& name = entry.first;
    const id_set_t& id_set = entry.second;
    size_t srm_id_size = id_set.size() / shard_size;
    for (int i = 0; i < shard_size; ++i) {
      (*aux)[i] = &(*pull_requests)[i].srm_map[name];
      (*aux)[i]->reserve(srm_id_size);
    }
    for (int_t id : id_set) {
      int shard_id = shard_->GetSRMShardId(id);
      (*aux)[shard_id]->emplace(id);
    }
  }

  for (const auto& entry : full_pull_request.id_freq_map) {
    int_t id = entry.first;
    freq_t freq = entry.second;
    int shard_id = shard_->GetSRMShardId(id);
    (*pull_requests)[shard_id].id_freq_map.emplace(id, freq);
  }
}

void ModelShard::SplitGrad(const TensorMap& param, TensorMap* full_grad,
                           std::vector<std::unique_ptr<TensorMap>>* grads,
                           std::vector<srm_t*>* aux) const {
  DXASSERT(shard_->shard_mode() == 1);
  int shard_size = shard_->shard_size();
  DXASSERT((int)grads->size() == shard_size);
  DXASSERT((int)aux->size() == shard_size);

  for (auto& grad : *grads) {
    grad->ClearValue();
  }

  for (auto& entry : *full_grad) {
    const std::string& name = entry.first;
    auto it = param.find(name);
    if (it == param.end()) {
      continue;
    }

    const Any& Wany = it->second;
    Any& Gany = entry.second;
    if (Wany.is<tsr_t>()) {
      int shard_id = shard_->GetTSRShardId(name);
      if (Gany.is<tsr_t>()) {
        auto& G = Gany.unsafe_to_ref<tsr_t>();
        // view, zero-copy
        (*grads)[shard_id]->get_or_insert<tsr_t>(name) = G.get_view();
      } else if (Gany.is<srm_t>()) {
        auto& G = Gany.unsafe_to_ref<srm_t>();
        int col = G.col();
        (*grads)[shard_id]->get_or_insert<srm_t>(name) = std::move(G);
        G.clear();
        G.set_col(col);
      }
    } else if (Wany.is<srm_t>()) {
      if (Gany.is<srm_t>()) {
        auto& G = Gany.unsafe_to_ref<srm_t>();
        size_t srm_id_size = G.size() / shard_size;
        for (int i = 0; i < shard_size; ++i) {
          (*aux)[i] = &(*grads)[i]->get_or_insert<srm_t>(name);
          (*aux)[i]->set_col(G.col());
          (*aux)[i]->reserve(srm_id_size);
        }
        for (const auto& _entry : G) {
          int_t id = _entry.first;
          const float_t* embedding = _entry.second;
          int shard_id = shard_->GetSRMShardId(id);
          // view, zero-copy
          (*aux)[shard_id]->assign_view(id, embedding);
        }
      }
    }
  }

  for (auto& grad : *grads) {
    grad->RemoveEmptyValue();
  }
}

void ModelShard::SplitParam(const TensorMap& full_param,
                            std::vector<std::unique_ptr<TensorMap>>* params,
                            std::vector<srm_t*>* aux) const {
  DXASSERT(shard_->shard_mode() == 1);
  int shard_size = shard_->shard_size();
  DXASSERT((int)params->size() == shard_size);
  DXASSERT((int)aux->size() == shard_size);

  for (auto& param : *params) {
    param->ClearValue();
  }

  for (const auto& entry : full_param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      int shard_id = shard_->GetTSRShardId(name);
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      // view, zero-copy
      (*params)[shard_id]->get_or_insert<tsr_t>(name) = W.get_view();
    } else if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      size_t srm_id_size = W.size() / shard_size;
      for (int i = 0; i < shard_size; ++i) {
        (*aux)[i] = &(*params)[i]->get_or_insert<srm_t>(name);
        (*aux)[i]->set_col(W.col());
        (*aux)[i]->reserve(srm_id_size);
      }
      for (const auto& _entry : W) {
        int_t id = _entry.first;
        const float_t* embedding = _entry.second;
        int shard_id = shard_->GetSRMShardId(id);
        // view, zero-copy
        (*aux)[shard_id]->assign_view(id, embedding);
      }
    }
  }

  for (auto& param : *params) {
    param->RemoveEmptyValue();
  }
}

}  // namespace deepx_core
