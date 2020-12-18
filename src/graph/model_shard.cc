// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/feature_kv_util.h>
#include <deepx_core/graph/model_shard.h>

namespace deepx_core {

/************************************************************************/
/* ShardInfo functions */
/************************************************************************/
std::string GetShardInfoFile(const std::string& dir) {
  return dir + "/shard_info.bin";
}

bool HasShardInfo(const std::string& dir) {
  std::string file = GetShardInfoFile(dir);
  AutoFileSystem fs;
  return fs.Open(file) && fs.IsFile(file);
}

bool SaveShardInfo(const std::string& dir, const ShardInfo& shard_info) {
  std::string file = GetShardInfoFile(dir);
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving shard info to %s...", file.c_str());
  os << shard_info.shard_size;
  if (!os) {
    DXERROR("Failed to write shard info.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool LoadShardInfo(const std::string& dir, ShardInfo* shard_info) {
  std::string file = GetShardInfoFile(dir);
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading shard info from %s...", file.c_str());
  is >> shard_info->shard_size;
  if (!is) {
    DXERROR("Failed to read shard info.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

ShardInfo GetShardInfo(const std::string& dir, int shard_size) {
  ShardInfo shard_info;
  if (!dir.empty() && HasShardInfo(dir) && LoadShardInfo(dir, &shard_info)) {
    return shard_info;
  }
  shard_info.shard_size = shard_size;
  return shard_info;
}

/************************************************************************/
/* ModelShard */
/************************************************************************/
std::string ModelShard::GetSuffix() const {
  std::string suffix;
  if (shard_id_ >= 0) {
    suffix = "." + std::to_string(shard_id_);
  }
  return suffix;
}

std::string ModelShard::GetModelFile(const std::string& dir) const {
  return dir + "/model.bin" + GetSuffix();
}

std::string ModelShard::GetTextModelFile(const std::string& dir) const {
  return dir + "/model.txt" + GetSuffix();
}

std::string ModelShard::GetFeatureKVModelFile(const std::string& dir) const {
  return dir + "/model.feature_kv" + GetSuffix();
}

std::string ModelShard::GetOptimizerFile(const std::string& dir) const {
  return dir + "/optimizer.bin" + GetSuffix();
}

std::string ModelShard::GetTSStoreFile(const std::string& dir) const {
  return dir + "/ts_store.bin" + GetSuffix();
}

std::string ModelShard::GetFreqStoreFile(const std::string& dir) const {
  return dir + "/freq_store.bin" + GetSuffix();
}

std::string ModelShard::GetSuccessFile(const std::string& dir) const {
  return dir + "/SUCCESS_" + GetSuffix();
}

void ModelShard::Init(int shard_id, const Graph* graph) noexcept {
  shard_id_ = shard_id;
  graph_ = graph;
}

bool ModelShard::InitModelPlaceholder() {
  model_.reset(new Model);
  model_->Init(graph_);
  return model_->InitParamPlaceholder();
}

bool ModelShard::InitModel() {
  model_.reset(new Model);
  model_->Init(graph_);
  return model_->InitParam(engine_);
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
  ts_store_->Init(graph_);
  return ts_store_->InitParam(model_->param());
}

bool ModelShard::InitFreqStore(freq_t freq_filter_threshold) {
  freq_store_.reset(new FreqStore);
  freq_store_->set_freq_filter_threshold(freq_filter_threshold);
  return freq_store_->InitParam(model_->param());
}

bool ModelShard::InitOLStore(freq_t update_threshold,
                             float_t distance_threshold) {
  ol_store_.reset(new OLStore);
  ol_store_->set_update_threshold(update_threshold);
  ol_store_->set_distance_threshold(distance_threshold);
  ol_store_->Init(graph_, model_->mutable_param());
  return ol_store_->InitParam();
}

void ModelShard::InitLock() {
  model_->InitLock();
  optimizer_->InitLock(model_->mutable_param_lock());
  if (ts_store_) {
    ts_store_->InitLock();
  }
  if (freq_store_) {
    freq_store_->InitLock();
  }
}

bool ModelShard::SaveModel(const std::string& dir) const {
  return model_->Save(GetModelFile(dir));
}

bool ModelShard::LoadModel(const std::string& dir) {
  model_.reset(new Model);
  model_->Init(graph_);
  return model_->Load(GetModelFile(dir));
}

bool ModelShard::SaveTextModel(const std::string& dir) const {
  return model_->SaveText(GetTextModelFile(dir));
}

bool ModelShard::SaveFeatureKVModel(const std::string& dir,
                                    int feature_kv_protocol_version) const {
  return FeatureKVUtil::SaveModel(GetFeatureKVModelFile(dir), *graph_,
                                  model_->param(), feature_kv_protocol_version);
}

bool ModelShard::SaveOLFeatureKVModel(const std::string& dir,
                                      int feature_kv_protocol_version) const {
  return ol_store_->SaveFeatureKVModel(GetFeatureKVModelFile(dir),
                                       feature_kv_protocol_version);
}

bool ModelShard::SaveOptimizer(const std::string& dir) const {
  return deepx_core::SaveOptimizer(GetOptimizerFile(dir), *optimizer_);
}

bool ModelShard::LoadOptimizer(const std::string& dir,
                               const std::string& optimizer_config) {
  optimizer_ = deepx_core::LoadOptimizer(GetOptimizerFile(dir));
  if (!optimizer_) {
    return false;
  }
  optimizer_->Init(graph_, model_->mutable_param());
  if (!optimizer_config.empty()) {
    return InitOptimizerConfig(optimizer_config);
  }
  return true;
}

bool ModelShard::SaveTSStore(const std::string& dir) const {
  return ts_store_->Save(GetTSStoreFile(dir));
}

bool ModelShard::LoadTSStore(const std::string& dir, ts_t now,
                             ts_t expire_threshold) {
  ts_store_.reset(new TSStore);
  ts_store_->set_now(now);
  ts_store_->set_expire_threshold(expire_threshold);
  ts_store_->Init(graph_);
  return ts_store_->Load(GetTSStoreFile(dir));
}

bool ModelShard::SaveFreqStore(const std::string& dir) const {
  return freq_store_->Save(GetFreqStoreFile(dir));
}

bool ModelShard::LoadFreqStore(const std::string& dir,
                               freq_t freq_filter_threshold) {
  freq_store_.reset(new FreqStore);
  freq_store_->set_freq_filter_threshold(freq_filter_threshold);
  return freq_store_->Load(GetFreqStoreFile(dir));
}

bool ModelShard::SaveSuccess(const std::string& dir) const {
  std::string file = GetSuccessFile(dir);
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving SUCCESS file to %s...", file.c_str());
  DXINFO("Done.");
  return true;
}

bool ModelShard::WarmupModel(const std::string& dir) {
  Model other;
  if (!other.Load(GetModelFile(dir))) {
    return false;
  }
  model_->Warmup(&other);
  return true;
}

bool ModelShard::WarmupOptimizer(const std::string& dir) {
  std::unique_ptr<Optimizer> other =
      deepx_core::LoadOptimizer(GetOptimizerFile(dir));
  if (!other) {
    return false;
  }
  optimizer_->Warmup(other.get());
  return true;
}

bool ModelShard::WarmupTSStore(const std::string& dir) {
  TSStore other;
  if (!other.Load(GetTSStoreFile(dir))) {
    return false;
  }
  ts_store_->Warmup(&other);
  return true;
}

bool ModelShard::WarmupFreqStore(const std::string& dir) {
  FreqStore other;
  if (!other.Load(GetFreqStoreFile(dir))) {
    return false;
  }
  freq_store_->Warmup(&other);
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

  // 'overwritten_param' can be nullptr.
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
    freq_store_->RemoveIf(
        [&expired](const FreqStore::id_freq_map_t::value_type& entry) {
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
  thread_pool_->emplace([this, pull_request, param, completion_handler]() {
    Pull(pull_request, param);
    completion_handler();
  });
}

void ModelShard::AsyncPush(TensorMap* grad, TensorMap* overwritten_param,
                           const std::function<void()>& completion_handler) {
  thread_pool_->emplace([this, grad, overwritten_param, completion_handler]() {
    Push(grad, overwritten_param);
    completion_handler();
  });
}

}  // namespace deepx_core
