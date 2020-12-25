// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#pragma once
#include <deepx_core/common/thread_pool.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/freq_store.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model.h>
#include <deepx_core/graph/ol_store.h>
#include <deepx_core/graph/optimizer.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/graph/ts_store.h>
#include <deepx_core/tensor/data_type.h>
#include <functional>
#include <memory>
#include <random>
#include <string>

namespace deepx_core {

/************************************************************************/
/* ShardInfo */
/************************************************************************/
struct ShardInfo {
  int shard_size = 0;
};

/************************************************************************/
/* ShardInfo functions */
/************************************************************************/
std::string GetShardInfoFile(const std::string& dir);
bool HasShardInfo(const std::string& dir);
bool SaveShardInfo(const std::string& dir, const ShardInfo& shard_info);
bool LoadShardInfo(const std::string& dir, ShardInfo* shard_info);
ShardInfo GetShardInfo(const std::string& dir, int shard_size);

/************************************************************************/
/* ModelShard */
/************************************************************************/
class ModelShard : public DataType {
 private:
  std::default_random_engine engine_;
  int shard_id_ = 0;
  const Graph* graph_ = nullptr;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Optimizer> optimizer_;
  std::unique_ptr<TSStore> ts_store_;
  std::unique_ptr<FreqStore> freq_store_;
  std::unique_ptr<OLStore> ol_store_;
  std::unique_ptr<ThreadPool> thread_pool_;

 public:
  template <typename Int>
  void seed(Int s) {
    engine_.seed((std::default_random_engine::result_type)s);
  }
  std::default_random_engine& engine() noexcept { return engine_; }
  int shard_id() const noexcept { return shard_id_; }
  const Graph& graph() const noexcept { return *graph_; }
  Model* mutable_model() noexcept { return model_.get(); }
  const Model& model() const noexcept { return *model_; }
  TensorMap* mutable_param() noexcept { return model_->mutable_param(); }
  const TensorMap& param() const noexcept { return model_->param(); }
  Optimizer* mutable_optimizer() noexcept { return optimizer_.get(); }
  const Optimizer& optimizer() const noexcept { return *optimizer_; }
  TSStore* mutable_ts_store() noexcept { return ts_store_.get(); }
  const TSStore& ts_store() const noexcept { return *ts_store_; }
  FreqStore* mutable_freq_store() noexcept { return freq_store_.get(); }
  const FreqStore& freq_store() const noexcept { return *freq_store_; }
  OLStore* mutable_ol_store() noexcept { return ol_store_.get(); }
  const OLStore& ol_store() const noexcept { return *ol_store_; }

 private:
  std::string GetSuffix() const;
  std::string GetModelFile(const std::string& dir) const;
  std::string GetTextModelFile(const std::string& dir) const;
  std::string GetFeatureKVModelFile(const std::string& dir) const;
  std::string GetOptimizerFile(const std::string& dir) const;
  std::string GetTSStoreFile(const std::string& dir) const;
  std::string GetFreqStoreFile(const std::string& dir) const;
  std::string GetSuccessFile(const std::string& dir) const;

 public:
  void Init(int shard_id, const Graph* graph) noexcept;
  bool InitModelPlaceholder();
  bool InitModel();
  bool InitOptimizer(const std::string& optimizer,
                     const std::string& optimizer_config);
  bool InitOptimizerConfig(const std::string& optimizer_config);
  bool InitTSStore(ts_t now, ts_t expire_threshold);
  bool InitFreqStore(freq_t freq_filter_threshold);
  bool InitOLStore(freq_t update_threshold, float_t distance_threshold);
  void InitLock();

  bool SaveModel(const std::string& dir) const;
  bool SaveTextModel(const std::string& dir) const;
  bool SaveFeatureKVModel(const std::string& dir,
                          int feature_kv_protocol_version) const;
  bool SaveOLFeatureKVModel(const std::string& dir,
                            int feature_kv_protocol_version) const;
  bool SaveOptimizer(const std::string& dir) const;
  bool SaveTSStore(const std::string& dir) const;
  bool SaveFreqStore(const std::string& dir) const;
  bool SaveSuccess(const std::string& dir) const;

  bool LoadModel(const std::string& dir);
  bool LoadOptimizer(const std::string& dir,
                     const std::string& optimizer_config);
  bool LoadTSStore(const std::string& dir, ts_t now, ts_t expire_threshold);
  bool LoadFreqStore(const std::string& dir, freq_t freq_filter_threshold);

  bool WarmupModel(const std::string& dir);
  bool WarmupOptimizer(const std::string& dir);
  bool WarmupTSStore(const std::string& dir);
  bool WarmupFreqStore(const std::string& dir);

 public:
  // thread safe after 'InitLock'
  void Pull(PullRequest* pull_request, TensorMap* param);
  // thread safe after 'InitLock'
  void Push(TensorMap* grad, TensorMap* overwritten_param);
  void ExpireTSStore();

 public:
  bool InitThreadPool();
  void StartThreadPool();
  void StopThreadPool();
  void AsyncPull(PullRequest* pull_request, TensorMap* param,
                 const std::function<void()>& completion_handler);
  void AsyncPush(TensorMap* grad, TensorMap* overwritten_param,
                 const std::function<void()>& completion_handler);
};

}  // namespace deepx_core
