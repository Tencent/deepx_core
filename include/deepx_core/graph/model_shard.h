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
#include <deepx_core/graph/shard.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/graph/ts_store.h>
#include <deepx_core/tensor/data_type.h>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* ModelShard */
/************************************************************************/
class ModelShard : public DataType {
 private:
  std::default_random_engine engine_;
  const Shard* shard_ = nullptr;
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
  const Shard& shard() const noexcept { return *shard_; }
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
  // backward compatibility
  static std::string GetSuffixLegacy(const Shard* shard, int shard_id);
  static std::string GetSuffix(const Shard* shard, int shard_id);
  // backward compatibility
  static std::string GetModelFileLegacy(const std::string& dir,
                                        const Shard* shard, int shard_id);
  static std::string GetModelFile(const std::string& dir, const Shard* shard,
                                  int shard_id);
  static std::string GetTextModelFile(const std::string& dir,
                                      const Shard* shard, int shard_id);
  static std::string GetFeatureKVModelFile(const std::string& dir,
                                           const Shard* shard, int shard_id);
  // backward compatibility
  static std::string GetOptimizerFileLegacy(const std::string& dir,
                                            const Shard* shard, int shard_id);
  static std::string GetOptimizerFile(const std::string& dir,
                                      const Shard* shard, int shard_id);
  // backward compatibility
  static std::string GetTSStoreFileLegacy(const std::string& dir,
                                          const Shard* shard, int shard_id);
  static std::string GetTSStoreFile(const std::string& dir, const Shard* shard,
                                    int shard_id);
  // backward compatibility
  static std::string GetFreqStoreFileLegacy(const std::string& dir,
                                            const Shard* shard, int shard_id);
  static std::string GetFreqStoreFile(const std::string& dir,
                                      const Shard* shard, int shard_id);
  // backward compatibility
  static std::string GetSuccessFileLegacy(const std::string& dir,
                                          const Shard* shard, int shard_id);
  static std::string GetSuccessFile(const std::string& dir, const Shard* shard,
                                    int shard_id);

 private:
  // backward compatibility
  std::string GetModelFileLegacy(const std::string& dir) const;
  std::string GetModelFile(const std::string& dir) const;
  std::string GetTextModelFile(const std::string& dir) const;
  std::string GetFeatureKVModelFile(const std::string& dir) const;
  // backward compatibility
  std::string GetOptimizerFileLegacy(const std::string& dir) const;
  std::string GetOptimizerFile(const std::string& dir) const;
  // backward compatibility
  std::string GetTSStoreFileLegacy(const std::string& dir) const;
  std::string GetTSStoreFile(const std::string& dir) const;
  // backward compatibility
  std::string GetFreqStoreFileLegacy(const std::string& dir) const;
  std::string GetFreqStoreFile(const std::string& dir) const;
  // backward compatibility
  std::string GetSuccessFileLegacy(const std::string& dir) const;
  std::string GetSuccessFile(const std::string& dir) const;

 public:
  void InitShard(const Shard* shard, int shard_id) noexcept;
  void InitGraph(const Graph* graph) noexcept;
  bool InitModelPlaceholder();
  bool InitModel();
  bool InitOptimizer(const std::string& optimizer,
                     const std::string& optimizer_config);
  bool InitOptimizerConfig(const std::string& optimizer_config);
  bool InitTSStore(ts_t now, ts_t expire_threshold);
  bool InitFreqStore(freq_t freq_filter_threshold);
  bool InitOLStore(freq_t update_threshold, float_t distance_threshold);
  bool InitLock();

  // backward compatibility
  bool SaveModelLegacy(const std::string& dir) const;
  bool SaveModel(const std::string& dir) const;
  bool SaveTextModel(const std::string& dir) const;
  // backward compatibility
  bool SaveFeatureKVModelLegacy(const std::string& dir,
                                int feature_kv_protocol_version) const;
  bool SaveFeatureKVModel(const std::string& dir,
                          int feature_kv_protocol_version) const;
  // backward compatibility
  bool SaveOLFeatureKVModelLegacy(const std::string& dir,
                                  int feature_kv_protocol_version) const;
  bool SaveOLFeatureKVModel(const std::string& dir,
                            int feature_kv_protocol_version) const;
  // backward compatibility
  bool SaveOptimizerLegacy(const std::string& dir) const;
  bool SaveOptimizer(const std::string& dir) const;
  // backward compatibility
  bool SaveTSStoreLegacy(const std::string& dir) const;
  bool SaveTSStore(const std::string& dir) const;
  // backward compatibility
  bool SaveFreqStoreLegacy(const std::string& dir) const;
  bool SaveFreqStore(const std::string& dir) const;
  // backward compatibility
  bool SaveSuccessLegacy(const std::string& dir) const;
  bool SaveSuccess(const std::string& dir) const;

 private:
  // Load 'remote_shard' from 'dir'.
  //
  // Return 1, 'shard_' and 'remote_shard' are the same.
  // Return 0, 'shard_' and 'remote_shard' differ in shard size or shard func.
  // Return -1, error.

  // backward compatibility
  int GetShardStatusLegacy(const std::string& dir, Shard* remote_shard) const;
  int GetShardStatus(const std::string& dir, Shard* remote_shard) const;

 public:
  // backward compatibility
  bool LoadModelLegacy(const std::string& dir);
  bool LoadModel(const std::string& dir);
  // backward compatibility
  bool LoadOptimizerLegacy(const std::string& dir,
                           const std::string& optimizer_config);
  bool LoadOptimizer(const std::string& dir,
                     const std::string& optimizer_config);
  // backward compatibility
  bool LoadTSStoreLegacy(const std::string& dir, ts_t now,
                         ts_t expire_threshold);
  bool LoadTSStore(const std::string& dir, ts_t now, ts_t expire_threshold);
  // backward compatibility
  bool LoadFreqStoreLegacy(const std::string& dir,
                           freq_t freq_filter_threshold);
  bool LoadFreqStore(const std::string& dir, freq_t freq_filter_threshold);

  // backward compatibility
  bool WarmupModelLegacy(const std::string& dir);
  bool WarmupModel(const std::string& dir);
  // backward compatibility
  bool WarmupOptimizerLegacy(const std::string& dir);
  bool WarmupOptimizer(const std::string& dir);
  // backward compatibility
  bool WarmupTSStoreLegacy(const std::string& dir);
  bool WarmupTSStore(const std::string& dir);
  // backward compatibility
  bool WarmupFreqStoreLegacy(const std::string& dir);
  bool WarmupFreqStore(const std::string& dir);

 public:
  // thread safe after 'InitLock'
  void Pull(PullRequest* pull_request, TensorMap* param);
  // thread safe after 'InitLock'
  // 'overwritten_param' can be nullptr.
  void Push(TensorMap* grad, TensorMap* overwritten_param);
  void ExpireTSStore();

 public:
  bool InitThreadPool();
  void StartThreadPool();
  void StopThreadPool();
  void AsyncPull(PullRequest* pull_request, TensorMap* param,
                 const std::function<void()>& completion_handler);
  // 'overwritten_param' can be nullptr.
  void AsyncPush(TensorMap* grad, TensorMap* overwritten_param,
                 const std::function<void()>& completion_handler);

 public:
  void SplitPullRequest(const PullRequest& full_pull_request,
                        std::vector<PullRequest>* pull_requests,
                        std::vector<id_set_t*>* aux) const;
  void SplitGrad(const TensorMap& param, TensorMap* full_grad,
                 std::vector<std::unique_ptr<TensorMap>>* grads,
                 std::vector<srm_t*>* aux) const;
  void SplitParam(const TensorMap& full_param,
                  std::vector<std::unique_ptr<TensorMap>>* params,
                  std::vector<srm_t*>* aux) const;
};

}  // namespace deepx_core
