// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/model.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/graph/optimizer.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* TrainerContext */
/************************************************************************/
class TrainerContext : public DataType {
 protected:
  std::string instance_reader_;
  std::string instance_reader_config_;
  int batch_ = 0;
  int verbose_ = 0;
  freq_t freq_filter_threshold_ = 0;
  std::string target_name_;

  ModelShard* local_model_shard_ = nullptr;
  std::unique_ptr<OpContext> op_context_;
  int op_context_batch_ = -1;
  double file_loss_ = 0;
  double file_loss_weight_ = 0;

 public:
  void set_instance_reader(const std::string& instance_reader) {
    instance_reader_ = instance_reader;
  }
  void set_instance_reader_config(const std::string& instance_reader_config) {
    instance_reader_config_ = instance_reader_config;
  }
  void set_batch(int batch) noexcept { batch_ = batch; }
  void set_verbose(int verbose) noexcept { verbose_ = verbose; }
  void set_freq_filter_threshold(freq_t freq_filter_threshold) noexcept {
    freq_filter_threshold_ = freq_filter_threshold;
  }
  void set_target_name(const std::string& target_name) {
    target_name_ = target_name;
  }
  double file_loss() const noexcept { return file_loss_; }
  double file_loss_weight() const noexcept { return file_loss_weight_; }

 protected:
  int enable_profile_ = 0;
  std::unordered_map<std::string, double> profile_map_;

 protected:
  void DumpProfile() const;

 protected:
  void _Init(ModelShard* local_model_shard);

 public:
  TrainerContext();
  virtual ~TrainerContext();
  virtual void TrainBatch() = 0;
  virtual void TrainFile(int thread_id, const std::string& file);
  virtual void PredictBatch() = 0;
  virtual void DumpPredictBatch(OutputStream& os) const;  // NOLINT
  virtual void PredictFile(int thread_id, const std::string& file,
                           const std::string& out_file);
};

/************************************************************************/
/* TrainerContextNonShard */
/************************************************************************/
class TrainerContextNonShard : public TrainerContext {
 protected:
  Optimizer* optimizer_ = nullptr;

 public:
  void Init(ModelShard* model_shard);
  void TrainBatch() override;
  void PredictBatch() override;
};

/************************************************************************/
/* TrainerContextShard */
/************************************************************************/
class TrainerContextShard : public TrainerContext {
 protected:
  int shard_size_ = 0;
  std::vector<ModelShard*> model_shards_;
  std::vector<Model*> models_;
  std::vector<Optimizer*> optimizers_;

  PullRequest pull_request_;
  std::vector<PullRequest> pull_requests_;
  int pull_request_active_ = 0;
  std::vector<int> pull_request_masks_;
  std::vector<std::unique_ptr<TensorMap>> params_;
  std::vector<std::unique_ptr<TensorMap>> grads_;
  std::vector<std::unique_ptr<TensorMap>> overwritten_params_;
  std::vector<id_set_t*> aux1_;
  std::vector<srm_t*> aux2_;

  ThreadPool::wait_token_t wait_token_;

 public:
  void Init(std::vector<ModelShard>* model_shards,
            ModelShard* local_model_shard);
  void TrainBatch() override;
  void PredictBatch() override;

 protected:
  void CompletionHandler();
  void WaitForCompletion();
  void Pull(int is_train);
  void Push();
};

}  // namespace deepx_core
