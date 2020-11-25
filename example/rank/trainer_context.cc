// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include "trainer_context.h"
#include <deepx_core/common/any_map.h>
#include <deepx_core/common/misc.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/instance_reader.h>
#include <chrono>
#include <mutex>
#include <sstream>

namespace deepx_core {

/************************************************************************/
/* TrainerContext */
/************************************************************************/
void TrainerContext::DoInit(Model* local_model) {
  local_model_ = local_model;
  op_context_.reset(new OpContext);
  op_context_->Init(&local_model_->graph(), local_model_->mutable_param());
  op_context_batch_ = -1;
  file_loss_ = 0;
  file_loss_weight_ = 0;
}

void TrainerContext::TrainFile(int thread_id, const std::string& file) {
  DXCHECK_THROW(op_context_->InitOp({target_name_}, 0));
  op_context_->mutable_inst()->clear();
  op_context_batch_ = -1;
  file_loss_ = 0;
  file_loss_weight_ = 0;

  std::unique_ptr<InstanceReader> instance_reader(
      NewInstanceReader(instance_reader_));
  DXCHECK_THROW(instance_reader);
  StringMap config;
  DXCHECK_THROW(ParseConfig(instance_reader_config_, &config));
  config["batch"] = std::to_string(batch_);
  DXCHECK_THROW(instance_reader->InitConfig(config));
  DXCHECK_THROW(instance_reader->Open(file));

  std::size_t processed_batch = 0;
  std::size_t verbose_batch = GetVerboseBatch(verbose_);
  auto begin = std::chrono::steady_clock::now();

  auto dump_speed = [this, thread_id, &processed_batch, &begin]() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    DXINFO("[%d] %f instances/s, file_loss=%f", thread_id,
           processed_batch * batch_ * 1000.0 / ms.count(),
           file_loss_ / file_loss_weight_);
  };

  Instance* inst = op_context_->mutable_inst();
  while (instance_reader->GetBatch(inst)) {
    TrainBatch();
    if (verbose_ && ++processed_batch % verbose_batch == 0) {
      dump_speed();
    }
  }

  if (inst->batch() > 0) {
    TrainBatch();
  }

  if (verbose_) {
    dump_speed();
  }
}

void TrainerContext::DumpPredictBatch(OutputStream& os) const {
  const tsr_t* Y = nullptr;
  const tsr_t* W = nullptr;
  const tsrs_t* uuid = nullptr;
  const auto* Z = op_context_->ptr().get<tsr_t*>(target_name_);

  DXCHECK_THROW(Z->is_rank(2));
  int batch = Z->dim(0);

  const Instance& inst = op_context_->inst();
  auto it = inst.find(Y_NAME);
  if (it != inst.end()) {
    Y = &it->second.to_ref<tsr_t>();
    DXCHECK_THROW(Y->is_rank(2));
    DXCHECK_THROW(Y->dim(0) == batch);
  }

  it = inst.find(W_NAME);
  if (it != inst.end()) {
    W = &it->second.to_ref<tsr_t>();
    DXCHECK_THROW(W->is_rank(2));
    DXCHECK_THROW(W->dim(0) == batch);
  }

  if (Y && W) {
    DXCHECK_THROW(Y->dim(1) == W->dim(1));
  }

  it = inst.find(UUID_NAME);
  if (it != inst.end()) {
    uuid = &it->second.to_ref<tsrs_t>();
    DXCHECK_THROW(uuid->is_rank(1));
    DXCHECK_THROW(uuid->dim(0) == batch);
  }

  std::ostringstream oss;
  for (int i = 0; i < batch; ++i) {
    oss.clear();
    oss.str("");
    if (uuid) {
      oss << " " << uuid->data(i);
    }
    if (Y) {
      for (int j = 0; j < Y->dim(1); ++j) {
        oss << " " << Y->data(i * Y->dim(1) + j);
      }
    }
    if (W) {
      for (int j = 0; j < W->dim(1); ++j) {
        oss << " " << W->data(i * W->dim(1) + j);
      }
    }
    for (int j = 0; j < Z->dim(1); ++j) {
      oss << " " << Z->data(i * Z->dim(1) + j);
    }
    oss << std::endl;
    std::string s = oss.str();
    os.Write(s.data() + 1, s.size() - 1);  // trim the leading space
  }
}

void TrainerContext::PredictFile(int thread_id, const std::string& file,
                                 const std::string& out_file) {
  DXCHECK_THROW(op_context_->InitOp({target_name_}, -1));
  op_context_->mutable_inst()->clear();
  op_context_batch_ = -1;

  std::unique_ptr<InstanceReader> instance_reader(
      NewInstanceReader(instance_reader_));
  DXCHECK_THROW(instance_reader);
  StringMap config;
  DXCHECK_THROW(ParseConfig(instance_reader_config_, &config));
  config["batch"] = std::to_string(batch_);
  DXCHECK_THROW(instance_reader->InitConfig(config));
  DXCHECK_THROW(instance_reader->Open(file));

  AutoOutputFileStream os;
  DXCHECK_THROW(os.Open(out_file));

  std::size_t processed_batch = 0;
  std::size_t verbose_batch = GetVerboseBatch(verbose_);
  auto begin = std::chrono::steady_clock::now();

  auto dump_speed = [this, thread_id, &processed_batch, &begin]() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    DXINFO("[%d] %f instances/s", thread_id,
           processed_batch * batch_ * 1000.0 / ms.count());
  };

  Instance* inst = op_context_->mutable_inst();
  while (instance_reader->GetBatch(inst)) {
    PredictBatch();
    DumpPredictBatch(os);
    if (verbose_ && ++processed_batch % verbose_batch == 0) {
      dump_speed();
    }
  }

  if (inst->batch() > 0) {
    PredictBatch();
    DumpPredictBatch(os);
  }

  if (verbose_) {
    dump_speed();
  }
}

/************************************************************************/
/* TrainerContextNoShard */
/************************************************************************/
void TrainerContextNoShard::Init(ModelShard* model_shard) {
  DoInit(model_shard->mutable_model());

  model_shard_ = model_shard;
  optimizer_ = model_shard_->mutable_optimizer();
}

void TrainerContextNoShard::TrainBatch() {
  const Instance& inst = op_context_->inst();
  if (op_context_batch_ != inst.batch()) {
    op_context_batch_ = inst.batch();
    op_context_->InitForward();
    op_context_->InitBackward();
  }

  op_context_->Forward();
  op_context_->Backward();
  optimizer_->Update(op_context_->mutable_grad());
  file_loss_ += op_context_->loss();
  file_loss_weight_ += 1;
}

void TrainerContextNoShard::PredictBatch() {
  const Instance& inst = op_context_->inst();
  if (op_context_batch_ != inst.batch()) {
    op_context_batch_ = inst.batch();
    op_context_->InitPredict();
  }

  op_context_->Predict();
}

/************************************************************************/
/* TrainerContextShard */
/************************************************************************/
void TrainerContextShard::Init(std::vector<ModelShard>* model_shards,
                               ModelShard* local_model_shard) {
  DoInit(local_model_shard->mutable_model());

  shard_size_ = (int)model_shards->size();
  model_shards_.resize(shard_size_);
  models_.resize(shard_size_);
  optimizers_.resize(shard_size_);
  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i] = &(*model_shards)[i];
    models_[i] = model_shards_[i]->mutable_model();
    optimizers_[i] = model_shards_[i]->mutable_optimizer();
  }

  pull_requests_.resize(shard_size_);
  pull_request_masks_.resize(shard_size_);
  params_.resize(shard_size_);
  grads_.resize(shard_size_);
  overwritten_params_.resize(shard_size_);
  for (int i = 0; i < shard_size_; ++i) {
    params_[i].reset(new TensorMap);
    grads_[i].reset(new TensorMap);
    overwritten_params_[i].reset(new TensorMap);
  }
  aux1_.resize(shard_size_);
  aux2_.resize(shard_size_);
}

void TrainerContextShard::TrainBatch() {
  const Instance& inst = op_context_->inst();
  if (op_context_batch_ != inst.batch()) {
    op_context_batch_ = inst.batch();
    op_context_->InitForward();
    op_context_->InitBackward();
  }

  Pull(1);
  op_context_->Forward();
  op_context_->Backward();
  Push();
  file_loss_ += op_context_->loss();
  file_loss_weight_ += 1;
}

void TrainerContextShard::PredictBatch() {
  const Instance& inst = op_context_->inst();
  if (op_context_batch_ != inst.batch()) {
    op_context_batch_ = inst.batch();
    op_context_->InitPredict();
  }

  Pull(0);
  op_context_->Predict();
}

void TrainerContextShard::CompletionHandler() {
  std::unique_lock<std::mutex> guard(wait_token_.mutex);
  if (--wait_token_.remain == 0) {
    wait_token_.cond.notify_all();
  }
}

void TrainerContextShard::WaitForCompletion() {
  std::unique_lock<std::mutex> guard(wait_token_.mutex);
  while (wait_token_.remain > 0) {
    wait_token_.cond.wait(guard);
  }
}

void TrainerContextShard::Pull(int is_train) {
  op_context_->GetPullRequest(&pull_request_);
  pull_request_.is_train = is_train;
  if (freq_filter_threshold_ > 0 && is_train) {
    FreqStore::GetIdFreqMap(op_context_->inst(), &pull_request_.id_freq_map);
  }
  local_model_->SplitPullRequest(pull_request_, &pull_requests_, &aux1_);

  pull_request_active_ = 0;
  for (int i = 0; i < shard_size_; ++i) {
    if (pull_requests_[i].empty()) {
      pull_request_masks_[i] = 0;
    } else {
      pull_request_masks_[i] = 1;
      ++pull_request_active_;
    }
  }

  wait_token_.remain = pull_request_active_;
  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      model_shards_[i]->AsyncPull(&pull_requests_[i], params_[i].get(),
                                  [this]() { CompletionHandler(); });
    } else {
      params_[i]->clear();
    }
  }
  WaitForCompletion();

  local_model_->SetParam(&params_);
}

void TrainerContextShard::Push() {
  local_model_->SplitGrad(local_model_->param(), op_context_->mutable_grad(),
                          &grads_, &aux2_);
  local_model_->SplitParam(op_context_->overwritten_param(),
                           &overwritten_params_, &aux2_);

  wait_token_.remain = pull_request_active_;
  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      model_shards_[i]->AsyncPush(grads_[i].get(), overwritten_params_[i].get(),
                                  [this]() { CompletionHandler(); });
    }
  }
  WaitForCompletion();
}

}  // namespace deepx_core
