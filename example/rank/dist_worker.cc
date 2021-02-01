// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/common/misc.h>
#include <deepx_core/common/profile_util.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/ps/tcp_connection.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "dist_flags.h"
#include "model_zoo.h"
#include "trainer_context.h"

namespace deepx_core {
namespace {

/************************************************************************/
/* TrainerContextDist */
/************************************************************************/
class TrainerContextDist : public TrainerContext {
 private:
  IoContext io_;
  TcpConnections ps_conns_;
  InputStringStream is_;
  OutputStringStream os_;
  int shard_size_ = 0;
  PullRequest pull_request_;
  std::vector<PullRequest> pull_requests_;
  std::vector<int> pull_request_masks_;
  std::vector<std::unique_ptr<TensorMap>> params_;
  std::vector<std::unique_ptr<TensorMap>> grads_;
  std::vector<std::unique_ptr<TensorMap>> overwritten_params_;
  std::vector<id_set_t*> aux1_;
  std::vector<srm_t*> aux2_;

 public:
  TrainerContextDist();
  void Init(ModelShard* local_model_shard);
  void TrainBatch() override;
  void PredictBatch() override;

 private:
  void Pull();
  void Push();
};

TrainerContextDist::TrainerContextDist() : io_(), ps_conns_(&io_) {}

void TrainerContextDist::Init(ModelShard* local_model_shard) {
  _Init(local_model_shard);

  DXCHECK_THROW(ps_conns_.ConnectRetry(FLAGS_ps_endpoints) == 0);

  shard_size_ = FLAGS_shard.shard_size();
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

void TrainerContextDist::TrainBatch() {
  const Instance& inst = op_context_->inst();
  if (op_context_batch_ != inst.batch()) {
    op_context_batch_ = inst.batch();
    if (!enable_profile_) {
      op_context_->InitForward();
      op_context_->InitBackward();
    } else {
      {
        NanosecondTimerGuard guard(profile_map_["OpContext::InitForward"]);
        op_context_->InitForward();
      }
      {
        NanosecondTimerGuard guard(profile_map_["OpContext::InitBackward"]);
        op_context_->InitBackward();
      }
    }
  }

  if (!enable_profile_) {
    op_context_->GetPullRequest(&pull_request_);
    Pull();
    op_context_->Forward();
    op_context_->Backward();
    Push();
  } else {
    {
      NanosecondTimerGuard guard(profile_map_["OpContext::GetPullRequest"]);
      op_context_->GetPullRequest(&pull_request_);
    }
    {
      NanosecondTimerGuard guard(profile_map_["Pull"]);
      Pull();
    }
    {
      NanosecondTimerGuard guard(profile_map_["OpContext::Forward"]);
      op_context_->Forward();
    }
    {
      NanosecondTimerGuard guard(profile_map_["OpContext::Backward"]);
      op_context_->Backward();
    }
    {
      NanosecondTimerGuard guard(profile_map_["Push"]);
      Push();
    }
  }

  file_loss_ += op_context_->loss();
  file_loss_weight_ += 1;
}

void TrainerContextDist::PredictBatch() {
  const Instance& inst = op_context_->inst();
  if (op_context_batch_ != inst.batch()) {
    op_context_batch_ = inst.batch();
    if (!enable_profile_) {
      op_context_->InitPredict();
    } else {
      NanosecondTimerGuard guard(profile_map_["OpContext::InitPredict"]);
      op_context_->InitPredict();
    }
  }

  if (!enable_profile_) {
    op_context_->GetPullRequest(&pull_request_);
    Pull();
    op_context_->Predict();
  } else {
    {
      NanosecondTimerGuard guard(profile_map_["OpContext::GetPullRequest"]);
      op_context_->GetPullRequest(&pull_request_);
    }
    {
      NanosecondTimerGuard guard(profile_map_["Pull"]);
      Pull();
    }
    {
      NanosecondTimerGuard guard(profile_map_["OpContext::Predict"]);
      op_context_->Predict();
    }
  }
}

void TrainerContextDist::Pull() {
  if (FLAGS_freq_filter_threshold > 0 && FLAGS_is_train) {
    FreqStore::GetIdFreqMap(op_context_->inst(), &pull_request_.id_freq_map);
  }
  pull_request_.is_train = FLAGS_is_train;
  local_model_shard_->SplitPullRequest(pull_request_, &pull_requests_, &aux1_);

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_requests_[i].empty()) {
      pull_request_masks_[i] = 0;
    } else {
      pull_request_masks_[i] = 1;
    }
  }

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      std::string& buf =
          ps_conns_[i]->mutable_out_message()->mutable_pull_request()->buf;
      buf.clear();
      os_.SetView(&buf);
      os_ << pull_requests_[i];
      DXCHECK_THROW(os_);
    }
  }

  DXCHECK_THROW(ps_conns_.RpcPullRequest(&pull_request_masks_) == 0);

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      const const_string_view& buf =
          ps_conns_[i]->in_message().pull_response().buf;
      is_.SetView(buf.data(), buf.size());
      // view, zero-copy
      ReadView(is_, *params_[i]);
      DXCHECK_THROW(is_);
    } else {
      params_[i]->clear();
    }
  }

  local_model_shard_->mutable_model()->SetParam(&params_);
}

void TrainerContextDist::Push() {
  local_model_shard_->SplitGrad(local_model_shard_->param(),
                                op_context_->mutable_grad(), &grads_, &aux2_);
  local_model_shard_->SplitParam(op_context_->overwritten_param(),
                                 &overwritten_params_, &aux2_);

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      std::string& buf =
          ps_conns_[i]->mutable_out_message()->mutable_push_notify()->buf;
      buf.clear();
      os_.SetView(&buf);
      os_ << *grads_[i] << *overwritten_params_[i];
      DXCHECK_THROW(os_);
    }
  }

  DXCHECK_THROW(ps_conns_.RpcPushNotify(&pull_request_masks_) == 0);
}

/************************************************************************/
/* TrainerDist */
/************************************************************************/
class TrainerDist {
 private:
  IoContext io_;
  TcpConnection cs_conn_;
  Graph graph_;
  ModelShard local_model_shard_;
  TrainerContextDist context_;

 public:
  TrainerDist();
  void Init();
  void Train();
  void Predict();
};

TrainerDist::TrainerDist() : io_(), cs_conn_(&io_) {}

void TrainerDist::Init() {
  if (FLAGS_is_train) {
    if (FLAGS_in_model.empty()) {
      std::unique_ptr<ModelZoo> model_zoo(NewModelZoo(FLAGS_model));
      DXCHECK_THROW(model_zoo);
      StringMap config;
      DXCHECK_THROW(ParseConfig(FLAGS_model_config, &config));
      DXCHECK_THROW(model_zoo->InitConfig(config));
      DXCHECK_THROW(model_zoo->InitGraph(&graph_));
    } else {
      DXCHECK_THROW(LoadGraph(FLAGS_in_model, &graph_));
    }
  } else {
    DXCHECK_THROW(LoadGraph(FLAGS_in_model, &graph_));
  }

  local_model_shard_.InitShard(&FLAGS_shard, 0);
  local_model_shard_.InitGraph(&graph_);
  DXCHECK_THROW(local_model_shard_.InitModelPlaceholder());

  context_.set_instance_reader(FLAGS_instance_reader);
  context_.set_instance_reader_config(FLAGS_instance_reader_config);
  context_.set_batch(FLAGS_batch);
  context_.set_verbose(FLAGS_verbose);
  if (FLAGS_freq_filter_threshold > 0) {
    context_.set_freq_filter_threshold(
        (DataType::freq_t)FLAGS_freq_filter_threshold);
  }
  // Check out graph target conventions.
  context_.set_target_name(graph_.target(FLAGS_is_train ? 0 : 1).name());
  context_.Init(&local_model_shard_);
}

void TrainerDist::Train() {
  int epoch = 0;
  std::string file;
  for (;;) {
    DXINFO("Epoch %d begins.", epoch + 1);
    DXCHECK_THROW(cs_conn_.ConnectRetry(FLAGS_cs_endpoint) == 0);
    for (;;) {
      if (cs_conn_.RpcFileRequest() == 0) {
        epoch = cs_conn_.in_message().file_response().epoch;
        file = cs_conn_.in_message().file_response().file;
        if (file.empty()) {
          DXINFO("Worker got no new file.");
          std::this_thread::sleep_for(std::chrono::seconds(5));  // magic number
          continue;
        } else {
          DXINFO("Worker has got file: %s.", file.c_str());
          context_.TrainFile(0, file);
          auto* file_finished_notify =
              cs_conn_.mutable_out_message()->mutable_file_finish_notify();
          file_finished_notify->file = file;
          file_finished_notify->loss = context_.file_loss();
          file_finished_notify->loss_weight = context_.file_loss_weight();
          DXCHECK_THROW(cs_conn_.RpcFileFinishNotify() == 0);
        }
      } else {
        DXINFO("Failed to RpcFileRequest.");
        break;
      }
    }

    cs_conn_.Close();
    DXINFO("Epoch %d completed.", epoch + 1);
    if (epoch == FLAGS_epoch - 1) {
      break;
    }
  }
}

void TrainerDist::Predict() {
  std::string file;
  DXCHECK_THROW(cs_conn_.ConnectRetry(FLAGS_cs_endpoint) == 0);
  for (;;) {
    if (cs_conn_.RpcFileRequest() == 0) {
      file = cs_conn_.in_message().file_response().file;
      if (file.empty()) {
        DXINFO("Worker got no new file.");
        std::this_thread::sleep_for(std::chrono::seconds(5));  // magic number
        continue;
      } else {
        DXINFO("Worker has got file: %s.", file.c_str());
        context_.PredictFile(0, file,
                             GetOutputPredictFile(FLAGS_out_predict, file));
        auto* file_finished_notify =
            cs_conn_.mutable_out_message()->mutable_file_finish_notify();
        file_finished_notify->file = file;
        file_finished_notify->loss = 0;
        file_finished_notify->loss_weight = 0;
        DXCHECK_THROW(cs_conn_.RpcFileFinishNotify() == 0);
      }
    } else {
      DXINFO("Failed to RpcFileRequest.");
      break;
    }
  }

  cs_conn_.Close();
}

}  // namespace

void RunWorker() {
  TrainerDist trainer;
  trainer.Init();
  if (FLAGS_is_train) {
    trainer.Train();
  } else {
    trainer.Predict();
  }
}

}  // namespace deepx_core
