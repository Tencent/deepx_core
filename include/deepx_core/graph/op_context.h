// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/op.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* OpContext */
/************************************************************************/
class OpContext : public DataType {
 private:
  const Graph* graph_ = nullptr;
  TensorMap* param_ = nullptr;
  int forward_chain_size_ = 0;
  int backward_chain_size_ = 0;
  std::vector<std::unique_ptr<Op>> forward_chain_;
  std::vector<Op*> backward_chain_;
  int has_loss_ = 0;
  std::string loss_name_;
  Hidden hidden_;
  TensorMap ptr_;
  TensorMap grad_;
  TensorMap grad_ptr_;
  TensorMap overwritten_param_;
  TensorMap overwritten_ptr_;

 public:
  const Graph& graph() const noexcept { return *graph_; }
  TensorMap* mutable_param() noexcept { return param_; }
  const TensorMap& param() const noexcept { return *param_; }
  int has_loss() const noexcept { return has_loss_; }
  const std::string& loss_name() const noexcept { return loss_name_; }
  float_t loss() const noexcept { return hidden_.loss(); }
  Instance* mutable_inst() noexcept { return hidden_.mutable_inst(); }
  const Instance& inst() const noexcept { return hidden_.inst(); }
  Hidden* mutable_hidden() noexcept { return &hidden_; }
  const Hidden& hidden() const noexcept { return hidden_; }
  TensorMap* mutable_ptr() noexcept { return &ptr_; }
  const TensorMap& ptr() const noexcept { return ptr_; }
  TensorMap* mutable_grad() noexcept { return &grad_; }
  const TensorMap& grad() const noexcept { return grad_; }
  TensorMap* mutable_grad_ptr() noexcept { return &grad_ptr_; }
  const TensorMap& grad_ptr() const noexcept { return grad_ptr_; }
  TensorMap* mutable_overwritten_param() noexcept {
    return &overwritten_param_;
  }
  const TensorMap& overwritten_param() const noexcept {
    return overwritten_param_;
  }
  TensorMap* mutable_overwritten_ptr() noexcept { return &overwritten_ptr_; }
  const TensorMap& overwritten_ptr() const noexcept { return overwritten_ptr_; }

 private:
  int enable_profile_ = 0;
  struct OpProfile {
    const GraphNode* node = nullptr;
    double init_forward = 0;
    double init_predict = 0;
    double init_backward = 0;
    double forward = 0;
    double predict = 0;
    double backward = 0;
    double get_pull_request = 0;

    void clear() noexcept {
      node = nullptr;
      init_forward = 0;
      init_predict = 0;
      init_backward = 0;
      forward = 0;
      predict = 0;
      backward = 0;
      get_pull_request = 0;
    }
  };
  OpProfile global_profile_;
  std::unordered_map<Op*, OpProfile> profile_map_;

 private:
  void DumpProfile() const;

 public:
  OpContext();
  ~OpContext();
  void Init(const Graph* graph, TensorMap* param) noexcept;
  bool InitOp(const std::vector<int>& target_indices, int loss_index);
  bool InitOp(const std::vector<std::string>& target_names, int loss_index);
  bool InitOp(const std::vector<GraphTarget>& targets, int loss_index);
  void InitForward();
  void InitPredict();
  void InitBackward();
  void Forward();
  void Predict();
  void Backward();
  void GetPullRequest(PullRequest* pull_request);
};

}  // namespace deepx_core
