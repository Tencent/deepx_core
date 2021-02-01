// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/profile_util.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/op_context.h>
#include <cstdlib>  // getenv
#include <cstring>  // strcmp
#include <unordered_set>
#include <utility>

namespace deepx_core {

void OpContext::DumpProfile() const {
  if (profile_map_.empty()) {
    return;
  }

  std::vector<ProfileItem> items;
  auto add_op_profile = [&items](const OpProfile& op_profile) {
#define ADD_OP_PROFILE_MEMBER(member, method)     \
  if (op_profile.member > 0) {                    \
    std::string phase;                            \
    if (op_profile.node) {                        \
      phase = op_profile.node->class_name();      \
      phase += "::";                              \
      phase += #method;                           \
      phase += "(";                               \
      phase += op_profile.node->name();           \
      phase += ")";                               \
    } else {                                      \
      phase = #method;                            \
    }                                             \
    items.emplace_back(phase, op_profile.member); \
  }
    ADD_OP_PROFILE_MEMBER(init_forward, InitForward);
    ADD_OP_PROFILE_MEMBER(init_predict, InitPredict);
    ADD_OP_PROFILE_MEMBER(init_backward, InitBackward);
    ADD_OP_PROFILE_MEMBER(forward, Forward);
    ADD_OP_PROFILE_MEMBER(predict, Predict);
    ADD_OP_PROFILE_MEMBER(backward, Backward);
    ADD_OP_PROFILE_MEMBER(get_pull_request, GetPullRequest);
#undef ADD_OP_PROFILE_MEMBER
  };
  add_op_profile(global_profile_);
  for (const auto& entry : profile_map_) {
    add_op_profile(entry.second);
  }
  DumpProfileItems(&items);
}

OpContext::OpContext() {
  const char* enable_profile = getenv("DEEPX_OP_CONTEXT_ENABLE_PROFILE");
  if (enable_profile && strcmp(enable_profile, "1") == 0) {
    enable_profile_ = 1;
  } else {
    enable_profile_ = 0;
  }
}

OpContext::~OpContext() {
  if (enable_profile_) {
    DumpProfile();
  }
}

void OpContext::Init(const Graph* graph, TensorMap* param) noexcept {
  graph_ = graph;
  param_ = param;
}

bool OpContext::InitOp(const std::vector<int>& target_indices, int loss_index) {
  std::vector<GraphTarget> targets;
  targets.reserve(target_indices.size());
  for (size_t i = 0; i < target_indices.size(); ++i) {  // NOLINT
    if (target_indices[i] >= graph_->target_size()) {
      DXERROR("Invalid target index: %d.", target_indices[i]);
      return false;
    }
    targets.emplace_back(graph_->target(target_indices[i]));
  }
  return InitOp(targets, loss_index);
}

bool OpContext::InitOp(const std::vector<std::string>& target_names,
                       int loss_index) {
  std::vector<GraphTarget> targets;
  for (size_t i = 0; i < target_names.size(); ++i) {  // NOLINT
    const GraphTarget* target = graph_->find_target(target_names[i]);
    if (target == nullptr) {
      DXERROR("Invalid target name: %s.", target_names[i].c_str());
      return false;
    }
    targets.emplace_back(*target);
  }
  return InitOp(targets, loss_index);
}

bool OpContext::InitOp(const std::vector<GraphTarget>& targets,
                       int loss_index) {
  forward_chain_size_ = 0;
  backward_chain_size_ = 0;
  forward_chain_.clear();
  backward_chain_.clear();
  has_loss_ = 0;
  loss_name_.clear();
  hidden_.clear();
  ptr_.clear();
  grad_.clear();
  grad_ptr_.clear();
  overwritten_param_.clear();
  overwritten_ptr_.clear();

  if (enable_profile_) {
    DumpProfile();
    global_profile_.clear();
    profile_map_.clear();
  }

  std::unordered_set<std::string> dedup;
  for (size_t i = 0; i < targets.size(); ++i) {
    int is_loss_index = ((int)i == loss_index);
    const GraphTarget& target = targets[i];
    for (int j = 0; j < target.forward_size(); ++j) {  // NOLINT
      const GraphNode* node = target.forward(j);
      if (dedup.count(node->name()) > 0) {
        continue;
      }

      std::unique_ptr<Op> op(NewOp(node->class_name()));
      if (!op) {
        return false;
      }
      op->Init(graph_, node, param_, &hidden_, &ptr_, &grad_, &grad_ptr_,
               &overwritten_param_, &overwritten_ptr_);
      ++forward_chain_size_;
      forward_chain_.emplace_back(std::move(op));
      if (is_loss_index) {
        ++backward_chain_size_;
        backward_chain_.emplace_back(forward_chain_.back().get());
      }
      if (enable_profile_) {
        profile_map_[forward_chain_.back().get()].node = node;
      }
      dedup.emplace(node->name());
    }

    if (is_loss_index) {
      has_loss_ = 1;
      loss_name_ = target.name();
    }
  }
  return true;
}

void OpContext::InitForward() {
  if (!enable_profile_) {
    for (int i = 0; i < forward_chain_size_; ++i) {
      forward_chain_[i]->InitForward();
    }

    if (has_loss_) {
      auto& Z = hidden_.get<tsr_t>(loss_name_);
      DXCHECK_THROW(Z.is_scalar());
      hidden_.set_loss(Z.data());
    } else {
      hidden_.clear_loss();
    }
  } else {
    for (int i = 0; i < forward_chain_size_; ++i) {
      Op* op = forward_chain_[i].get();
      NanosecondTimerGuard guard(profile_map_[op].init_forward);
      op->InitForward();
    }

    NanosecondTimerGuard guard(global_profile_.init_forward);
    if (has_loss_) {
      auto& Z = hidden_.get<tsr_t>(loss_name_);
      DXCHECK_THROW(Z.is_scalar());
      hidden_.set_loss(Z.data());
    } else {
      hidden_.clear_loss();
    }
  }
}

void OpContext::InitPredict() {
  if (!enable_profile_) {
    for (int i = 0; i < forward_chain_size_; ++i) {
      forward_chain_[i]->InitPredict();
    }

    if (has_loss_) {
      auto& Z = hidden_.get<tsr_t>(loss_name_);
      DXCHECK_THROW(Z.is_scalar());
      hidden_.set_loss(Z.data());
    } else {
      hidden_.clear_loss();
    }
  } else {
    for (int i = 0; i < forward_chain_size_; ++i) {
      Op* op = forward_chain_[i].get();
      NanosecondTimerGuard guard(profile_map_[op].init_predict);
      op->InitPredict();
    }

    NanosecondTimerGuard guard(global_profile_.init_predict);
    if (has_loss_) {
      auto& Z = hidden_.get<tsr_t>(loss_name_);
      DXCHECK_THROW(Z.is_scalar());
      hidden_.set_loss(Z.data());
    } else {
      hidden_.clear_loss();
    }
  }
}

void OpContext::InitBackward() {
  if (!enable_profile_) {
    DXCHECK_THROW(has_loss_);
    auto& G = grad_.get_or_insert<tsr_t>(loss_name_);
    G.resize(1);
    grad_ptr_[loss_name_] = &G;

    for (int i = 0; i < backward_chain_size_; ++i) {
      backward_chain_[backward_chain_size_ - i - 1]->InitBackward();
    }
  } else {
    {
      NanosecondTimerGuard guard(global_profile_.init_backward);
      DXCHECK_THROW(has_loss_);
      auto& G = grad_.get_or_insert<tsr_t>(loss_name_);
      G.resize(1);
      grad_ptr_[loss_name_] = &G;
    }

    for (int i = 0; i < backward_chain_size_; ++i) {
      Op* op = backward_chain_[backward_chain_size_ - i - 1];
      NanosecondTimerGuard guard(profile_map_[op].init_backward);
      op->InitBackward();
    }
  }
}

void OpContext::Forward() {
  if (!enable_profile_) {
    for (int i = 0; i < forward_chain_size_; ++i) {
      forward_chain_[i]->Forward();
    }
  } else {
    for (int i = 0; i < forward_chain_size_; ++i) {
      Op* op = forward_chain_[i].get();
      NanosecondTimerGuard guard(profile_map_[op].forward);
      op->Forward();
    }
  }
}

void OpContext::Predict() {
  if (!enable_profile_) {
    for (int i = 0; i < forward_chain_size_; ++i) {
      forward_chain_[i]->Predict();
    }
  } else {
    for (int i = 0; i < forward_chain_size_; ++i) {
      Op* op = forward_chain_[i].get();
      NanosecondTimerGuard guard(profile_map_[op].predict);
      op->Predict();
    }
  }
}

void OpContext::Backward() {
  if (!enable_profile_) {
    grad_.ZerosValue();

    if (has_loss_) {
      auto& G = grad_.unsafe_get<tsr_t>(loss_name_);
      G.data(0) = 1;
    }

    overwritten_param_.ZerosValue();

    for (int i = 0; i < backward_chain_size_; ++i) {
      backward_chain_[backward_chain_size_ - i - 1]->Backward();
    }
  } else {
    {
      NanosecondTimerGuard guard(global_profile_.backward);
      grad_.ZerosValue();

      if (has_loss_) {
        auto& G = grad_.unsafe_get<tsr_t>(loss_name_);
        G.data(0) = 1;
      }

      overwritten_param_.ZerosValue();
    }

    for (int i = 0; i < backward_chain_size_; ++i) {
      Op* op = backward_chain_[backward_chain_size_ - i - 1];
      NanosecondTimerGuard guard(profile_map_[op].backward);
      op->Backward();
    }
  }
}

void OpContext::GetPullRequest(PullRequest* pull_request) {
  if (!enable_profile_) {
    pull_request->clear();
    for (int i = 0; i < forward_chain_size_; ++i) {
      forward_chain_[i]->GetPullRequest(pull_request);
    }
  } else {
    {
      NanosecondTimerGuard guard(global_profile_.get_pull_request);
      pull_request->clear();
    }

    for (int i = 0; i < forward_chain_size_; ++i) {
      Op* op = forward_chain_[i].get();
      NanosecondTimerGuard guard(profile_map_[op].get_pull_request);
      op->GetPullRequest(pull_request);
    }
  }
}

}  // namespace deepx_core
