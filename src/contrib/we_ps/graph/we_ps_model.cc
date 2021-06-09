// Copyright 2021 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/contrib/we_ps/graph/we_ps_model.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/tensor/tensor_type.h>

namespace deepx_core {

void WePSModel::InitClient(WePSClient* client) noexcept { client_ = client; }

void WePSModel::InitGraph(const Graph* graph) noexcept { graph_ = graph; }

bool WePSModel::InitPlaceholder() {
  model_.reset(new Model);
  model_->Init(graph_);
  return model_->InitParamPlaceholder() && InitParamPlaceholder();
}

bool WePSModel::InitDenseParam() {
  for (auto& entry : *mutable_param()) {
    const std::string& name = entry.first;
    Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      const GraphNode* node = graph_->find_node(name);
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      DXINFO("Initializing TSR %s...", name.c_str());
      W.rand_init(engine_, node->initializer_type(),
                  (float_t)node->initializer_param1(),
                  (float_t)node->initializer_param2());
    }
  }
  if (!client_->SetTSR(param())) {
    DXINFO("Failed to SetTSR.");
    return false;
  }
  return true;
}

bool WePSModel::InitOptimizer(const std::string& optimizer,
                              const std::string& optimizer_config) {
  optimizer_ = NewWePSOptimizer(optimizer);
  if (!optimizer_) {
    return false;
  }
  optimizer_->Init(graph_, model_->mutable_param());
  if (!optimizer_->InitParam()) {
    return false;
  }
  return InitOptimizerConfig(optimizer_config);
}

bool WePSModel::InitOptimizerConfig(const std::string& optimizer_config) {
  StringMap config;
  if (!ParseConfig(optimizer_config, &config)) {
    DXERROR("Failed to parse WePSOptimizer config: %s.",
            optimizer_config.c_str());
    return false;
  }
  return optimizer_->InitConfig(config);
}

bool WePSModel::InitParamPlaceholder() {
  delta_param_.reset(new TensorMap);
  new_param_.reset(new TensorMap);
  for (const auto& entry : graph_->name_2_node()) {
    const GraphNode* node = entry.second;
    if (node->node_type() != GRAPH_NODE_TYPE_PARAM || !node->need_grad()) {
      continue;
    }
    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        auto& delta_W = delta_param_->insert<tsr_t>(node->name());
        delta_W.resize(node->shape());
        // no rand_init
        // 'new_param_' has no TSR
      } break;
      case TENSOR_TYPE_SRM: {
        auto& delta_W = delta_param_->insert<srm_t>(node->name());
        delta_W.set_col(node->shape()[1]);
        delta_W.set_initializer(node->initializer_type(),
                                (float_t)node->initializer_param1(),
                                (float_t)node->initializer_param2());
        // no reserve
        auto& new_W = new_param_->insert<srm_t>(node->name());
        new_W.set_col(node->shape()[1]);
        new_W.set_initializer(node->initializer_type(),
                              (float_t)node->initializer_param1(),
                              (float_t)node->initializer_param2());
        // no reserve
      } break;
    }
  }
  return true;
}

std::string WePSModel::GetOptimizerFile(const std::string& dir) noexcept {
  return dir + "/optimizer.bin";
}

bool WePSModel::SaveGraph(const Graph& graph) {
  if (!client_->SetGraph(graph)) {
    DXINFO("Failed to SetGraph.");
    return false;
  }
  return true;
}

bool WePSModel::LoadGraph(Graph* graph) {
  int graph_exist = 0;
  if (!client_->GetGraph(graph, &graph_exist)) {
    DXINFO("Failed to GetGraph.");
    return false;
  }
  return (bool)graph_exist;
}

bool WePSModel::LoadDenseParam() {
  if (!client_->GetTSR(mutable_param())) {
    DXINFO("Failed to GetTSR.");
    return false;
  }
  return true;
}

bool WePSModel::SaveOptimizer(const std::string& dir) const {
  return deepx_core::SaveWePSOptimizer(GetOptimizerFile(dir), *optimizer_);
}

bool WePSModel::LoadOptimizer(const std::string& dir,
                              const std::string& optimizer_config) {
  std::string name;
  if (!LoadWePSOptimizerName(GetOptimizerFile(dir), &name)) {
    return false;
  }
  optimizer_ = NewWePSOptimizer(name);
  if (!optimizer_) {
    return false;
  }
  optimizer_->Init(graph_, model_->mutable_param());
  if (!deepx_core::LoadWePSOptimizer(GetOptimizerFile(dir), optimizer_.get())) {
    return false;
  }

  if (!optimizer_config.empty()) {
    return InitOptimizerConfig(optimizer_config);
  }
  return true;
}

bool WePSModel::Pull(const PullRequest& pull_request) {
  mutable_param()->ClearSRMValue();
  if (!client_->GetSRM(pull_request.srm_map, mutable_param())) {
    DXINFO("Failed to GetSRM.");
    return false;
  }

  if (pull_request.is_train) {
    new_param_->ClearSRMValue();
    for (const auto& entry : pull_request.srm_map) {
      const std::string& name = entry.first;
      const id_set_t& id_set = entry.second;

      auto& local_W = mutable_param()->get<srm_t>(name);
      auto& new_W = new_param_->get<srm_t>(name);
      new_W.reserve(id_set.size());
      for (int_t id : id_set) {
        auto it = local_W.find(id);
        if (it == local_W.end()) {
          // get random values for missing keys
          const float_t* embedding = local_W.get_row(engine_, id);
          new_W.assign(id, embedding);
        }
      }
    }
  }

  return true;
}

bool WePSModel::Push(TensorMap* grad, TensorMap* overwritten_param) {
  if (!grad->empty()) {
    delta_param_->ClearSRMValue();
    optimizer_->Update(grad, delta_param_.get());

    if (!client_->UpdateTSR(*delta_param_, mutable_param())) {
      DXINFO("Failed to UpdateTSR.");
      return false;
    }
    // 'new_param_' must be pushed before calling 'UpdateSRM'.
    if (!client_->SetSRM(*new_param_)) {
      DXINFO("Failed to SetSRM.");
      return false;
    }
    if (!client_->UpdateSRM(*delta_param_)) {
      DXINFO("Failed to UpdateSRM.");
      return false;
    }
  }

  if (overwritten_param && !overwritten_param->empty()) {
    if (!client_->SetTSR(*overwritten_param)) {
      DXINFO("Failed to SetTSR.");
      return false;
    }
    if (!client_->SetSRM(*overwritten_param)) {
      DXINFO("Failed to SetSRM.");
      return false;
    }
  }

  return true;
}

}  // namespace deepx_core
