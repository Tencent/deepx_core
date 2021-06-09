// Copyright 2021 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
//

#pragma once
#include <deepx_core/contrib/we_ps/client/we_ps_client.h>
#include <deepx_core/contrib/we_ps/optimizer/we_ps_optimizer.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <random>
#include <string>

namespace deepx_core {

/************************************************************************/
/* WePSModel */
/************************************************************************/
class WePSModel : public DataType {
 private:
  std::default_random_engine engine_;
  WePSClient* client_ = nullptr;
  const Graph* graph_ = nullptr;
  std::unique_ptr<Model> model_;
  std::unique_ptr<WePSOptimizer> optimizer_;
  std::unique_ptr<TensorMap> delta_param_;
  std::unique_ptr<TensorMap> new_param_;

 public:
  template <typename Int>
  void seed(Int s) {
    engine_.seed((std::default_random_engine::result_type)s);
  }
  std::default_random_engine& engine() noexcept { return engine_; }
  WePSClient* mutable_client() noexcept { return client_; }
  const Graph& graph() const noexcept { return *graph_; }
  Model* mutable_model() noexcept { return model_.get(); }
  const Model& model() const noexcept { return *model_; }
  TensorMap* mutable_param() noexcept { return model_->mutable_param(); }
  const TensorMap& param() const noexcept { return model_->param(); }
  WePSOptimizer* mutable_optimizer() noexcept { return optimizer_.get(); }
  const WePSOptimizer& optimizer() const noexcept { return *optimizer_; }
  TensorMap* mutable_delta_param() noexcept { return delta_param_.get(); }
  const TensorMap& delta_param() const noexcept { return *delta_param_; }
  TensorMap* mutable_new_param() noexcept { return new_param_.get(); }
  const TensorMap& new_param() const noexcept { return *new_param_; }

 public:
  void InitClient(WePSClient* client) noexcept;
  void InitGraph(const Graph* graph) noexcept;
  bool InitPlaceholder();
  bool InitDenseParam();
  bool InitOptimizer(const std::string& optimizer,
                     const std::string& optimizer_config);
  bool InitOptimizerConfig(const std::string& optimizer_config);

 private:
  bool InitParamPlaceholder();

 private:
  static std::string GetOptimizerFile(const std::string& dir) noexcept;

 public:
  bool SaveGraph(const Graph& graph);
  bool LoadGraph(Graph* graph);
  bool LoadDenseParam();
  bool SaveOptimizer(const std::string& dir) const;
  bool LoadOptimizer(const std::string& dir,
                     const std::string& optimizer_config);

 public:
  bool Pull(const PullRequest& pull_request);
  bool Push(TensorMap* grad, TensorMap* overwritten_param);
};

}  // namespace deepx_core
