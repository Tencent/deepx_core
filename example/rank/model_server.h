// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace deepx_core {

class Graph;
class Model;
class OpContext;

using feature_t = std::pair<uint64_t, float>;
using features_t = std::vector<feature_t>;

class ModelServer {
 private:
  std::unique_ptr<Graph> graph_;
  std::string target_name_;
  std::unique_ptr<Model> model_;

 public:
  ModelServer();
  ~ModelServer();
  ModelServer(const ModelServer&) = delete;
  ModelServer& operator=(const ModelServer&) = delete;

 public:
  bool Load(const std::string& file);
  bool LoadGraph(const std::string& file);
  bool LoadModel(const std::string& file);

 public:
  bool Predict(const features_t& features, float* prob) const;
  bool Predict(const features_t& features, std::vector<float>* probs) const;
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  // only for DTNModel
  bool DTNBatchPredict(const features_t& user_features,
                       const std::vector<features_t>& batch_item_features,
                       std::vector<std::vector<float>>* batch_probs) const;

 public:
  using op_context_ptr_t = std::unique_ptr<OpContext, void (*)(OpContext*)>;
  op_context_ptr_t NewOpContext() const;
  bool Predict(OpContext* op_context, const features_t& features,
               float* prob) const;
  bool Predict(OpContext* op_context, const features_t& features,
               std::vector<float>* probs) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  // only for DTNModel
  bool DTNBatchPredict(OpContext* op_context, const features_t& user_features,
                       const std::vector<features_t>& batch_item_features,
                       std::vector<std::vector<float>>* batch_probs) const;
};

}  // namespace deepx_core
