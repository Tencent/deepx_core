// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "model_server.h"
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/instance_reader.h>
#include <deepx_core/graph/model.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/instance/base.h>
#include "model_zoo/dtn.h"

namespace deepx_core {

using float_t = InstanceReader::float_t;
using int_t = InstanceReader::int_t;
using tsr_t = InstanceReader::tsr_t;
using csr_t = InstanceReader::csr_t;

static void EmplaceRow(const features_t& features, csr_t* X) {
  static constexpr float MAX_FEATURE_VALUE =
      InstanceReaderHelper<float, uint64_t>::MAX_FEATURE_VALUE;
  for (const auto& entry : features) {
    if (-MAX_FEATURE_VALUE <= entry.second &&
        entry.second <= MAX_FEATURE_VALUE) {
      X->emplace((int_t)entry.first, (float_t)entry.second);
    }
  }
  X->add_row();
}

ModelServer::ModelServer() {}

ModelServer::~ModelServer() {}

bool ModelServer::Load(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  DXINFO("Loading graph from %s...", file.c_str());
  graph_.reset(new Graph);
  if (!graph_->Read(is)) {
    return false;
  }

  // Graph target conventions.
  // Offline train, target 0.
  // Offline predict, target 1.
  // Online infer, target 2 if exists, otherwise target 1.
  if (graph_->target_size() >= 3) {
    target_name_ = graph_->target(2).name();
  } else {
    target_name_ = graph_->target(1).name();
  }
  DXINFO("Done.");

  DXINFO("Loading model from %s...", file.c_str());
  model_.reset(new Model);
  model_->Init(graph_.get());
  if (!model_->Read(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool ModelServer::LoadGraph(const std::string& file) {
  graph_.reset(new Graph);
  if (!graph_->Load(file)) {
    return false;
  }

  // Check out graph target conventions.
  if (graph_->target_size() >= 3) {
    target_name_ = graph_->target(2).name();
  } else {
    target_name_ = graph_->target(1).name();
  }
  return true;
}

bool ModelServer::LoadModel(const std::string& file) {
  model_.reset(new Model);
  model_->Init(graph_.get());
  return model_->Load(file);
}

bool ModelServer::Predict(const features_t& features, float* prob) const {
  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(X_NAME);
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  *prob = (float)P.data(0);
  return true;
}

bool ModelServer::Predict(const features_t& features,
                          std::vector<float>* probs) const {
  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(X_NAME);
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  const float_t* _P = P.data();
  probs->resize(col);
  for (int j = 0; j < col; ++j) {
    (*probs)[j] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(const std::vector<features_t>& batch_features,
                               std::vector<float>* batch_prob) const {
  if (batch_features.empty()) {
    return false;
  }

  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(X_NAME);
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  batch_prob->resize(X.row());
  const float_t* _P = P.data();
  for (int i = 0; i < X.row(); ++i) {
    (*batch_prob)[i] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(
    const std::vector<features_t>& batch_features,
    std::vector<std::vector<float>>* batch_probs) const {
  if (batch_features.empty()) {
    return false;
  }

  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(X_NAME);
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  batch_probs->resize(X.row());
  const float_t* _P = P.data();
  batch_probs->resize(X.row());
  for (int i = 0; i < X.row(); ++i) {
    auto& batch_prob = (*batch_probs)[i];
    batch_prob.resize(col);
    for (int j = 0; j < col; ++j) {
      batch_prob[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

bool ModelServer::DTNBatchPredict(
    const features_t& user_features,
    const std::vector<features_t>& batch_item_features,
    std::vector<std::vector<float>>* batch_probs) const {
  if (batch_item_features.empty()) {
    return false;
  }

  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& Xuser = inst->insert<csr_t>(DTN_X_USER_NAME);
  EmplaceRow(user_features, &Xuser);
  auto& Xitem = inst->insert<csr_t>(DTN_X_ITEM_NAME);
  for (const auto& features : batch_item_features) {
    EmplaceRow(features, &Xitem);
  }
  inst->set_batch(Xitem.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(Xitem.row(), col));
  batch_probs->resize(Xitem.row());
  const float_t* _P = P.data();
  batch_probs->resize(Xitem.row());
  for (int i = 0; i < Xitem.row(); ++i) {
    auto& batch_prob = (*batch_probs)[i];
    batch_prob.resize(col);
    for (int j = 0; j < col; ++j) {
      batch_prob[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

static void DeleteOpContext(OpContext* op_context) noexcept {
  delete op_context;
}

auto ModelServer::NewOpContext() const -> op_context_ptr_t {
  op_context_ptr_t op_context(new OpContext, DeleteOpContext);

  if (!graph_ || !model_) {
    op_context.reset();
    return op_context;
  }

  op_context->Init(graph_.get(), model_->mutable_param());
  if (!op_context->InitOp({target_name_}, -1)) {
    op_context.reset();
    return op_context;
  }

  return op_context;
}

bool ModelServer::Predict(OpContext* op_context, const features_t& features,
                          float* prob) const {
  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(X_NAME);
  X.clear();
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  *prob = (float)P.data(0);
  return true;
}

bool ModelServer::Predict(OpContext* op_context, const features_t& features,
                          std::vector<float>* probs) const {
  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(X_NAME);
  X.clear();
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  const float_t* _P = P.data();
  probs->resize(col);
  for (int j = 0; j < col; ++j) {
    (*probs)[j] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(OpContext* op_context,
                               const std::vector<features_t>& batch_features,
                               std::vector<float>* batch_prob) const {
  if (batch_features.empty()) {
    return false;
  }

  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(X_NAME);
  X.clear();
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  batch_prob->resize(X.row());
  const float_t* _P = P.data();
  for (int i = 0; i < X.row(); ++i) {
    (*batch_prob)[i] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(
    OpContext* op_context, const std::vector<features_t>& batch_features,
    std::vector<std::vector<float>>* batch_probs) const {
  if (batch_features.empty()) {
    return false;
  }

  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(X_NAME);
  X.clear();
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  batch_probs->resize(X.row());
  const float_t* _P = P.data();
  batch_probs->resize(X.row());
  for (int i = 0; i < X.row(); ++i) {
    auto& batch_prob = (*batch_probs)[i];
    batch_prob.resize(col);
    for (int j = 0; j < col; ++j) {
      batch_prob[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

bool ModelServer::DTNBatchPredict(
    OpContext* op_context, const features_t& user_features,
    const std::vector<features_t>& batch_item_features,
    std::vector<std::vector<float>>* batch_probs) const {
  if (batch_item_features.empty()) {
    return false;
  }

  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& Xuser = inst->get_or_insert<csr_t>(DTN_X_USER_NAME);
  Xuser.clear();
  EmplaceRow(user_features, &Xuser);
  auto& Xitem = inst->get_or_insert<csr_t>(DTN_X_ITEM_NAME);
  Xitem.clear();
  for (const auto& features : batch_item_features) {
    EmplaceRow(features, &Xitem);
  }
  inst->set_batch(Xitem.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(Xitem.row(), col));
  batch_probs->resize(Xitem.row());
  const float_t* _P = P.data();
  batch_probs->resize(Xitem.row());
  for (int i = 0; i < Xitem.row(); ++i) {
    auto& batch_prob = (*batch_probs)[i];
    batch_prob.resize(col);
    for (int j = 0; j < col; ++j) {
      batch_prob[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

}  // namespace deepx_core
