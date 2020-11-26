// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// include all headers needed by operators
#include <deepx_core/common/class_factory.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/op.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace deepx_core {

#define OP_REGISTER(class_name, name) \
  CLASS_FACTORY_REGISTER(Op, class_name, name)
#define OP_NEW(name) CLASS_FACTORY_NEW(Op, name)
#define OP_NAMES() CLASS_FACTORY_NAMES(Op)
#define DEFINE_OP_LIKE(clazz_name) \
  const char* class_name() const noexcept override { return #clazz_name; }

#define GRAPH_NODE_OP_REGISTER(name) \
  GRAPH_NODE_REGISTER(name##Node);   \
  OP_REGISTER(name##Op, #name "Node")

/************************************************************************/
/* OpImpl */
/************************************************************************/
class OpImpl : public Op {
 protected:
  const Graph* graph_ = nullptr;
  const GraphNode* node_ = nullptr;
  TensorMap* param_ = nullptr;
  Hidden* hidden_ = nullptr;
  TensorMap* ptr_ = nullptr;
  TensorMap* grad_ = nullptr;
  TensorMap* grad_ptr_ = nullptr;
  TensorMap* overwritten_param_ = nullptr;
  TensorMap* overwritten_ptr_ = nullptr;

 public:
  void Init(const Graph* graph, const GraphNode* node, TensorMap* param,
            Hidden* hidden, TensorMap* ptr, TensorMap* grad,
            TensorMap* grad_ptr, TensorMap* overwritten_param,
            TensorMap* overwritten_ptr) final {
    graph_ = graph;
    node_ = node;
    param_ = param;
    hidden_ = hidden;
    ptr_ = ptr;
    grad_ = grad;
    grad_ptr_ = grad_ptr;
    overwritten_param_ = overwritten_param;
    overwritten_ptr_ = overwritten_ptr;
  }

  void InitForward() override {}
  void InitPredict() override { InitForward(); }
  void InitBackward() override {}
  void Forward() override {}
  void Predict() override { Forward(); }
  void Backward() override {}
  void GetPullRequest(PullRequest* pull_request) const override;

 protected:
  tsr_t* InitHiddenTSR(const GraphNode* node, const Shape& shape) {
    auto& Z = hidden_->get_or_insert<tsr_t>(node->name());
    Z.resize(shape);
    tsr_t* tsr = &Z;
    (*ptr_)[node->name()] = tsr;
    return tsr;
  }

  tsr_t* InitHiddenTSRView(const GraphNode* node) {
    auto& Z = hidden_->get_or_insert<tsr_t>(node->name());
    tsr_t* tsr = &Z;
    (*ptr_)[node->name()] = tsr;
    return tsr;
  }

  tsr_t* GetPtrTSR(const GraphNode* node) {
    return ptr_->get<tsr_t*>(node->name());
  }

  srm_t* GetPtrSRM(const GraphNode* node) {
    return ptr_->get<srm_t*>(node->name());
  }

  csr_t* GetPtrCSR(const GraphNode* node) {
    return ptr_->get<csr_t*>(node->name());
  }

  tsri_t* GetPtrTSRI(const GraphNode* node) {
    return ptr_->get<tsri_t*>(node->name());
  }

  tsrs_t* GetPtrTSRS(const GraphNode* node) {
    return ptr_->get<tsrs_t*>(node->name());
  }

  tsr_t* InitPtrTSR(const GraphNode* node, tsr_t* tsr) {
    (*ptr_)[node->name()] = tsr;
    return tsr;
  }

  srm_t* InitPtrSRM(const GraphNode* node, srm_t* srm) {
    (*ptr_)[node->name()] = srm;
    return srm;
  }

  csr_t* InitPtrCSR(const GraphNode* node, csr_t* csr) {
    (*ptr_)[node->name()] = csr;
    return csr;
  }

  tsri_t* InitPtrTSRI(const GraphNode* node, tsri_t* tsri) {
    (*ptr_)[node->name()] = tsri;
    return tsri;
  }

  tsrs_t* InitPtrTSRS(const GraphNode* node, tsrs_t* tsrs) {
    (*ptr_)[node->name()] = tsrs;
    return tsrs;
  }

  tsr_t* InitGradTSR(const GraphNode* node, const Shape& shape) {
    if (node->need_grad()) {
      auto& G = grad_->get_or_insert<tsr_t>(node->name());
      G.resize(shape);
      tsr_t* tsr = &G;
      (*grad_ptr_)[node->name()] = tsr;
      return tsr;
    }
    return nullptr;
  }

  tsr_t* InitGradTSRView(const GraphNode* node) {
    if (node->need_grad()) {
      auto& G = grad_->get_or_insert<tsr_t>(node->name());
      tsr_t* tsr = &G;
      (*grad_ptr_)[node->name()] = tsr;
      return tsr;
    }
    return nullptr;
  }

  srm_t* InitGradSRM(const GraphNode* node, const Shape& shape) {
    if (node->need_grad()) {
      auto& G = grad_->get_or_insert<srm_t>(node->name());
      G.set_col(shape[1]);
      srm_t* srm = &G;
      (*grad_ptr_)[node->name()] = srm;
      return srm;
    }
    return nullptr;
  }

  tsr_t* GetGradPtrTSR(const GraphNode* node) {
    if (node->need_grad()) {
      return grad_ptr_->get<tsr_t*>(node->name());
    }
    return nullptr;
  }

  tsr_t* InitOverwrittenParamTSR(const GraphNode* node, const Shape& shape) {
    auto& W = overwritten_param_->get_or_insert<tsr_t>(node->name());
    W.resize(shape);
    tsr_t* tsr = &W;
    (*overwritten_ptr_)[node->name()] = tsr;
    return tsr;
  }

  tsr_t* InitOverwrittenParamTSRView(const GraphNode* node) {
    auto& W = overwritten_param_->get_or_insert<tsr_t>(node->name());
    tsr_t* tsr = &W;
    (*overwritten_ptr_)[node->name()] = tsr;
    return tsr;
  }

  srm_t* InitOverwrittenParamSRM(const GraphNode* node, const Shape& shape) {
    auto& W = overwritten_param_->get_or_insert<srm_t>(node->name());
    W.set_col(shape[1]);
    srm_t* srm = &W;
    (*overwritten_ptr_)[node->name()] = srm;
    return srm;
  }

  srm_t* InitOverwrittenParamSRM(const GraphNode* node, int col) {
    auto& W = overwritten_param_->get_or_insert<srm_t>(node->name());
    W.set_col(col);
    srm_t* srm = &W;
    (*overwritten_ptr_)[node->name()] = srm;
    return srm;
  }

  tsr_t* GetOverwrittenPtrTSR(const GraphNode* node) {
    return overwritten_ptr_->get<tsr_t*>(node->name());
  }

  srm_t* GetOverwrittenPtrSRM(const GraphNode* node) {
    return overwritten_ptr_->get<srm_t*>(node->name());
  }
};

/************************************************************************/
/* OpUnaryBase */
/************************************************************************/
class OpUnaryBase : public OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const tsr_t* X_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;

 public:
  virtual const Shape& InferShape() = 0;

  void InitForward() override {
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    Z_ = InitHiddenTSR(node_, InferShape());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
  }
};

/************************************************************************/
/* OpUnaryElementWiseBase */
/************************************************************************/
class OpUnaryElementWiseBase : public OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const tsr_t* X_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;

 public:
  void InitForward() override {
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    Z_ = InitHiddenTSR(node_, X_->shape());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
  }
};

/************************************************************************/
/* OpBinaryBase */
/************************************************************************/
class OpBinaryBase : public OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Ynode_ = nullptr;
  const tsr_t* X_ = nullptr;
  const tsr_t* Y_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* gY_ = nullptr;

 public:
  virtual const Shape& InferShape() = 0;

  void InitForward() override {
    Xnode_ = node_->input(0);
    Ynode_ = node_->input(1);
    X_ = GetPtrTSR(Xnode_);
    Y_ = GetPtrTSR(Ynode_);
    Z_ = InitHiddenTSR(node_, InferShape());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
    gY_ = InitGradTSR(Ynode_, Y_->shape());
  }
};

/************************************************************************/
/* OpBinaryElementWiseBase */
/************************************************************************/
class OpBinaryElementWiseBase : public OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Ynode_ = nullptr;
  const tsr_t* X_ = nullptr;
  const tsr_t* Y_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* gY_ = nullptr;

 public:
  void InitForward() override {
    Xnode_ = node_->input(0);
    Ynode_ = node_->input(1);
    X_ = GetPtrTSR(Xnode_);
    Y_ = GetPtrTSR(Ynode_);
    if (X_->shape() != Y_->shape()) {
      DXTHROW_RUNTIME_ERROR("Inconsistent shape: %s vs %s.",
                            to_string(X_->shape()).c_str(),
                            to_string(Y_->shape()).c_str());
    }
    Z_ = InitHiddenTSR(node_, X_->shape());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
    gY_ = InitGradTSR(Ynode_, Y_->shape());
  }
};

}  // namespace deepx_core
