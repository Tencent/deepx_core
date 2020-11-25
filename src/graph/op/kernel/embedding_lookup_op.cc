// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

bool EmbeddingLookupInferShape(int Xrow, const Shape& W, Shape* Z) noexcept {
  if (!W.is_rank(2)) {
    DXERROR("Invalid W: rank of W %d must be 2.", W.rank());
    return false;
  }
  Z->resize(Xrow, W[1]);
  return true;
}

template <typename T, typename I>
void EmbeddingLookup(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                     Tensor<T>* Z) noexcept {
  LLSparseTensor<T, I>::gesmm_mod(X, W, 0, Z);
}

template <typename T, typename I>
void EmbeddingLookupBackward(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                             const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                             SparseRowMatrix<T, I>* gW) noexcept {
  LLSparseTensor<T, I>::gestmm_mod(W.dim(0), X, gZ, 1, gW);
}

template <typename T, typename I>
void SparseEmbeddingLookup(const CSRMatrix<T, I>& X,
                           const SparseRowMatrix<T, I>& W,
                           Tensor<T>* Z) noexcept {
  LLSparseTensor<T, I>::gesmsm(X, W, 0, Z);
}

template <typename T, typename I>
void SparseEmbeddingLookupBackward(const CSRMatrix<T, I>& X,
                                   const SparseRowMatrix<T, I>& /*W*/,
                                   const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                                   SparseRowMatrix<T, I>* gW) noexcept {
  LLSparseTensor<T, I>::gestmm(X, gZ, 1, gW);
}

}  // namespace

EmbeddingLookupNode::EmbeddingLookupNode(std::string name, GraphNode* X,
                                         GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->node_type() == GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_CSR);
  DXCHECK_THROW(W->tensor_type() == TENSOR_TYPE_TSR ||
                W->tensor_type() == TENSOR_TYPE_SRM);
  input_ = {X, W};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (X->shape().is_rank(2) && !W->shape().empty()) {
    (void)EmbeddingLookupInferShape(X->shape()[0], W->shape(), &shape_);
  }
}

class EmbeddingLookupOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Wnode_ = nullptr;
  int W_node_type_ = GRAPH_NODE_TYPE_NONE;
  int W_tensor_type_ = TENSOR_TYPE_NONE;
  const csr_t* X_ = nullptr;
  const tsr_t* Wtsr_ = nullptr;
  const srm_t* Wsrm_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  srm_t* gW_ = nullptr;

 public:
  DEFINE_OP_LIKE(EmbeddingLookupOp);

  void InitForward() override {
    Xnode_ = node_->input(0);
    DXCHECK_THROW(!Xnode_->need_grad());
    Wnode_ = node_->input(1);
    W_node_type_ = Wnode_->node_type();
    W_tensor_type_ = Wnode_->tensor_type();
    X_ = GetPtrCSR(Xnode_);
    Wtsr_ = nullptr;
    Wsrm_ = nullptr;
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        Wtsr_ = GetPtrTSR(Wnode_);
        DXCHECK_THROW(
            EmbeddingLookupInferShape(X_->row(), Wtsr_->shape(), &Zshape_));
        break;
      case TENSOR_TYPE_SRM:
        Wsrm_ = GetPtrSRM(Wnode_);
        DXCHECK_THROW(
            EmbeddingLookupInferShape(X_->row(), Wsrm_->shape(), &Zshape_));
        break;
    }
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        gW_ = InitGradSRM(Wnode_, Wtsr_->shape());
        break;
      case TENSOR_TYPE_SRM:
        gW_ = InitGradSRM(Wnode_, Wsrm_->shape());
        break;
    }
  }

  void Forward() override {
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        EmbeddingLookup(*X_, *Wtsr_, Z_);
        break;
      case TENSOR_TYPE_SRM:
        SparseEmbeddingLookup(*X_, *Wsrm_, Z_);
        break;
    }
  }

  void Backward() override {
    if (gW_) {
      switch (W_tensor_type_) {
        case TENSOR_TYPE_TSR:
          EmbeddingLookupBackward(*X_, *Wtsr_, *Z_, *gZ_, gW_);
          break;
        case TENSOR_TYPE_SRM:
          SparseEmbeddingLookupBackward(*X_, *Wsrm_, *Z_, *gZ_, gW_);
          break;
      }
    }
  }

  void GetPullRequest(PullRequest* pull_request) const override {
    if (W_node_type_ != GRAPH_NODE_TYPE_PARAM) {
      return;
    }

    const std::string& Wname = Wnode_->name();
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        pull_request->tsr_set.emplace(Wname);
        break;
      case TENSOR_TYPE_SRM:
        pull_request->srm_map[Wname].insert(X_->col_begin(), X_->col_end());
        break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(EmbeddingLookup);

}  // namespace deepx_core
