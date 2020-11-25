// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

bool TFEmbeddingLookupInferShape(const Shape& X, const Shape& W,
                                 Shape* Z) noexcept {
  int Xrank = X.rank();
  if (Xrank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  if (!W.is_rank(2)) {
    DXERROR("Invalid W: rank of W %d must be 2.", W.rank());
    return false;
  }

  int Zrank = Xrank + 1;
  if (Zrank > SHAPE_MAX_RANK) {
    DXERROR("Rank of output %d is too large.", Zrank);
    return false;
  }

  int Zdims[SHAPE_MAX_RANK];
  for (int i = 0; i < Xrank; ++i) {
    Zdims[i] = X[i];
  }
  Zdims[Xrank] = W[1];
  Z->assign(&Zdims[0], &Zdims[Zrank]);
  return true;
}

template <typename T, typename I>
void TFEmbeddingLookup(const Tensor<I>& X, const Tensor<T>& W,
                       Tensor<T>* Z) noexcept {
  int Wrow = W.dim(0);
  int Wcol = W.dim(1);
  const I* _X = X.data();
  const T* _W = W.data();
  T* _Z = Z->data();

  if (Wcol == 1) {
    for (int i = 0; i < X.total_dim(); ++i) {
      *_Z = _W[*_X % Wrow];
      _X += 1;
      _Z += 1;
    }
  } else {
    const T* Wj;
    for (int i = 0; i < X.total_dim(); ++i) {
      Wj = _W + (*_X % Wrow) * Wcol;
      LLMath<T>::copy(Wcol, Wj, _Z);
      _X += 1;
      _Z += Wcol;
    }
  }
}

template <typename T, typename I>
void TFEmbeddingLookupBackward(const Tensor<I>& X, const Tensor<T>& W,
                               const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                               SparseRowMatrix<T, I>* gW) noexcept {
  int Wrow = W.dim(0);
  int Wcol = W.dim(1);
  const I* _X = X.data();
  const T* _gZ = gZ.data();

  if (Wcol == 1) {
    for (int i = 0; i < X.total_dim(); ++i) {
      gW->get_scalar_no_init(*_X % Wrow) += *_gZ;
      _X += 1;
      _gZ += 1;
    }
  } else {
    T* gWj;
    for (int i = 0; i < X.total_dim(); ++i) {
      gWj = gW->get_row_no_init(*_X % Wrow);
      LLMath<T>::add(Wcol, _gZ, gWj, gWj);
      _X += 1;
      _gZ += Wcol;
    }
  }
}

template <typename T, typename I>
void TFEmbeddingLookup(const Tensor<I>& X, const SparseRowMatrix<T, I>& W,
                       Tensor<T>* Z) noexcept {
  int Wcol = W.col();
  const I* _X = X.data();
  T* _Z = Z->data();

  if (Wcol == 1) {
    for (int i = 0; i < X.total_dim(); ++i) {
      *_Z = W.get_scalar_no_init(*_X);
      _X += 1;
      _Z += 1;
    }
  } else {
    const T* Wj;
    for (int i = 0; i < X.total_dim(); ++i) {
      Wj = W.get_row_no_init(*_X);
      if (Wj) {
        LLMath<T>::copy(Wcol, Wj, _Z);
      } else {
        LLMath<T>::zero(Wcol, _Z);
      }
      _X += 1;
      _Z += Wcol;
    }
  }
}

template <typename T, typename I>
void TFEmbeddingLookupBackward(const Tensor<I>& X,
                               const SparseRowMatrix<T, I>& W,
                               const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                               SparseRowMatrix<T, I>* gW) noexcept {
  int Wcol = W.col();
  const I* _X = X.data();
  const T* _gZ = gZ.data();

  if (Wcol == 1) {
    for (int i = 0; i < X.total_dim(); ++i) {
      gW->get_scalar_no_init(*_X) += *_gZ;
      _X += 1;
      _gZ += 1;
    }
  } else {
    T* gWj;
    for (int i = 0; i < X.total_dim(); ++i) {
      gWj = gW->get_row_no_init(*_X);
      LLMath<T>::add(Wcol, _gZ, gWj, gWj);
      _X += 1;
      _gZ += Wcol;
    }
  }
}

}  // namespace

TFEmbeddingLookupNode::TFEmbeddingLookupNode(std::string name, GraphNode* X,
                                             GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->node_type() == GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSRI);
  DXCHECK_THROW(W->tensor_type() == TENSOR_TYPE_TSR ||
                W->tensor_type() == TENSOR_TYPE_SRM);
  input_ = {X, W};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (!X->shape().empty() && !W->shape().empty()) {
    (void)TFEmbeddingLookupInferShape(X->shape(), W->shape(), &shape_);
  }
}

class TFEmbeddingLookupOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Wnode_ = nullptr;
  int W_node_type_ = GRAPH_NODE_TYPE_NONE;
  int W_tensor_type_ = TENSOR_TYPE_NONE;
  const tsri_t* X_ = nullptr;
  const tsr_t* Wtsr_ = nullptr;
  const srm_t* Wsrm_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  srm_t* gW_ = nullptr;

 public:
  DEFINE_OP_LIKE(TFEmbeddingLookupOp);

  void InitForward() override {
    Xnode_ = node_->input(0);
    DXCHECK_THROW(!Xnode_->need_grad());
    Wnode_ = node_->input(1);
    W_node_type_ = Wnode_->node_type();
    W_tensor_type_ = Wnode_->tensor_type();
    X_ = GetPtrTSRI(Xnode_);
    Wtsr_ = nullptr;
    Wsrm_ = nullptr;
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        Wtsr_ = GetPtrTSR(Wnode_);
        DXCHECK_THROW(
            TFEmbeddingLookupInferShape(X_->shape(), Wtsr_->shape(), &Zshape_));
        break;
      case TENSOR_TYPE_SRM:
        Wsrm_ = GetPtrSRM(Wnode_);
        DXCHECK_THROW(
            TFEmbeddingLookupInferShape(X_->shape(), Wsrm_->shape(), &Zshape_));
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
        TFEmbeddingLookup(*X_, *Wtsr_, Z_);
        break;
      case TENSOR_TYPE_SRM:
        TFEmbeddingLookup(*X_, *Wsrm_, Z_);
        break;
    }
  }

  void Backward() override {
    if (gW_) {
      switch (W_tensor_type_) {
        case TENSOR_TYPE_TSR:
          TFEmbeddingLookupBackward(*X_, *Wtsr_, *Z_, *gZ_, gW_);
          break;
        case TENSOR_TYPE_SRM:
          TFEmbeddingLookupBackward(*X_, *Wsrm_, *Z_, *gZ_, gW_);
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
        pull_request->srm_map[Wname].insert(X_->begin(), X_->end());
        break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(TFEmbeddingLookup);

}  // namespace deepx_core
