// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
#include <sage2/sgemm.h>
#endif

namespace deepx_core {
namespace {

bool FullyConnectInferShape(const Shape& X, const Shape& W, Shape* Z) noexcept {
  if (!X.is_rank(2)) {
    DXERROR("Invalid X: rank of X %d must be 2.", X.rank());
    return false;
  }

  if (!W.is_rank(2)) {
    DXERROR("Invalid W: rank of W %d must be 2.", W.rank());
    return false;
  }

  if (X[1] != W[0]) {
    DXERROR("Invalid X and W: inconsistent dim %d vs %d.", X[1], W[0]);
    return false;
  }

  Z->resize(X[0], W[1]);
  return true;
}

bool FullyConnectInferShape(const Shape& X, const Shape& W, const Shape& b,
                            Shape* Z) noexcept {
  if (!FullyConnectInferShape(X, W, Z)) {
    return false;
  }

  Shape expected(1, W[1]);
  if (b != expected) {
    DXERROR("Invalid b: inconsistent shape %s vs %s.", to_string(b).c_str(),
            to_string(expected).c_str());
    return false;
  }
  return true;
}

}  // namespace

FullyConnectNode::FullyConnectNode(std::string name, GraphNode* X, GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(W->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {X, W};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (!X->shape().empty() && !W->shape().empty()) {
    (void)FullyConnectInferShape(X->shape(), W->shape(), &shape_);
  }
}

FullyConnectNode::FullyConnectNode(std::string name, GraphNode* X, GraphNode* W,
                                   GraphNode* b)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(W->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(b->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {X, W, b};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (!X->shape().empty() && !W->shape().empty() && !b->shape().empty()) {
    (void)FullyConnectInferShape(X->shape(), W->shape(), b->shape(), &shape_);
  }
}

class FullyConnectOp : public OpImpl {
 protected:
  const tsr_t* X_ = nullptr;
  const tsr_t* W_ = nullptr;
  const tsr_t* b_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* gW_ = nullptr;
  tsr_t* gb_ = nullptr;
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
  int m_ = 0;
  int n_ = 0;
  int k_ = 0;
  void* forward_jit_ = nullptr;
  void* backward_gX_jit_ = nullptr;
  void* backward_gW_jit_ = nullptr;
  sage2_sgemm_t forward_ = nullptr;
  sage2_sgemm_t backward_gX_ = nullptr;
  sage2_sgemm_t backward_gW_ = nullptr;

 public:
  ~FullyConnectOp() override {
    if (forward_jit_) {
      sage2_sgemm_jit_uninit(forward_jit_);
    }
    if (backward_gX_jit_) {
      sage2_sgemm_jit_uninit(backward_gX_jit_);
    }
    if (backward_gW_jit_) {
      sage2_sgemm_jit_uninit(backward_gW_jit_);
    }
  }
#endif

 public:
  DEFINE_OP_LIKE(FullyConnectOp);

  void InitForward() override {
    X_ = GetPtrTSR(node_->input(0));
    W_ = GetPtrTSR(node_->input(1));
    if (node_->input_size() == 2) {
      b_ = nullptr;
      DXCHECK_THROW(FullyConnectInferShape(X_->shape(), W_->shape(), &Zshape_));
    } else {
      b_ = GetPtrTSR(node_->input(2));
      DXCHECK_THROW(FullyConnectInferShape(X_->shape(), W_->shape(),
                                           b_->shape(), &Zshape_));
    }
    Z_ = InitHiddenTSR(node_, Zshape_);
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
    m_ = X_->shape()[0];
    n_ = W_->shape()[1];
    k_ = X_->shape()[1];
    if (forward_jit_) {
      sage2_sgemm_jit_uninit(forward_jit_);
    }
    forward_jit_ =
        sage2_sgemm_jit_init(101, 111, 111, m_, n_, k_, 1, k_, n_, 0, n_);
    DXASSERT(forward_jit_);
    forward_ = sage2_sgemm_jit_get(forward_jit_);
    DXASSERT(forward_jit_);
#endif
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(node_->input(0), X_->shape());
    gW_ = InitGradTSR(node_->input(1), W_->shape());
    if (b_) {
      gb_ = InitGradTSR(node_->input(2), b_->shape());
    } else {
      gb_ = nullptr;
    }
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
    if (gX_) {
      if (backward_gX_jit_) {
        sage2_sgemm_jit_uninit(backward_gX_jit_);
      }
      backward_gX_jit_ =
          sage2_sgemm_jit_init(101, 111, 112, m_, k_, n_, 1, n_, n_, 1, k_);
      DXASSERT(backward_gX_jit_);
      backward_gX_ = sage2_sgemm_jit_get(backward_gX_jit_);
      DXASSERT(backward_gX_);
    }
    if (gW_) {
      if (backward_gW_jit_) {
        sage2_sgemm_jit_uninit(backward_gW_jit_);
      }
      backward_gW_jit_ =
          sage2_sgemm_jit_init(101, 112, 111, k_, n_, m_, 1, k_, n_, 1, n_);
      DXASSERT(backward_gW_jit_);
      backward_gW_ = sage2_sgemm_jit_get(backward_gW_jit_);
      DXASSERT(backward_gW_);
    }
#endif
  }

  void Forward() override {
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
    forward_(forward_jit_, X_->data(), W_->data(), Z_->data());
#else
    ll_tensor_t::gemm(0, 0, *X_, *W_, Z_);
#endif
    if (b_) {
      ll_tensor_t::add_row(1, *Z_, 1, *b_, Z_);
    }
  }

  void Backward() override {
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
    if (gX_) {
      backward_gX_(backward_gX_jit_, gZ_->data(), W_->data(), gX_->data());
    }
    if (gW_) {
      backward_gW_(backward_gW_jit_, X_->data(), gZ_->data(), gW_->data());
    }
#else
    if (gX_) {
      ll_tensor_t::gemm(0, 1, 1, *gZ_, *W_, 1, gX_);
    }
    if (gW_) {
      ll_tensor_t::gemm(1, 0, 1, *X_, *gZ_, 1, gW_);
    }
#endif
    if (gb_) {
      ll_tensor_t::sum_row(1, *gZ_, 1, gb_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(FullyConnect);

}  // namespace deepx_core
