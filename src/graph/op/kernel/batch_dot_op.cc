// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

bool BatchDotInferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
  if (!X.is_rank(2)) {
    DXERROR("Invalid X: rank of X %d must be 2.", X.rank());
    return false;
  }

  if (X != Y) {
    DXERROR("Invalid X and Y: inconsistent shape %s vs %s.",
            to_string(X).c_str(), to_string(Y).c_str());
    return false;
  }

  int batch = X[0];
  Z->resize(batch, 1);
  return true;
}

template <typename T>
void BatchDot(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  for (int i = 0; i < batch; ++i) {
    *_Z = LLMath<T>::dot(m, _X, _Y);
    _X += m;
    _Y += m;
    _Z += 1;
  }
}

template <typename T>
void BatchDotBackward(const Tensor<T>& X, const Tensor<T>& Y,
                      const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                      Tensor<T>* gX, Tensor<T>* gY) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);

  auto update_x =
      [ batch, m, &gZ ](const Tensor<T>& Y, Tensor<T>* gX) noexcept {
    const T* _Y = Y.data();
    const T* _gZ = gZ.data();
    T* _gX = gX->data();
    for (int i = 0; i < batch; ++i) {
      LLMath<T>::axpy(m, *_gZ, _Y, _gX);
      _gZ += 1;
      _Y += m;
      _gX += m;
    }
  };

  if (gX) {
    update_x(Y, gX);
  }

  if (gY) {
    update_x(X, gY);
  }
}

}  // namespace

BatchDotNode::BatchDotNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchDotInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BatchDotOp : public OpBinaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(BatchDotOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(BatchDotInferShape(X_->shape(), Y_->shape(), &Zshape_));
    return Zshape_;
  }

  void Forward() override { BatchDot(*X_, *Y_, Z_); }

  void Backward() override {
    if (gZ_) {
      BatchDotBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchDot);

}  // namespace deepx_core
