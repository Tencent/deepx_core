// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

bool SequenceMaskInferShape(const Shape& X, int max_size, Shape* Z) noexcept {
  if (!X.is_rank(1)) {
    DXERROR("Invalid X: rank of X %d must be 1.", X.rank());
    return false;
  }

  if (max_size <= 0) {
    DXERROR("Invalid max_size: max_size %d must be positive.", max_size);
    return false;
  }

  Z->resize(X[0], max_size);
  return true;
}

template <typename T>
void SequenceMask(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  const T* _X = X.data();
  T* _Z = Z->data();
  int batch = Z->dim(0);
  int m = Z->dim(1);
  int Xi;
  Z->zeros();
  for (int i = 0; i < batch; ++i) {
    Xi = (int)*_X;
    if (Xi > 0) {
      if (Xi > m) {
        Xi = m;
      }
      for (int j = 0; j < Xi; ++j) {
        _Z[j] = 1;
      }
    }
    _X += 1;
    _Z += m;
  }
}

}  // namespace

SequenceMaskNode::SequenceMaskNode(std::string name, GraphNode* X, int max_size)
    : GraphNodeUnaryBase(std::move(name), X), max_size_(max_size) {
  if (!X->shape().empty()) {
    (void)SequenceMaskInferShape(X->shape(), max_size_, &shape_);
  }
}

class SequenceMaskOp : public OpUnaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(SequenceMaskOp);

  const Shape& InferShape() override {
    int max_size = ((const SequenceMaskNode*)node_)->max_size();
    DXCHECK_THROW(SequenceMaskInferShape(X_->shape(), max_size, &Zshape_));
    return Zshape_;
  }

  void Forward() override { SequenceMask(*X_, Z_); }
};

GRAPH_NODE_OP_REGISTER(SequenceMask);

}  // namespace deepx_core
