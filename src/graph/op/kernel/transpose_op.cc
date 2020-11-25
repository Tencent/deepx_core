// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#include "transpose.h"

namespace deepx_core {

TransposeNode::TransposeNode(std::string name, GraphNode* X, const Shape& axes)
    : GraphNodeUnaryBase(std::move(name), X), axes_(axes) {
  if (!X->shape().empty()) {
    (void)TransposeInferShape(X->shape(), axes_, &shape_);
  }
}

class TransposeOp : public OpUnaryBase {
 private:
  TransposeAux aux_;

 public:
  DEFINE_OP_LIKE(TransposeOp);

  const Shape& InferShape() override {
    const Shape& axes = ((const TransposeNode*)node_)->axes();
    DXCHECK_THROW(TransposePrepare(X_->shape(), axes, &aux_));
    return aux_.Z;
  }

  void Forward() override { Transpose(*X_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      TransposeBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Transpose);

}  // namespace deepx_core
