// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

VariableNode::VariableNode(std::string name, const Shape& shape,
                           int tensor_type)
    : GraphNode(std::move(name)) {
  // 'name' must be valid.
  DXCHECK_THROW(IsValidName());
  // backward compatibility
  if (tensor_type == TENSOR_TYPE_SRP || tensor_type == TENSOR_TYPE_SVP) {
    tensor_type = TENSOR_TYPE_SRM;
  }
  DXCHECK_THROW(tensor_type == TENSOR_TYPE_TSR ||
                tensor_type == TENSOR_TYPE_SRM);
  // 'shape' must be valid.
  switch (tensor_type) {
    case TENSOR_TYPE_TSR:
      DXCHECK_THROW(shape.rank() > 0);
      DXCHECK_THROW(shape.total_dim() > 0);
      break;
    case TENSOR_TYPE_SRM:
      DXCHECK_THROW(shape.is_rank(2));
      break;
  }
  node_type_ = GRAPH_NODE_TYPE_PARAM;
  tensor_type_ = tensor_type;
  shape_ = shape;
  need_grad_ = 1;
}

VariableNode::VariableNode(std::string name, const Shape& shape,
                           int tensor_type, int initializer_type,
                           double initializer_param1, double initializer_param2)
    : VariableNode(std::move(name), shape, tensor_type) {
  initializer_type_ = initializer_type;
  initializer_param1_ = initializer_param1;
  initializer_param2_ = initializer_param2;
}

VariableNode::VariableNode(std::string name, const Shape& shape)
    : VariableNode(std::move(name), shape, TENSOR_TYPE_TSR) {}

VariableNode::VariableNode(std::string name, const Shape& shape,
                           int initializer_type, double initializer_param1,
                           double initializer_param2)
    : VariableNode(std::move(name), shape, TENSOR_TYPE_TSR, initializer_type,
                   initializer_param1, initializer_param2) {}

class VariableOp : public OpImpl {
 public:
  DEFINE_OP_LIKE(VariableOp);

  void InitForward() override {
    Any& Wany = param_->at(node_->name());
    switch (node_->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        auto& W = Wany.unsafe_to_ref<tsr_t>();
        InitPtrTSR(node_, &W);
      } break;
      case TENSOR_TYPE_SRM: {
        auto& W = Wany.unsafe_to_ref<srm_t>();
        InitPtrSRM(node_, &W);
      } break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(Variable);

}  // namespace deepx_core
