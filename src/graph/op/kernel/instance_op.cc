// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

InstanceNode::InstanceNode(std::string name, const Shape& shape,
                           int tensor_type)
    : GraphNode(std::move(name)) {
  // 'name' must be valid.
  DXCHECK_THROW(IsValidName());
  DXCHECK_THROW(
      tensor_type == TENSOR_TYPE_TSR || tensor_type == TENSOR_TYPE_CSR ||
      tensor_type == TENSOR_TYPE_TSRI || tensor_type == TENSOR_TYPE_TSRS);
  node_type_ = GRAPH_NODE_TYPE_INSTANCE;
  tensor_type_ = tensor_type;
  shape_ = shape;
}

class InstanceOp : public OpImpl {
 public:
  DEFINE_OP_LIKE(InstanceOp);

  void InitForward() override {
    switch (node_->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        auto& tsr = hidden_->mutable_inst()->get<tsr_t>(node_->name());
        InitPtrTSR(node_, &tsr);
      } break;
      case TENSOR_TYPE_CSR: {
        auto& csr = hidden_->mutable_inst()->get<csr_t>(node_->name());
        InitPtrCSR(node_, &csr);
      } break;
      case TENSOR_TYPE_TSRI: {
        auto& tsri = hidden_->mutable_inst()->get<tsri_t>(node_->name());
        InitPtrTSRI(node_, &tsri);
      } break;
      case TENSOR_TYPE_TSRS: {
        auto& tsrs = hidden_->mutable_inst()->get<tsrs_t>(node_->name());
        InitPtrTSRS(node_, &tsrs);
      } break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(Instance);

}  // namespace deepx_core
