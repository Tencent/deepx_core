// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph_node.h>

namespace deepx_core {

/************************************************************************/
/* GraphNode */
/************************************************************************/
GraphNode::GraphNode(std::string name) noexcept : name_(std::move(name)) {}

bool GraphNode::IsAttrEqual(const GraphNode* other) const noexcept {
  return this->type_index() == other->type_index();
}

void GraphNode::Write(OutputStream& os) {
  input_name_.resize(input_.size());
  for (size_t i = 0; i < input_.size(); ++i) {
    input_name_[i] = input_[i]->name();
  }

  int version = 3;
  os << version;
  os << name_ << node_id_;
  os << input_name_;
  os << node_type_ << tensor_type_ << shape_;
  os << initializer_type_ << initializer_param1_ << initializer_param2_;
  os << need_grad_;
}

void GraphNode::Read(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read graph node.");
    return;
  }

  if (version == 0) {
    DXERROR("Version 0 is deprecated.");
    is.set_bad();
  } else if (version == 1) {
    DXERROR("Version 1 is deprecated.");
    is.set_bad();
  } else if (version == 2) {
    int output_degree;
    is >> name_ >> node_id_;
    is >> input_name_;
    is >> node_type_ >> tensor_type_ >> shape_;
    is >> initializer_type_ >> initializer_param1_ >> initializer_param2_;
    is >> need_grad_;
    is >> output_degree >> input_fork_;
  } else if (version == 3) {
    is >> name_ >> node_id_;
    is >> input_name_;
    is >> node_type_ >> tensor_type_ >> shape_;
    is >> initializer_type_ >> initializer_param1_ >> initializer_param2_;
    is >> need_grad_;
  } else {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
  }

  if (is) {
    // backward compatibility
    if (tensor_type_ == TENSOR_TYPE_SRP || tensor_type_ == TENSOR_TYPE_SVP) {
      tensor_type_ = TENSOR_TYPE_SRM;
    }
  }
}

bool GraphNode::IsValidName() const noexcept {
  if (name_.empty()) {
    return false;
  }

  for (char c : name_) {
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '/' || c == ':')) {
      return false;
    }
  }
  return true;
}

bool GraphNode::HasShape(const std::vector<GraphNode*>& X) noexcept {
  for (const GraphNode* _X : X) {
    if (_X->shape().empty()) {
      return false;
    }
  }
  return true;
}

/************************************************************************/
/* base node */
/************************************************************************/
GraphNodeUnaryBase::GraphNodeUnaryBase(std::string name, GraphNode* X)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {X};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;
}

GraphNodeUnaryElementWiseBase::GraphNodeUnaryElementWiseBase(std::string name,
                                                             GraphNode* X)
    : GraphNodeUnaryBase(std::move(name), X) {
  if (!X->shape().empty()) {
    shape_ = X->shape();
  }
}

GraphNodeBinaryBase::GraphNodeBinaryBase(std::string name, GraphNode* X,
                                         GraphNode* Y)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(Y->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {X, Y};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;
}

GraphNodeBinaryElementWiseBase::GraphNodeBinaryElementWiseBase(std::string name,
                                                               GraphNode* X,
                                                               GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    if (X->shape() == Y->shape()) {
      shape_ = X->shape();
    }
  }
}

}  // namespace deepx_core
