// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

/************************************************************************/
/* Reshape */
/************************************************************************/
ReshapeNode::ReshapeNode(std::string name, GraphNode* X, const Shape& shape)
    : GraphNodeUnaryBase(std::move(name), X) {
  shape_ = shape;
}

class ReshapeOp : public OpUnaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(ReshapeOp);

  const Shape& InferShape() override {
    Zshape_ = X_->shape();
    Zshape_.reshape(node_->shape());
    return Zshape_;
  }

  void Forward() override { Z_->set_data(*X_); }

  void Backward() override {
    if (gX_) {
      const float_t* gZ = gZ_->data();
      float_t* gX = gX_->data();
      ll_math_t::add(gX_->total_dim(), gX, gZ, gX);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Reshape);

/************************************************************************/
/* ReshapeFast */
/************************************************************************/
ReshapeFastNode::ReshapeFastNode(std::string name, GraphNode* X,
                                 const Shape& shape)
    : GraphNodeUnaryBase(std::move(name), X) {
  shape_ = shape;
}

class ReshapeFastOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  tsr_t* X_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;

 public:
  DEFINE_OP_LIKE(ReshapeFastOp);

  void InitForward() override {
    DXCHECK_THROW(node_->input_fork() == 0);
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    Zshape_ = X_->shape();
    Zshape_.reshape(node_->shape());
    Z_ = InitHiddenTSRView(node_);
    Z_->view(Zshape_, X_->data());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSRView(Xnode_);
    if (gX_) {
      gX_->view(X_->shape(), gZ_->data());
    }
  }
};

GRAPH_NODE_OP_REGISTER(ReshapeFast);

/************************************************************************/
/* Reshape2 */
/************************************************************************/
Reshape2Node::Reshape2Node(std::string name, GraphNode* X, const Shape& shape)
    : GraphNodeUnaryBase(std::move(name), X), new_shape_(shape) {
  shape_ = X->shape();
  shape_.reshape_nothrow(new_shape_);
}

class Reshape2Op : public OpUnaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(Reshape2Op);

  const Shape& InferShape() override {
    Zshape_ = X_->shape();
    Zshape_.reshape(((const Reshape2Node*)node_)->new_shape());
    return Zshape_;
  }

  void Forward() override { Z_->set_data(*X_); }

  void Backward() override {
    if (gX_) {
      const float_t* gZ = gZ_->data();
      float_t* gX = gX_->data();
      ll_math_t::add(gX_->total_dim(), gX, gZ, gX);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Reshape2);

/************************************************************************/
/* Reshape2Fast */
/************************************************************************/
Reshape2FastNode::Reshape2FastNode(std::string name, GraphNode* X,
                                   const Shape& shape)
    : GraphNodeUnaryBase(std::move(name), X), new_shape_(shape) {
  shape_ = X->shape();
  shape_.reshape_nothrow(new_shape_);
}

class Reshape2FastOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  tsr_t* X_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;

 public:
  DEFINE_OP_LIKE(Reshape2FastOp);

  void InitForward() override {
    DXCHECK_THROW(node_->input_fork() == 0);
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    Zshape_ = X_->shape();
    Zshape_.reshape(((const Reshape2FastNode*)node_)->new_shape());
    Z_ = InitHiddenTSRView(node_);
    Z_->view(Zshape_, X_->data());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSRView(Xnode_);
    if (gX_) {
      gX_->view(X_->shape(), gZ_->data());
    }
  }
};

GRAPH_NODE_OP_REGISTER(Reshape2Fast);

/************************************************************************/
/* ExpandDim */
/************************************************************************/
ExpandDimNode::ExpandDimNode(std::string name, GraphNode* X, int axis)
    : GraphNodeUnaryBase(std::move(name), X), axis_(axis) {
  shape_ = X->shape();
  shape_.expand_dim_nothrow(axis_);
}

class ExpandDimOp : public OpUnaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(ExpandDimOp);

  const Shape& InferShape() override {
    Zshape_ = X_->shape();
    Zshape_.expand_dim(((const ExpandDimNode*)node_)->axis());
    return Zshape_;
  }

  void Forward() override { Z_->set_data(*X_); }

  void Backward() override {
    if (gX_) {
      const float_t* gZ = gZ_->data();
      float_t* gX = gX_->data();
      ll_math_t::add(gX_->total_dim(), gX, gZ, gX);
    }
  }
};

GRAPH_NODE_OP_REGISTER(ExpandDim);

/************************************************************************/
/* ExpandDimFast */
/************************************************************************/
ExpandDimFastNode::ExpandDimFastNode(std::string name, GraphNode* X, int axis)
    : GraphNodeUnaryBase(std::move(name), X), axis_(axis) {
  shape_ = X->shape();
  shape_.expand_dim_nothrow(axis_);
}

class ExpandDimFastOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  tsr_t* X_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;

 public:
  DEFINE_OP_LIKE(ExpandDimFastOp);

  void InitForward() override {
    DXCHECK_THROW(node_->input_fork() == 0);
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    Zshape_ = X_->shape();
    Zshape_.expand_dim(((const ExpandDimFastNode*)node_)->axis());
    Z_ = InitHiddenTSRView(node_);
    Z_->view(Zshape_, X_->data());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSRView(Xnode_);
    if (gX_) {
      gX_->view(X_->shape(), gZ_->data());
    }
  }
};

GRAPH_NODE_OP_REGISTER(ExpandDimFast);

/************************************************************************/
/* Squeeze */
/************************************************************************/
SqueezeNode::SqueezeNode(std::string name, GraphNode* X, int axis)
    : GraphNodeUnaryBase(std::move(name), X), axis_(axis) {
  shape_ = X->shape();
  shape_.squeeze_nothrow(axis_);
}

class SqueezeOp : public OpUnaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(SqueezeOp);

  const Shape& InferShape() override {
    Zshape_ = X_->shape();
    Zshape_.squeeze(((const SqueezeNode*)node_)->axis());
    DXCHECK_THROW(!Zshape_.empty());
    return Zshape_;
  }

  void Forward() override { Z_->set_data(*X_); }

  void Backward() override {
    if (gX_) {
      const float_t* gZ = gZ_->data();
      float_t* gX = gX_->data();
      ll_math_t::add(gX_->total_dim(), gX, gZ, gX);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Squeeze);

/************************************************************************/
/* SqueezeFast */
/************************************************************************/
SqueezeFastNode::SqueezeFastNode(std::string name, GraphNode* X, int axis)
    : GraphNodeUnaryBase(std::move(name), X), axis_(axis) {
  shape_ = X->shape();
  shape_.squeeze_nothrow(axis_);
}

class SqueezeFastOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  tsr_t* X_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;

 public:
  DEFINE_OP_LIKE(SqueezeFastOp);

  void InitForward() override {
    DXCHECK_THROW(node_->input_fork() == 0);
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    Zshape_ = X_->shape();
    Zshape_.squeeze(((const SqueezeFastNode*)node_)->axis());
    DXCHECK_THROW(!Zshape_.empty());
    Z_ = InitHiddenTSRView(node_);
    Z_->view(Zshape_, X_->data());
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSRView(Xnode_);
    if (gX_) {
      gX_->view(X_->shape(), gZ_->data());
    }
  }
};

GRAPH_NODE_OP_REGISTER(SqueezeFast);

}  // namespace deepx_core
