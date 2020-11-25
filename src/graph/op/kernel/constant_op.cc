// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

/************************************************************************/
/* Constant */
/************************************************************************/
ConstantNode::ConstantNode(std::string name, const Shape& shape, double value)
    : GraphNode(std::move(name)),
      constant_type_(CONSTANT_TYPE_VALUE),
      value_(value) {
  // 'shape' must be valid.
  DXCHECK_THROW(shape.rank() > 0);
  DXCHECK_THROW(shape.total_dim() > 0);
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;
  shape_ = shape;
  need_grad_ = 0;
}

ConstantNode::ConstantNode(std::string name, const Shape& shape,
                           std::vector<double> values)
    : GraphNode(std::move(name)),
      constant_type_(CONSTANT_TYPE_VALUES),
      values_(std::move(values)) {
  // 'shape' must be valid.
  DXCHECK_THROW(shape.rank() > 0);
  DXCHECK_THROW(shape.total_dim() > 0);
  DXCHECK_THROW(shape.total_dim() == (int)values_.size());
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;
  shape_ = shape;
  need_grad_ = 0;
}

ConstantNode::ConstantNode(std::string name, const Shape& shape,
                           int initializer_type, double initializer_param1,
                           double initializer_param2)
    : GraphNode(std::move(name)), constant_type_(CONSTANT_TYPE_INITIALIZER) {
  // 'shape' must be valid.
  DXCHECK_THROW(shape.rank() > 0);
  DXCHECK_THROW(shape.total_dim() > 0);
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;
  shape_ = shape;
  initializer_type_ = initializer_type;
  initializer_param1_ = initializer_param1;
  initializer_param2_ = initializer_param2;
  need_grad_ = 0;
}

class ConstantOp : public OpImpl {
 private:
  tsr_t* Z_ = nullptr;

 public:
  DEFINE_OP_LIKE(ConstantOp);

  void InitForward() override {
    const ConstantNode* node = (const ConstantNode*)node_;  // NOLINT
    Z_ = InitHiddenTSR(node_, node_->shape());
    int constant_type = node->constant_type();
    switch (constant_type) {
      case ConstantNode::CONSTANT_TYPE_VALUE: {
        auto value = (float_t)node->value();
        Z_->constant(value);
      } break;
      case ConstantNode::CONSTANT_TYPE_VALUES: {
        const auto& values = node->values();
        for (int i = 0; i < Z_->total_dim(); ++i) {
          Z_->data(i) = (float_t)values[i];
        }
      } break;
    }
  }

  void Forward() override {
    const ConstantNode* node = (const ConstantNode*)node_;  // NOLINT
    int constant_type = node->constant_type();
    switch (constant_type) {
      case ConstantNode::CONSTANT_TYPE_INITIALIZER:
        Z_->rand_init(hidden_->engine(), node->initializer_type(),
                      (float_t)node->initializer_param1(),
                      (float_t)node->initializer_param2());
        break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(Constant);

/************************************************************************/
/* Zeros */
/************************************************************************/
ZerosNode::ZerosNode(std::string name, const Shape& shape)
    : ConstantNode(std::move(name), shape, 0) {}

class ZerosOp : public ConstantOp {
 public:
  DEFINE_OP_LIKE(ZerosOp);
};

GRAPH_NODE_OP_REGISTER(Zeros);

/************************************************************************/
/* Ones */
/************************************************************************/
OnesNode::OnesNode(std::string name, const Shape& shape)
    : ConstantNode(std::move(name), shape, 1) {}

class OnesOp : public ConstantOp {
 public:
  DEFINE_OP_LIKE(OnesOp);
};

GRAPH_NODE_OP_REGISTER(Ones);

/************************************************************************/
/* RandomNormal */
/************************************************************************/
RandomNormalNode::RandomNormalNode(std::string name, const Shape& shape,
                                   double mean, double stddev)
    : ConstantNode(std::move(name), shape, TENSOR_INITIALIZER_TYPE_RANDN, mean,
                   stddev) {}

class RandomNormalOp : public ConstantOp {
 public:
  DEFINE_OP_LIKE(RandomNormalOp);
};

GRAPH_NODE_OP_REGISTER(RandomNormal);

/************************************************************************/
/* RandomUniform */
/************************************************************************/
RandomUniformNode::RandomUniformNode(std::string name, const Shape& shape,
                                     double _min, double _max)
    : ConstantNode(std::move(name), shape, TENSOR_INITIALIZER_TYPE_RAND, _min,
                   _max) {}

class RandomUniformOp : public ConstantOp {
 public:
  DEFINE_OP_LIKE(RandomUniformOp);
};

GRAPH_NODE_OP_REGISTER(RandomUniform);

/************************************************************************/
/* ConstantLike */
/************************************************************************/
ConstantLikeNode::ConstantLikeNode(std::string name, GraphNode* X, double value)
    : GraphNodeUnaryElementWiseBase(std::move(name), X),
      constant_type_(CONSTANT_TYPE_VALUE),
      value_(value) {}

ConstantLikeNode::ConstantLikeNode(std::string name, GraphNode* X,
                                   int initializer_type,
                                   double initializer_param1,
                                   double initializer_param2)
    : GraphNodeUnaryElementWiseBase(std::move(name), X),
      constant_type_(CONSTANT_TYPE_INITIALIZER) {
  initializer_type_ = initializer_type;
  initializer_param1_ = initializer_param1;
  initializer_param2_ = initializer_param2;
}

class ConstantLikeOp : public OpUnaryElementWiseBase {
 public:
  DEFINE_OP_LIKE(ConstantLikeOp);

  void InitForward() override {
    OpUnaryElementWiseBase::InitForward();
    const ConstantLikeNode* node = (const ConstantLikeNode*)node_;  // NOLINT
    int constant_type = node->constant_type();
    switch (constant_type) {
      case ConstantLikeNode::CONSTANT_TYPE_VALUE: {
        auto value = (float_t)node->value();
        Z_->constant(value);
      } break;
    }
  }

  void Forward() override {
    const ConstantLikeNode* node = (const ConstantLikeNode*)node_;  // NOLINT
    int constant_type = node->constant_type();
    switch (constant_type) {
      case ConstantLikeNode::CONSTANT_TYPE_INITIALIZER:
        Z_->rand_init(hidden_->engine(), node->initializer_type(),
                      (float_t)node->initializer_param1(),
                      (float_t)node->initializer_param2());
        break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(ConstantLike);

/************************************************************************/
/* ZerosLike */
/************************************************************************/
ZerosLikeNode::ZerosLikeNode(std::string name, GraphNode* X)
    : ConstantLikeNode(std::move(name), X, 0) {}

class ZerosLikeOp : public ConstantLikeOp {
 public:
  DEFINE_OP_LIKE(ZerosLikeOp);
};

GRAPH_NODE_OP_REGISTER(ZerosLike);

/************************************************************************/
/* OnesLike */
/************************************************************************/
OnesLikeNode::OnesLikeNode(std::string name, GraphNode* X)
    : ConstantLikeNode(std::move(name), X, 1) {}

class OnesLikeOp : public ConstantLikeOp {
 public:
  DEFINE_OP_LIKE(OnesLikeOp);
};

GRAPH_NODE_OP_REGISTER(OnesLike);

/************************************************************************/
/* RandomNormalLike */
/************************************************************************/
RandomNormalLikeNode::RandomNormalLikeNode(std::string name, GraphNode* X,
                                           double mean, double stddev)
    : ConstantLikeNode(std::move(name), X, TENSOR_INITIALIZER_TYPE_RANDN, mean,
                       stddev) {}

class RandomNormalLikeOp : public ConstantLikeOp {
 public:
  DEFINE_OP_LIKE(RandomNormalLikeOp);
};

GRAPH_NODE_OP_REGISTER(RandomNormalLike);

/************************************************************************/
/* RandomUniformLike */
/************************************************************************/
RandomUniformLikeNode::RandomUniformLikeNode(std::string name, GraphNode* X,
                                             double _min, double _max)
    : ConstantLikeNode(std::move(name), X, TENSOR_INITIALIZER_TYPE_RAND, _min,
                       _max) {}

class RandomUniformLikeOp : public ConstantLikeOp {
 public:
  DEFINE_OP_LIKE(RandomUniformLikeOp);
};

GRAPH_NODE_OP_REGISTER(RandomUniformLike);

}  // namespace deepx_core
