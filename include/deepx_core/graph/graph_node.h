// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/class_factory.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/tensor/shape.h>
#include <deepx_core/tensor/tensor_type.h>
#include <cstdint>
#include <string>
#include <typeindex>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deepx_core {

#define GRAPH_NODE_REGISTER(class_name) \
  CLASS_FACTORY_REGISTER(GraphNode, class_name, #class_name)
#define GRAPH_NODE_NEW(class_name) CLASS_FACTORY_NEW(GraphNode, class_name)

#define DEFINE_GRAPH_NODE_LIKE_BASE(clazz_name) \
  clazz_name() = default;                       \
  clazz_name(const clazz_name&) = delete;       \
  clazz_name& operator=(const clazz_name&) = delete

#define DEFINE_GRAPH_NODE_LIKE(clazz_name)                                 \
  clazz_name() = default;                                                  \
  clazz_name(const clazz_name&) = delete;                                  \
  clazz_name& operator=(const clazz_name&) = delete;                       \
  const char* class_name() const noexcept override { return #clazz_name; } \
  std::type_index type_index() const noexcept override {                   \
    return typeid(clazz_name);                                             \
  }

#define _DEFINE_GRAPH_NODE_WRITE_READ(...) \
  void Write(OutputStream& os) override {  \
    GraphNode::Write(os);                  \
    os.WriteObject(__VA_ARGS__);           \
  }                                        \
  void Read(InputStream& is) override {    \
    GraphNode::Read(is);                   \
    is.ReadObject(__VA_ARGS__);            \
  }

#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL1(class_name, a1)            \
  bool IsAttrEqual(const GraphNode* other) const noexcept override { \
    return this->type_index() == other->type_index() &&              \
           this->a1 == ((const class_name*)other)->a1;               \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL2(class_name, a1, a2)        \
  bool IsAttrEqual(const GraphNode* other) const noexcept override { \
    return this->type_index() == other->type_index() &&              \
           this->a1 == ((const class_name*)other)->a1 &&             \
           this->a2 == ((const class_name*)other)->a2;               \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL3(class_name, a1, a2, a3)    \
  bool IsAttrEqual(const GraphNode* other) const noexcept override { \
    return this->type_index() == other->type_index() &&              \
           this->a1 == ((const class_name*)other)->a1 &&             \
           this->a2 == ((const class_name*)other)->a2 &&             \
           this->a3 == ((const class_name*)other)->a3;               \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL4(class_name, a1, a2, a3, a4) \
  bool IsAttrEqual(const GraphNode* other) const noexcept override {  \
    return this->type_index() == other->type_index() &&               \
           this->a1 == ((const class_name*)other)->a1 &&              \
           this->a2 == ((const class_name*)other)->a2 &&              \
           this->a3 == ((const class_name*)other)->a3 &&              \
           this->a4 == ((const class_name*)other)->a4;                \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL5(class_name, a1, a2, a3, a4, a5) \
  bool IsAttrEqual(const GraphNode* other) const noexcept override {      \
    return this->type_index() == other->type_index() &&                   \
           this->a1 == ((const class_name*)other)->a1 &&                  \
           this->a2 == ((const class_name*)other)->a2 &&                  \
           this->a3 == ((const class_name*)other)->a3 &&                  \
           this->a4 == ((const class_name*)other)->a4 &&                  \
           this->a5 == ((const class_name*)other)->a5;                    \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL6(class_name, a1, a2, a3, a4, a5, a6) \
  bool IsAttrEqual(const GraphNode* other) const noexcept override {          \
    return this->type_index() == other->type_index() &&                       \
           this->a1 == ((const class_name*)other)->a1 &&                      \
           this->a2 == ((const class_name*)other)->a2 &&                      \
           this->a3 == ((const class_name*)other)->a3 &&                      \
           this->a4 == ((const class_name*)other)->a4 &&                      \
           this->a5 == ((const class_name*)other)->a5 &&                      \
           this->a6 == ((const class_name*)other)->a6;                        \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL7(class_name, a1, a2, a3, a4, a5, a6, \
                                          a7)                                 \
  bool IsAttrEqual(const GraphNode* other) const noexcept override {          \
    return this->type_index() == other->type_index() &&                       \
           this->a1 == ((const class_name*)other)->a1 &&                      \
           this->a2 == ((const class_name*)other)->a2 &&                      \
           this->a3 == ((const class_name*)other)->a3 &&                      \
           this->a4 == ((const class_name*)other)->a4 &&                      \
           this->a5 == ((const class_name*)other)->a5 &&                      \
           this->a6 == ((const class_name*)other)->a6 &&                      \
           this->a7 == ((const class_name*)other)->a7;                        \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL8(class_name, a1, a2, a3, a4, a5, a6, \
                                          a7, a8)                             \
  bool IsAttrEqual(const GraphNode* other) const noexcept override {          \
    return this->type_index() == other->type_index() &&                       \
           this->a1 == ((const class_name*)other)->a1 &&                      \
           this->a2 == ((const class_name*)other)->a2 &&                      \
           this->a3 == ((const class_name*)other)->a3 &&                      \
           this->a4 == ((const class_name*)other)->a4 &&                      \
           this->a5 == ((const class_name*)other)->a5 &&                      \
           this->a6 == ((const class_name*)other)->a6 &&                      \
           this->a7 == ((const class_name*)other)->a7 &&                      \
           this->a8 == ((const class_name*)other)->a8;                        \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL9(class_name, a1, a2, a3, a4, a5, a6, \
                                          a7, a8, a9)                         \
  bool IsAttrEqual(const GraphNode* other) const noexcept override {          \
    return this->type_index() == other->type_index() &&                       \
           this->a1 == ((const class_name*)other)->a1 &&                      \
           this->a2 == ((const class_name*)other)->a2 &&                      \
           this->a3 == ((const class_name*)other)->a3 &&                      \
           this->a4 == ((const class_name*)other)->a4 &&                      \
           this->a5 == ((const class_name*)other)->a5 &&                      \
           this->a6 == ((const class_name*)other)->a6 &&                      \
           this->a7 == ((const class_name*)other)->a7 &&                      \
           this->a8 == ((const class_name*)other)->a8 &&                      \
           this->a9 == ((const class_name*)other)->a9;                        \
  }
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL10(class_name, a1, a2, a3, a4, a5, a6, \
                                           a7, a8, a9, a10)                    \
  bool IsAttrEqual(const GraphNode* other) const noexcept override {           \
    return this->type_index() == other->type_index() &&                        \
           this->a1 == ((const class_name*)other)->a1 &&                       \
           this->a2 == ((const class_name*)other)->a2 &&                       \
           this->a3 == ((const class_name*)other)->a3 &&                       \
           this->a4 == ((const class_name*)other)->a4 &&                       \
           this->a5 == ((const class_name*)other)->a5 &&                       \
           this->a6 == ((const class_name*)other)->a6 &&                       \
           this->a7 == ((const class_name*)other)->a7 &&                       \
           this->a8 == ((const class_name*)other)->a8 &&                       \
           this->a9 == ((const class_name*)other)->a9 &&                       \
           this->a10 == ((const class_name*)other)->a10;                       \
  }

#define _GRAPH_NODE_CONCAT(x, y) x y
#define _GRAPH_NODE_COUNT_ARGS_IMPL2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
                                     count, ...)                              \
  count
#define _GRAPH_NODE_COUNT_ARGS_IMPL1(args) _GRAPH_NODE_COUNT_ARGS_IMPL2 args
#define _GRAPH_NODE_COUNT_ARGS(...) \
  _GRAPH_NODE_COUNT_ARGS_IMPL1((__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL_IMPL2(count) \
  _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL##count
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL_IMPL1(count) \
  _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL_IMPL2(count)
#define _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL(count) \
  _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL_IMPL1(count)
#define DEFINE_GRAPH_NODE_ATTR(class_name, ...)                              \
  _GRAPH_NODE_CONCAT(                                                        \
      _DEFINE_GRAPH_NODE_IS_ATTR_EQUAL(_GRAPH_NODE_COUNT_ARGS(__VA_ARGS__)), \
      (class_name, __VA_ARGS__))                                             \
  _DEFINE_GRAPH_NODE_WRITE_READ(__VA_ARGS__)

#define DEFINE_GRAPH_NODE_CREATOR(class_name_without_Node)                 \
  template <typename... Args>                                              \
  GraphNode* class_name_without_Node(std::string name, Args&&... args) {   \
    return new class_name_without_Node##Node(std::move(name),              \
                                             std::forward<Args>(args)...); \
  }

/************************************************************************/
/* GRAPH_NODE_TYPE */
/************************************************************************/
enum GRAPH_NODE_TYPE {
  GRAPH_NODE_TYPE_NONE = 0,
  GRAPH_NODE_TYPE_PARAM = 1,
  GRAPH_NODE_TYPE_INSTANCE = 2,
  GRAPH_NODE_TYPE_HIDDEN = 3,
};

class Graph;
class SimpItem;
class GraphNode;

/************************************************************************/
/* GraphNode */
/************************************************************************/
class GraphNode {
 protected:
  // name of current node
  //
  // It must be unique in a graph.
  //
  // Valid node names consist of only letters, digits, underlines and slashes.
  std::string name_;

  // id of current node, set when compiling graph.
  //
  // In a graph, it starts from 0 to the number of nodes - 1, it is unique.
  uint16_t node_id_ = (uint16_t)-1;

  // input nodes of current node
  //
  // One node may have no or many input nodes.
  std::vector<GraphNode*> input_;
  // input node names of current node
  //
  // internal usage only
  mutable std::vector<std::string> input_name_;

  // output nodes of current node
  std::unordered_set<GraphNode*> output_;

  // node type of current node
  //
  // See 'GRAPH_NODE_TYPE'.
  int node_type_ = GRAPH_NODE_TYPE_NONE;

  // tensor type of current node
  //
  // See 'TENSOR_TYPE'.
  int tensor_type_ = TENSOR_TYPE_NONE;

  // shape of current node
  Shape shape_;

  // initialization of current node
  //
  // See 'TENSOR_INITIALIZER_TYPE'.
  int initializer_type_ = TENSOR_INITIALIZER_TYPE_NONE;
  // the 1st additional param of the initializer
  double initializer_param1_ = 0;
  // the 2nd additional param of the initializer
  double initializer_param2_ = 0;

  // whether current node needs grad
  int need_grad_ = 0;

  // whether any of the input nodes of current node fork(output degree > 1)
  int input_fork_ = 0;

  // whether current node is a target node
  int is_target_ = 0;

  friend class Graph;
  friend class SimpItem;

 public:
  const std::string& name() const noexcept { return name_; }

  uint16_t node_id() const noexcept { return node_id_; }

  int input_size() const noexcept { return (int)input_.size(); }
  template <typename Int>
  const GraphNode* input(Int i) const noexcept {
    return input_[(size_t)i];
  }
  const std::vector<GraphNode*>& input() const noexcept { return input_; }

  int output_size() const noexcept { return (int)output_.size(); }
  const std::unordered_set<GraphNode*>& output() const noexcept {
    return output_;
  }

  int node_type() const noexcept { return node_type_; }
  int tensor_type() const noexcept { return tensor_type_; }
  const Shape& shape() const noexcept { return shape_; }

  void set_initializer(int initializer_type, double initializer_param1 = 0,
                       double initializer_param2 = 0) noexcept {
    initializer_type_ = initializer_type;
    initializer_param1_ = initializer_param1;
    initializer_param2_ = initializer_param2;
  }
  int initializer_type() const noexcept { return initializer_type_; }
  double initializer_param1() const noexcept { return initializer_param1_; }
  double initializer_param2() const noexcept { return initializer_param2_; }

  void clear_need_grad() noexcept { need_grad_ = 0; }
  void set_need_grad(int need_grad) noexcept { need_grad_ = need_grad; }
  int need_grad() const noexcept { return need_grad_; }

  int input_fork() const noexcept { return input_fork_; }

  int is_target() const noexcept { return is_target_; }

 public:
  GraphNode() = default;
  explicit GraphNode(std::string name) noexcept;
  virtual ~GraphNode() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual std::type_index type_index() const noexcept = 0;
  virtual bool IsAttrEqual(const GraphNode* other) const noexcept;
  virtual void Write(OutputStream& os);  // NOLINT
  virtual void Read(InputStream& is);    // NOLINT

 public:
  bool IsValidName() const noexcept;

 public:
  static bool HasShape(const std::vector<GraphNode*>& X) noexcept;
};

/************************************************************************/
/* base node */
/************************************************************************/
class GraphNodeUnaryBase : public GraphNode {
 public:
  GraphNodeUnaryBase(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeUnaryBase);
};

class GraphNodeUnaryElementWiseBase : public GraphNodeUnaryBase {
 public:
  GraphNodeUnaryElementWiseBase(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeUnaryElementWiseBase);
};

class GraphNodeBinaryBase : public GraphNode {
 public:
  GraphNodeBinaryBase(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeBinaryBase);
};

class GraphNodeBinaryElementWiseBase : public GraphNodeBinaryBase {
 public:
  GraphNodeBinaryElementWiseBase(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeBinaryElementWiseBase);
};

class GraphNodeBroadcastBase : public GraphNodeBinaryBase {
 public:
  GraphNodeBroadcastBase(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeBroadcastBase);
};

class GraphNodeForAxisBase : public GraphNodeUnaryBase {
 private:
  int reduce_all_ = 0;  // reserved
  int axis_ = 0;
  DEFINE_GRAPH_NODE_ATTR(GraphNodeForAxisBase, reduce_all_, axis_);

 public:
  int axis() const noexcept { return axis_; }

 public:
  GraphNodeForAxisBase(std::string name, GraphNode* X, int axis);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeForAxisBase);
};

class GraphNodeReduceAxisBase : public GraphNodeUnaryBase {
 private:
  int reduce_all_ = 0;
  int axis_ = 0;
  int keep_dim_ = 0;
  DEFINE_GRAPH_NODE_ATTR(GraphNodeReduceAxisBase, reduce_all_, axis_,
                         keep_dim_);

 public:
  int reduce_all() const noexcept { return reduce_all_; }
  int axis() const noexcept { return axis_; }
  int keep_dim() const noexcept { return keep_dim_; }

 public:
  GraphNodeReduceAxisBase(std::string name, GraphNode* X, int axis,
                          int keep_dim);
  GraphNodeReduceAxisBase(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeReduceAxisBase);
};

/************************************************************************/
/* unary element-wise op */
/************************************************************************/
class SigmoidNode : public GraphNodeUnaryElementWiseBase {
 public:
  SigmoidNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(SigmoidNode);
};

class TanhNode : public GraphNodeUnaryElementWiseBase {
 public:
  TanhNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(TanhNode);
};

class ReluNode : public GraphNodeUnaryElementWiseBase {
 public:
  ReluNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ReluNode);
};

class LeakyReluNode : public GraphNodeUnaryElementWiseBase {
 private:
  double alpha_ = 0;
  DEFINE_GRAPH_NODE_ATTR(LeakyReluNode, alpha_);

 public:
  double alpha() const noexcept { return alpha_; }

 public:
  LeakyReluNode(std::string name, GraphNode* X, double alpha);
  DEFINE_GRAPH_NODE_LIKE(LeakyReluNode);
};

class EluNode : public GraphNodeUnaryElementWiseBase {
 private:
  double alpha_ = 0;
  DEFINE_GRAPH_NODE_ATTR(EluNode, alpha_);

 public:
  double alpha() const noexcept { return alpha_; }

 public:
  EluNode(std::string name, GraphNode* X, double alpha);
  DEFINE_GRAPH_NODE_LIKE(EluNode);
};

class SeluNode : public GraphNodeUnaryElementWiseBase {
 private:
  double lambda_ = 0;
  double alpha_ = 0;
  DEFINE_GRAPH_NODE_ATTR(SeluNode, lambda_, alpha_);

 public:
  double lambda() const noexcept { return lambda_; }
  double alpha() const noexcept { return alpha_; }

 public:
  SeluNode(std::string name, GraphNode* X, double lambda, double alpha);
  DEFINE_GRAPH_NODE_LIKE(SeluNode);
};

class GeluNode : public GraphNodeUnaryElementWiseBase {
 public:
  GeluNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(GeluNode);
};

class SoftPlusNode : public GraphNodeUnaryElementWiseBase {
 public:
  SoftPlusNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(SoftPlusNode);
};

class SwishNode : public GraphNodeUnaryElementWiseBase {
 public:
  SwishNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(SwishNode);
};

class ExpNode : public GraphNodeUnaryElementWiseBase {
 public:
  ExpNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ExpNode);
};

class LogNode : public GraphNodeUnaryElementWiseBase {
 public:
  LogNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(LogNode);
};

class NegateNode : public GraphNodeUnaryElementWiseBase {
 public:
  NegateNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(NegateNode);
};

class InvNode : public GraphNodeUnaryElementWiseBase {
 public:
  InvNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(InvNode);
};
using ReciprocalNode = InvNode;

class SqrtNode : public GraphNodeUnaryElementWiseBase {
 public:
  SqrtNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(SqrtNode);
};

class CbrtNode : public GraphNodeUnaryElementWiseBase {
 public:
  CbrtNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(CbrtNode);
};

class SquareNode : public GraphNodeUnaryElementWiseBase {
 public:
  SquareNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(SquareNode);
};

class CubicNode : public GraphNodeUnaryElementWiseBase {
 public:
  CubicNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(CubicNode);
};

class DropoutNode : public GraphNodeUnaryElementWiseBase {
 private:
  double keep_prob_ = 0;
  DEFINE_GRAPH_NODE_ATTR(DropoutNode, keep_prob_);

 public:
  double keep_prob() const noexcept { return keep_prob_; }

 public:
  DropoutNode(std::string name, GraphNode* X, double keep_prob);
  DEFINE_GRAPH_NODE_LIKE(DropoutNode);
};

class SignNode : public GraphNodeUnaryElementWiseBase {
 public:
  SignNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(SignNode);
};

class AbsNode : public GraphNodeUnaryElementWiseBase {
 public:
  AbsNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(AbsNode);
};

class ClipByValueNode : public GraphNodeUnaryElementWiseBase {
 private:
  double clip_value_min_ = 0;
  double clip_value_max_ = 0;
  DEFINE_GRAPH_NODE_ATTR(ClipByValueNode, clip_value_min_, clip_value_max_);

 public:
  double clip_value_min() const noexcept { return clip_value_min_; }
  double clip_value_max() const noexcept { return clip_value_max_; }

 public:
  ClipByValueNode(std::string name, GraphNode* X, double clip_value_min,
                  double clip_value_max);
  DEFINE_GRAPH_NODE_LIKE(ClipByValueNode);
};

class MatrixBandPartNode : public GraphNodeUnaryElementWiseBase {
 private:
  int num_lower_ = 0;
  int num_upper_ = 0;
  DEFINE_GRAPH_NODE_ATTR(MatrixBandPartNode, num_lower_, num_upper_);

 public:
  int num_lower() const noexcept { return num_lower_; }
  int num_upper() const noexcept { return num_upper_; }

 public:
  MatrixBandPartNode(std::string name, GraphNode* X, int num_lower,
                     int num_upper);
  DEFINE_GRAPH_NODE_LIKE(MatrixBandPartNode);
};

class IdentityNode : public GraphNodeUnaryElementWiseBase {
 public:
  IdentityNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(IdentityNode);
};

/************************************************************************/
/* binary element-wise op */
/************************************************************************/
class AddNode : public GraphNodeBinaryElementWiseBase {
 public:
  AddNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(AddNode);
};

class SubNode : public GraphNodeBinaryElementWiseBase {
 public:
  SubNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(SubNode);
};

class MulNode : public GraphNodeBinaryElementWiseBase {
 public:
  MulNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(MulNode);
};

class DivNode : public GraphNodeBinaryElementWiseBase {
 public:
  DivNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(DivNode);
};

class PowNode : public GraphNodeBinaryElementWiseBase {
 public:
  PowNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(PowNode);
};

class MaxNode : public GraphNodeBinaryElementWiseBase {
 public:
  MaxNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(MaxNode);
};

class MinNode : public GraphNodeBinaryElementWiseBase {
 public:
  MinNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(MinNode);
};

class EqualNode : public GraphNodeBinaryElementWiseBase {
 public:
  EqualNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(EqualNode);
};

class GreaterNode : public GraphNodeBinaryElementWiseBase {
 public:
  GreaterNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(GreaterNode);
};

class GreaterEqualNode : public GraphNodeBinaryElementWiseBase {
 public:
  GreaterEqualNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(GreaterEqualNode);
};

class LessNode : public GraphNodeBinaryElementWiseBase {
 public:
  LessNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(LessNode);
};

class LessEqualNode : public GraphNodeBinaryElementWiseBase {
 public:
  LessEqualNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(LessEqualNode);
};

/************************************************************************/
/* binary broadcast op */
/************************************************************************/
class BroadcastAddNode : public GraphNodeBroadcastBase {
 public:
  BroadcastAddNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastAddNode);
};

class BroadcastSubNode : public GraphNodeBroadcastBase {
 public:
  BroadcastSubNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastSubNode);
};

class BroadcastMulNode : public GraphNodeBroadcastBase {
 public:
  BroadcastMulNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastMulNode);
};

class BroadcastDivNode : public GraphNodeBroadcastBase {
 public:
  BroadcastDivNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastDivNode);
};

class BroadcastPowNode : public GraphNodeBroadcastBase {
 public:
  BroadcastPowNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastPowNode);
};

class BroadcastMaxNode : public GraphNodeBroadcastBase {
 public:
  BroadcastMaxNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastMaxNode);
};

class BroadcastMinNode : public GraphNodeBroadcastBase {
 public:
  BroadcastMinNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastMinNode);
};

class BroadcastEqualNode : public GraphNodeBroadcastBase {
 public:
  BroadcastEqualNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastEqualNode);
};

class BroadcastGreaterNode : public GraphNodeBroadcastBase {
 public:
  BroadcastGreaterNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastGreaterNode);
};

class BroadcastGreaterEqualNode : public GraphNodeBroadcastBase {
 public:
  BroadcastGreaterEqualNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastGreaterEqualNode);
};

class BroadcastLessNode : public GraphNodeBroadcastBase {
 public:
  BroadcastLessNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastLessNode);
};

class BroadcastLessEqualNode : public GraphNodeBroadcastBase {
 public:
  BroadcastLessEqualNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastLessEqualNode);
};

class BroadcastToNode : public GraphNodeUnaryBase {
 private:
  Shape new_shape_;
  DEFINE_GRAPH_NODE_ATTR(BroadcastToNode, new_shape_);

 public:
  const Shape& new_shape() const noexcept { return new_shape_; }

 public:
  BroadcastToNode(std::string name, GraphNode* X, const Shape& shape);
  DEFINE_GRAPH_NODE_LIKE(BroadcastToNode);
};

class BroadcastToLikeNode : public GraphNodeBinaryBase {
 public:
  BroadcastToLikeNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BroadcastToLikeNode);
};

/************************************************************************/
/* for axis op */
/************************************************************************/
class SoftmaxNode : public GraphNodeForAxisBase {
 public:
  SoftmaxNode(std::string name, GraphNode* X, int axis = -1);
  DEFINE_GRAPH_NODE_LIKE(SoftmaxNode);
};

class Softmax2Node : public GraphNodeForAxisBase {
 public:
  Softmax2Node(std::string name, GraphNode* X, int axis = -1);
  DEFINE_GRAPH_NODE_LIKE(Softmax2Node);
};

class LogSoftmaxNode : public GraphNodeForAxisBase {
 public:
  LogSoftmaxNode(std::string name, GraphNode* X, int axis = -1);
  DEFINE_GRAPH_NODE_LIKE(LogSoftmaxNode);
};

class Normalize2Node : public GraphNodeForAxisBase {
 public:
  Normalize2Node(std::string name, GraphNode* X, int axis = -1);
  DEFINE_GRAPH_NODE_LIKE(Normalize2Node);
};

/************************************************************************/
/* reduce axis op */
/************************************************************************/
class ReduceMeanNode : public GraphNodeReduceAxisBase {
 public:
  ReduceMeanNode(std::string name, GraphNode* X, int axis, int keep_dim);
  ReduceMeanNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ReduceMeanNode);
};

class ReduceSumNode : public GraphNodeReduceAxisBase {
 public:
  ReduceSumNode(std::string name, GraphNode* X, int axis, int keep_dim);
  ReduceSumNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ReduceSumNode);
};

class ReduceMaxNode : public GraphNodeReduceAxisBase {
 public:
  ReduceMaxNode(std::string name, GraphNode* X, int axis, int keep_dim);
  ReduceMaxNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ReduceMaxNode);
};

class ReduceMinNode : public GraphNodeReduceAxisBase {
 public:
  ReduceMinNode(std::string name, GraphNode* X, int axis, int keep_dim);
  ReduceMinNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ReduceMinNode);
};

class ReduceL1Node : public GraphNodeReduceAxisBase {
 public:
  ReduceL1Node(std::string name, GraphNode* X, int axis, int keep_dim);
  ReduceL1Node(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ReduceL1Node);
};

class ReduceL2Node : public GraphNodeReduceAxisBase {
 public:
  ReduceL2Node(std::string name, GraphNode* X, int axis, int keep_dim);
  ReduceL2Node(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ReduceL2Node);
};

class ArgMaxNode : public GraphNodeReduceAxisBase {
 public:
  ArgMaxNode(std::string name, GraphNode* X, int axis = 0);
  DEFINE_GRAPH_NODE_LIKE(ArgMaxNode);
};

class ArgMinNode : public GraphNodeReduceAxisBase {
 public:
  ArgMinNode(std::string name, GraphNode* X, int axis = 0);
  DEFINE_GRAPH_NODE_LIKE(ArgMinNode);
};

/************************************************************************/
/* fm op */
/************************************************************************/
class BatchFMInteractionNode : public GraphNodeUnaryBase {
 public:
  BatchFMInteractionNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(BatchFMInteractionNode);
};

class BatchFMInteraction2Node : public GraphNodeBinaryBase {
 public:
  BatchFMInteraction2Node(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BatchFMInteraction2Node);
};

class BatchFMQuadraticNode : public GraphNode {
 public:
  BatchFMQuadraticNode(std::string name, GraphNode* X, GraphNode* V);
  DEFINE_GRAPH_NODE_LIKE(BatchFMQuadraticNode);
};

class BatchGroupFMQuadraticNode : public GraphNodeUnaryBase {
 public:
  BatchGroupFMQuadraticNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(BatchGroupFMQuadraticNode);
};

class BatchGroupFMQuadratic2Node : public GraphNodeUnaryBase {
 public:
  BatchGroupFMQuadratic2Node(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(BatchGroupFMQuadratic2Node);
};

/************************************************************************/
/* conv op */
/************************************************************************/
class GraphNodeConvBase : public GraphNodeBinaryBase {
 public:
  enum DATA_FORMAT {
    DATA_FORMAT_NONE = 0,
    DATA_FORMAT_NCW = 1,
    DATA_FORMAT_NWC = 2,
    DATA_FORMAT_NCHW = 3,
    DATA_FORMAT_NHWC = 4,
    DATA_FORMAT_NCDHW = 5,
    DATA_FORMAT_NDHWC = 6,
  };
  enum PADDING_MODE {
    PADDING_MODE_NONE = 0,
    PADDING_MODE_SAME = 1,
    PADDING_MODE_VALID = 2,
    PADDING_MODE_USE_PADDINGS = 3,
  };

 private:
  int conv_rank_ = 0;
  int data_format_ = DATA_FORMAT_NONE;
  std::vector<int> strides_;
  std::vector<int> dilations_;
  int padding_mode_ = PADDING_MODE_NONE;
  std::vector<int> paddings_;
  DEFINE_GRAPH_NODE_ATTR(GraphNodeConvBase, conv_rank_, data_format_, strides_,
                         dilations_, padding_mode_, paddings_);

 public:
  int conv_rank() const noexcept { return conv_rank_; }
  int data_format() const noexcept { return data_format_; }
  const std::vector<int>& strides() const noexcept { return strides_; }
  int stride() const noexcept { return strides_[0]; }
  const std::vector<int>& dilations() const noexcept { return dilations_; }
  int dilation() const noexcept { return dilations_[0]; }
  int padding_mode() const noexcept { return padding_mode_; }
  const std::vector<int>& paddings() const noexcept { return paddings_; }
  int padding() const noexcept { return paddings_[0]; }

 public:
  GraphNodeConvBase(std::string name, GraphNode* X, GraphNode* K, int conv_rank,
                    int data_format, int stride, int dilation, int padding_mode,
                    int padding);
  GraphNodeConvBase(std::string name, GraphNode* X, GraphNode* K, int conv_rank,
                    int data_format, std::vector<int> strides,
                    std::vector<int> dilations, int padding_mode,
                    std::vector<int> paddings);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodeConvBase);
};

class Conv1dNode : public GraphNodeConvBase {
 public:
  Conv1dNode(std::string name, GraphNode* X, GraphNode* K, int data_format,
             int stride, int dilation, int padding);
  Conv1dNode(std::string name, GraphNode* X, GraphNode* K, int data_format,
             int stride, int dilation, int padding_mode, int padding);
  DEFINE_GRAPH_NODE_LIKE(Conv1dNode);
};

class Conv2dNode : public GraphNodeConvBase {
 public:
  Conv2dNode(std::string name, GraphNode* X, GraphNode* K, int data_format,
             std::vector<int> strides, std::vector<int> dilations,
             std::vector<int> paddings);
  Conv2dNode(std::string name, GraphNode* X, GraphNode* K, int data_format,
             std::vector<int> strides, std::vector<int> dilations,
             int padding_mode, std::vector<int> paddings);
  DEFINE_GRAPH_NODE_LIKE(Conv2dNode);
};

class Conv3dNode : public GraphNodeConvBase {
 public:
  Conv3dNode(std::string name, GraphNode* X, GraphNode* K, int data_format,
             std::vector<int> strides, std::vector<int> dilations,
             std::vector<int> paddings);
  Conv3dNode(std::string name, GraphNode* X, GraphNode* K, int data_format,
             std::vector<int> strides, std::vector<int> dilations,
             int padding_mode, std::vector<int> paddings);
  DEFINE_GRAPH_NODE_LIKE(Conv3dNode);
};

/************************************************************************/
/* pool op */
/************************************************************************/
class GraphNodePoolBase : public GraphNodeUnaryBase {
 public:
  enum POOL_TYPE {
    POOL_TYPE_NONE = 0,
    POOL_TYPE_MAX = 1,
    POOL_TYPE_AVG = 2,
  };
  enum DATA_FORMAT {
    DATA_FORMAT_NONE = 0,
    DATA_FORMAT_NCW = 1,
    DATA_FORMAT_NWC = 2,
    DATA_FORMAT_NCHW = 3,
    DATA_FORMAT_NHWC = 4,
    DATA_FORMAT_NCDHW = 5,
    DATA_FORMAT_NDHWC = 6,
  };
  enum PADDING_MODE {
    PADDING_MODE_NONE = 0,
    PADDING_MODE_SAME = 1,
    PADDING_MODE_VALID = 2,
    PADDING_MODE_USE_PADDINGS = 3,
  };

 private:
  int pool_type_ = POOL_TYPE_NONE;
  int pool_rank_ = 0;
  int data_format_ = DATA_FORMAT_NONE;
  std::vector<int> kernel_sizes_;
  std::vector<int> strides_;
  std::vector<int> dilations_;
  int padding_mode_ = PADDING_MODE_NONE;
  std::vector<int> paddings_;
  int ceil_mode_ = 0;
  int count_include_pad_ = 0;
  DEFINE_GRAPH_NODE_ATTR(GraphNodePoolBase, pool_type_, pool_rank_,
                         data_format_, kernel_sizes_, strides_, dilations_,
                         padding_mode_, paddings_, ceil_mode_,
                         count_include_pad_);

 public:
  int pool_type() const noexcept { return pool_type_; }
  int pool_rank() const noexcept { return pool_rank_; }
  int data_format() const noexcept { return data_format_; }
  const std::vector<int>& kernel_sizes() const noexcept {
    return kernel_sizes_;
  }
  const std::vector<int>& strides() const noexcept { return strides_; }
  int stride() const noexcept { return strides_[0]; }
  const std::vector<int>& dilations() const noexcept { return dilations_; }
  int dilation() const noexcept { return dilations_[0]; }
  int padding_mode() const noexcept { return padding_mode_; }
  const std::vector<int>& paddings() const noexcept { return paddings_; }
  int padding() const noexcept { return paddings_[0]; }
  int ceil_mode() const noexcept { return ceil_mode_; }
  int count_include_pad() const noexcept { return count_include_pad_; }

 public:
  GraphNodePoolBase(std::string name, GraphNode* X, int pool_type,
                    int pool_rank, int data_format, int kernel_size, int stride,
                    int dilation, int padding_mode, int padding, int ceil_mode,
                    int count_include_pad);
  GraphNodePoolBase(std::string name, GraphNode* X, int pool_type,
                    int pool_rank, int data_format,
                    std::vector<int> kernel_sizes, std::vector<int> strides,
                    std::vector<int> dilations, int padding_mode,
                    std::vector<int> paddings, int ceil_mode,
                    int count_include_pad);
  DEFINE_GRAPH_NODE_LIKE_BASE(GraphNodePoolBase);
};

class MaxPool1dNode : public GraphNodePoolBase {
 public:
  MaxPool1dNode(std::string name, GraphNode* X, int data_format,
                int kernel_size, int stride, int dilation, int padding,
                int ceil_mode);
  MaxPool1dNode(std::string name, GraphNode* X, int data_format,
                int kernel_size, int stride, int dilation, int padding_mode,
                int padding, int ceil_mode);
  DEFINE_GRAPH_NODE_LIKE(MaxPool1dNode);
};

class MaxPool2dNode : public GraphNodePoolBase {
 public:
  MaxPool2dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                std::vector<int> dilations, std::vector<int> paddings,
                int ceil_mode = 0);
  MaxPool2dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                std::vector<int> dilations, int padding_mode,
                std::vector<int> paddings, int ceil_mode = 0);
  DEFINE_GRAPH_NODE_LIKE(MaxPool2dNode);
};

class MaxPool3dNode : public GraphNodePoolBase {
 public:
  MaxPool3dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                std::vector<int> dilations, std::vector<int> paddings,
                int ceil_mode = 0);
  MaxPool3dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                std::vector<int> dilations, int padding_mode,
                std::vector<int> paddings, int ceil_mode = 0);
  DEFINE_GRAPH_NODE_LIKE(MaxPool3dNode);
};

class AvgPool1dNode : public GraphNodePoolBase {
 public:
  AvgPool1dNode(std::string name, GraphNode* X, int data_format,
                int kernel_size, int stride, int padding, int ceil_mode,
                int count_include_pad);
  AvgPool1dNode(std::string name, GraphNode* X, int data_format,
                int kernel_size, int stride, int padding_mode, int padding,
                int ceil_mode, int count_include_pad);
  DEFINE_GRAPH_NODE_LIKE(AvgPool1dNode);
};

class AvgPool2dNode : public GraphNodePoolBase {
 public:
  AvgPool2dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                std::vector<int> paddings, int ceil_mode = 0,
                int count_include_pad = 0);
  AvgPool2dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                int padding_mode, std::vector<int> paddings, int ceil_mode = 0,
                int count_include_pad = 0);
  DEFINE_GRAPH_NODE_LIKE(AvgPool2dNode);
};

class AvgPool3dNode : public GraphNodePoolBase {
 public:
  AvgPool3dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                std::vector<int> paddings, int ceil_mode = 0,
                int count_include_pad = 0);
  AvgPool3dNode(std::string name, GraphNode* X, int data_format,
                std::vector<int> kernel_sizes, std::vector<int> strides,
                int padding_mode, std::vector<int> paddings, int ceil_mode = 0,
                int count_include_pad = 0);
  DEFINE_GRAPH_NODE_LIKE(AvgPool3dNode);
};

/************************************************************************/
/* loss op */
/************************************************************************/
class AbsoluteErrorNode : public GraphNodeBinaryElementWiseBase {
 public:
  AbsoluteErrorNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(AbsoluteErrorNode);
};

class SquareErrorNode : public GraphNodeBinaryElementWiseBase {
 public:
  SquareErrorNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(SquareErrorNode);
};

class BCELossNode : public GraphNodeBinaryElementWiseBase {
 public:
  BCELossNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BCELossNode);
};

class BCELoss2Node : public GraphNodeBinaryElementWiseBase {
 public:
  BCELoss2Node(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BCELoss2Node);
};

class SigmoidBCELossNode : public GraphNodeBinaryElementWiseBase {
 public:
  SigmoidBCELossNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(SigmoidBCELossNode);
};

class SigmoidBCELoss2Node : public GraphNodeBinaryElementWiseBase {
 public:
  SigmoidBCELoss2Node(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(SigmoidBCELoss2Node);
};

class BatchCELossNode : public GraphNodeBinaryBase {
 public:
  BatchCELossNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BatchCELossNode);
};

class BatchCELoss2Node : public GraphNodeBinaryBase {
 public:
  BatchCELoss2Node(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BatchCELoss2Node);
};

class BatchSoftmaxCELossNode : public GraphNodeBinaryBase {
 public:
  BatchSoftmaxCELossNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BatchSoftmaxCELossNode);
};

class BatchSoftmaxCELoss2Node : public GraphNodeBinaryBase {
 public:
  BatchSoftmaxCELoss2Node(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BatchSoftmaxCELoss2Node);
};

class FocalLossNode : public GraphNodeBinaryElementWiseBase {
 private:
  double alpha_ = 0;
  double gamma_ = 0;
  DEFINE_GRAPH_NODE_ATTR(FocalLossNode, alpha_, gamma_);

 public:
  double alpha() const noexcept { return alpha_; }
  double gamma() const noexcept { return gamma_; }

 public:
  FocalLossNode(std::string name, GraphNode* X, GraphNode* Y, double alpha,
                double gamma);
  DEFINE_GRAPH_NODE_LIKE(FocalLossNode);
};

class SigmoidFocalLossNode : public GraphNodeBinaryElementWiseBase {
 private:
  double alpha_ = 0;
  double gamma_ = 0;
  DEFINE_GRAPH_NODE_ATTR(SigmoidFocalLossNode, alpha_, gamma_);

 public:
  double alpha() const noexcept { return alpha_; }
  double gamma() const noexcept { return gamma_; }

 public:
  SigmoidFocalLossNode(std::string name, GraphNode* X, GraphNode* Y,
                       double alpha, double gamma);
  DEFINE_GRAPH_NODE_LIKE(SigmoidFocalLossNode);
};

/************************************************************************/
/* instance op */
/************************************************************************/
class InstanceNode : public GraphNode {
 public:
  InstanceNode(std::string name, const Shape& shape, int tensor_type);
  DEFINE_GRAPH_NODE_LIKE(InstanceNode);
};

/************************************************************************/
/* variable op */
/************************************************************************/
class VariableNode : public GraphNode {
 public:
  VariableNode(std::string name, const Shape& shape, int tensor_type);
  VariableNode(std::string name, const Shape& shape, int tensor_type,
               int initializer_type, double initializer_param1,
               double initializer_param2);
  VariableNode(std::string name, const Shape& shape);
  VariableNode(std::string name, const Shape& shape, int initializer_type,
               double initializer_param1, double initializer_param2);
  DEFINE_GRAPH_NODE_LIKE(VariableNode);
};

/************************************************************************/
/* constant op */
/************************************************************************/
class ConstantNode : public GraphNode {
 public:
  enum CONSTANT_TYPE {
    CONSTANT_TYPE_NONE = 0,
    CONSTANT_TYPE_VALUE = 1,
    CONSTANT_TYPE_VALUES = 2,
    CONSTANT_TYPE_INITIALIZER = 3,
  };

 private:
  int constant_type_ = CONSTANT_TYPE_NONE;
  double value_ = 0;
  std::vector<double> values_;
  DEFINE_GRAPH_NODE_ATTR(ConstantNode, constant_type_, value_, values_);

 public:
  int constant_type() const noexcept { return constant_type_; }
  double value() const noexcept { return value_; }
  const std::vector<double>& values() const noexcept { return values_; }

 public:
  ConstantNode(std::string name, const Shape& shape, double value);
  ConstantNode(std::string name, const Shape& shape,
               std::vector<double> values);
  ConstantNode(std::string name, const Shape& shape, int initializer_type,
               double initializer_param1, double initializer_param2);
  DEFINE_GRAPH_NODE_LIKE(ConstantNode);
};

class ZerosNode : public ConstantNode {
 public:
  ZerosNode(std::string name, const Shape& shape);
  DEFINE_GRAPH_NODE_LIKE(ZerosNode);
};

class OnesNode : public ConstantNode {
 public:
  OnesNode(std::string name, const Shape& shape);
  DEFINE_GRAPH_NODE_LIKE(OnesNode);
};

class RandomNormalNode : public ConstantNode {
 public:
  RandomNormalNode(std::string name, const Shape& shape, double mean,
                   double stddev);
  DEFINE_GRAPH_NODE_LIKE(RandomNormalNode);
};

class RandomUniformNode : public ConstantNode {
 public:
  RandomUniformNode(std::string name, const Shape& shape, double _min,
                    double _max);
  DEFINE_GRAPH_NODE_LIKE(RandomUniformNode);
};

class ConstantLikeNode : public GraphNodeUnaryElementWiseBase {
 public:
  enum CONSTANT_TYPE {
    CONSTANT_TYPE_NONE = 0,
    CONSTANT_TYPE_VALUE = 1,
    CONSTANT_TYPE_INITIALIZER = 3,
  };

 private:
  int constant_type_ = CONSTANT_TYPE_NONE;
  double value_ = 0;
  DEFINE_GRAPH_NODE_ATTR(ConstantLikeNode, constant_type_, value_);

 public:
  int constant_type() const noexcept { return constant_type_; }
  double value() const noexcept { return value_; }

 public:
  ConstantLikeNode(std::string name, GraphNode* X, double value);
  ConstantLikeNode(std::string name, GraphNode* X, int initializer_type,
                   double initializer_param1, double initializer_param2);
  DEFINE_GRAPH_NODE_LIKE(ConstantLikeNode);
};

class ZerosLikeNode : public ConstantLikeNode {
 public:
  ZerosLikeNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(ZerosLikeNode);
};

class OnesLikeNode : public ConstantLikeNode {
 public:
  OnesLikeNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(OnesLikeNode);
};

class RandomNormalLikeNode : public ConstantLikeNode {
 public:
  RandomNormalLikeNode(std::string name, GraphNode* X, double mean,
                       double stddev);
  DEFINE_GRAPH_NODE_LIKE(RandomNormalLikeNode);
};

class RandomUniformLikeNode : public ConstantLikeNode {
 public:
  RandomUniformLikeNode(std::string name, GraphNode* X, double _min,
                        double _max);
  DEFINE_GRAPH_NODE_LIKE(RandomUniformLikeNode);
};

/************************************************************************/
/* other op */
/************************************************************************/
class TFEmbeddingLookupNode : public GraphNode {
 public:
  TFEmbeddingLookupNode(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(TFEmbeddingLookupNode);
};

class EmbeddingLookupNode : public GraphNode {
 public:
  EmbeddingLookupNode(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(EmbeddingLookupNode);
};

class GroupEmbeddingLookupNode : public GraphNode {
 private:
  std::vector<uint16_t> group_ids_;
  DEFINE_GRAPH_NODE_ATTR(GroupEmbeddingLookupNode, group_ids_);

 public:
  const std::vector<uint16_t>& group_ids() const noexcept { return group_ids_; }

 public:
  GroupEmbeddingLookupNode(std::string name, GraphNode* X,
                           const std::vector<GraphNode*>& W,
                           std::vector<uint16_t> group_ids);
  DEFINE_GRAPH_NODE_LIKE(GroupEmbeddingLookupNode);
};

class GroupEmbeddingLookup2Node : public GraphNode {
 private:
  std::vector<uint16_t> group_ids_;
  DEFINE_GRAPH_NODE_ATTR(GroupEmbeddingLookup2Node, group_ids_);

 public:
  const std::vector<uint16_t>& group_ids() const noexcept { return group_ids_; }

 public:
  GroupEmbeddingLookup2Node(std::string name, GraphNode* X, GraphNode* W,
                            std::vector<uint16_t> group_ids);
  DEFINE_GRAPH_NODE_LIKE(GroupEmbeddingLookup2Node);
};

class GEMMNode : public GraphNodeBinaryBase {
 private:
  int transX_ = 0;
  int transY_ = 0;
  DEFINE_GRAPH_NODE_ATTR(GEMMNode, transX_, transY_);

 public:
  int transX() const noexcept { return transX_; }
  int transY() const noexcept { return transY_; }

 public:
  GEMMNode(std::string name, GraphNode* X, GraphNode* Y, int transX,
           int transY);
  DEFINE_GRAPH_NODE_LIKE(GEMMNode);
};

class BatchGEMMNode : public GraphNodeBinaryBase {
 private:
  int transX_ = 0;
  int transY_ = 0;
  DEFINE_GRAPH_NODE_ATTR(BatchGEMMNode, transX_, transY_);

 public:
  int transX() const noexcept { return transX_; }
  int transY() const noexcept { return transY_; }

 public:
  BatchGEMMNode(std::string name, GraphNode* X, GraphNode* Y, int transX,
                int transY);
  DEFINE_GRAPH_NODE_LIKE(BatchGEMMNode);
};

class MatmulNode : public GraphNodeBinaryBase {
 public:
  MatmulNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(MatmulNode);
};

class Matmul2Node : public GraphNodeBinaryBase {
 private:
  int transX_ = 0;
  int transY_ = 0;
  DEFINE_GRAPH_NODE_ATTR(Matmul2Node, transX_, transY_);

 public:
  int transX() const noexcept { return transX_; }
  int transY() const noexcept { return transY_; }

 public:
  Matmul2Node(std::string name, GraphNode* X, GraphNode* Y, int transX,
              int transY);
  DEFINE_GRAPH_NODE_LIKE(Matmul2Node);
};

class FullyConnectNode : public GraphNode {
 public:
  FullyConnectNode(std::string name, GraphNode* X, GraphNode* W);
  FullyConnectNode(std::string name, GraphNode* X, GraphNode* W, GraphNode* b);
  DEFINE_GRAPH_NODE_LIKE(FullyConnectNode);
};

class TensorDotNode : public GraphNodeBinaryBase {
 private:
  int use_axes_n_ = 0;
  int axes_n_ = 0;
  Shape Xaxes_, Yaxes_;
  DEFINE_GRAPH_NODE_ATTR(TensorDotNode, use_axes_n_, axes_n_, Xaxes_, Yaxes_);

 public:
  int use_axes_n() const noexcept { return use_axes_n_; }
  int axes_n() const noexcept { return axes_n_; }
  const Shape& Xaxes() const noexcept { return Xaxes_; }
  const Shape& Yaxes() const noexcept { return Yaxes_; }

 public:
  TensorDotNode(std::string name, GraphNode* X, GraphNode* Y, int axes_n);
  TensorDotNode(std::string name, GraphNode* X, GraphNode* Y,
                const Shape& Xaxes, const Shape& Yaxes);
  DEFINE_GRAPH_NODE_LIKE(TensorDotNode);
};

class InnerNode : public GraphNodeBinaryBase {
 public:
  InnerNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(InnerNode);
};

class OuterNode : public GraphNodeBinaryBase {
 public:
  OuterNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(OuterNode);
};

class AddNNode : public GraphNode {
 public:
  AddNNode(std::string name, std::vector<GraphNode*> X);
  DEFINE_GRAPH_NODE_LIKE(AddNNode);
};

class ConcatNode : public GraphNode {
 private:
  int axis_ = 0;
  DEFINE_GRAPH_NODE_ATTR(ConcatNode, axis_);

 public:
  int axis() const noexcept { return axis_; }

 public:
  ConcatNode(std::string name, std::vector<GraphNode*> X, int axis = -1);
  DEFINE_GRAPH_NODE_LIKE(ConcatNode);
};

// ReshapeNode is deprecated by Reshape2Node.
class ReshapeNode : public GraphNodeUnaryBase {
 public:
  ReshapeNode(std::string name, GraphNode* X, const Shape& shape);
  DEFINE_GRAPH_NODE_LIKE(ReshapeNode);
};

// ReshapeFastNode is deprecated by Reshape2FastNode.
class ReshapeFastNode : public GraphNodeUnaryBase {
 public:
  ReshapeFastNode(std::string name, GraphNode* X, const Shape& shape);
  DEFINE_GRAPH_NODE_LIKE(ReshapeFastNode);
};

class Reshape2Node : public GraphNodeUnaryBase {
 private:
  Shape new_shape_;
  DEFINE_GRAPH_NODE_ATTR(Reshape2Node, new_shape_);

 public:
  const Shape& new_shape() const noexcept { return new_shape_; }

 public:
  Reshape2Node(std::string name, GraphNode* X, const Shape& shape);
  DEFINE_GRAPH_NODE_LIKE(Reshape2Node);
};

class Reshape2FastNode : public GraphNodeUnaryBase {
 private:
  Shape new_shape_;
  DEFINE_GRAPH_NODE_ATTR(Reshape2FastNode, new_shape_);

 public:
  const Shape& new_shape() const noexcept { return new_shape_; }

 public:
  Reshape2FastNode(std::string name, GraphNode* X, const Shape& shape);
  DEFINE_GRAPH_NODE_LIKE(Reshape2FastNode);
};
using ReshapeZeroCopyNode = Reshape2FastNode;

class ExpandDimNode : public GraphNodeUnaryBase {
 private:
  int axis_ = 0;
  DEFINE_GRAPH_NODE_ATTR(ExpandDimNode, axis_);

 public:
  int axis() const noexcept { return axis_; }

 public:
  ExpandDimNode(std::string name, GraphNode* X, int axis);
  DEFINE_GRAPH_NODE_LIKE(ExpandDimNode);
};

class ExpandDimFastNode : public GraphNodeUnaryBase {
 private:
  int axis_ = 0;
  DEFINE_GRAPH_NODE_ATTR(ExpandDimFastNode, axis_);

 public:
  int axis() const noexcept { return axis_; }

 public:
  ExpandDimFastNode(std::string name, GraphNode* X, int axis);
  DEFINE_GRAPH_NODE_LIKE(ExpandDimFastNode);
};
using ExpandDimZeroCopyNode = ExpandDimFastNode;

class SqueezeNode : public GraphNodeUnaryBase {
 private:
  int axis_ = 0;
  DEFINE_GRAPH_NODE_ATTR(SqueezeNode, axis_);

 public:
  int axis() const noexcept { return axis_; }

 public:
  SqueezeNode(std::string name, GraphNode* X, int axis);
  DEFINE_GRAPH_NODE_LIKE(SqueezeNode);
};

class SqueezeFastNode : public GraphNodeUnaryBase {
 private:
  int axis_ = 0;
  DEFINE_GRAPH_NODE_ATTR(SqueezeFastNode, axis_);

 public:
  int axis() const noexcept { return axis_; }

 public:
  SqueezeFastNode(std::string name, GraphNode* X, int axis);
  DEFINE_GRAPH_NODE_LIKE(SqueezeFastNode);
};
using SqueezeZeroCopyNode = SqueezeFastNode;

class TransposeNode : public GraphNodeUnaryBase {
 private:
  Shape axes_;
  DEFINE_GRAPH_NODE_ATTR(TransposeNode, axes_);

 public:
  const Shape& axes() const noexcept { return axes_; }

 public:
  TransposeNode(std::string name, GraphNode* X, const Shape& axes);
  DEFINE_GRAPH_NODE_LIKE(TransposeNode);
};

class SubscriptNode : public GraphNodeUnaryBase {
 private:
  int axis_ = 0;
  int index_ = 0;
  DEFINE_GRAPH_NODE_ATTR(SubscriptNode, axis_, index_);

 public:
  int axis() const noexcept { return axis_; }
  int index() const noexcept { return index_; }

 public:
  SubscriptNode(std::string name, GraphNode* X, int axis, int index);
  DEFINE_GRAPH_NODE_LIKE(SubscriptNode);
};

class Subscript2Node : public GraphNodeBinaryBase {
 private:
  int axis_ = 0;
  DEFINE_GRAPH_NODE_ATTR(Subscript2Node, axis_);

 public:
  int axis() const noexcept { return axis_; }

 public:
  Subscript2Node(std::string name, GraphNode* X, GraphNode* Y, int axis);
  DEFINE_GRAPH_NODE_LIKE(Subscript2Node);
};

class SubscriptRangeNode : public GraphNodeUnaryBase {
 private:
  int axis_ = 0;
  int begin_index_ = 0;
  int end_index_ = 0;
  DEFINE_GRAPH_NODE_ATTR(SubscriptRangeNode, axis_, begin_index_, end_index_);

 public:
  int axis() const noexcept { return axis_; }
  int begin_index() const noexcept { return begin_index_; }
  int end_index() const noexcept { return end_index_; }

 public:
  SubscriptRangeNode(std::string name, GraphNode* X, int axis, int begin_index,
                     int end_index);
  DEFINE_GRAPH_NODE_LIKE(SubscriptRangeNode);
};

class LayerNormNode : public GraphNode {
 public:
  LayerNormNode(std::string name, GraphNode* X, GraphNode* gamma,
                GraphNode* beta);
  DEFINE_GRAPH_NODE_LIKE(LayerNormNode);
};

class SequenceMaskNode : public GraphNodeUnaryBase {
 private:
  int max_size_ = 0;
  DEFINE_GRAPH_NODE_ATTR(SequenceMaskNode, max_size_);

 public:
  int max_size() const noexcept { return max_size_; }

 public:
  SequenceMaskNode(std::string name, GraphNode* X, int max_size);
  DEFINE_GRAPH_NODE_LIKE(SequenceMaskNode);
};

class WhereNode : public GraphNode {
 public:
  WhereNode(std::string name, GraphNode* C, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(WhereNode);
};

class TileNode : public GraphNodeUnaryBase {
 private:
  std::vector<int> reps_;
  DEFINE_GRAPH_NODE_ATTR(TileNode, reps_);

 public:
  const std::vector<int>& reps() const noexcept { return reps_; }

 public:
  TileNode(std::string name, GraphNode* X, int rep);
  TileNode(std::string name, GraphNode* X, std::vector<int> reps);
  DEFINE_GRAPH_NODE_LIKE(TileNode);
};

class BatchCosNode : public GraphNodeBinaryBase {
 public:
  BatchCosNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BatchCosNode);
};

class BatchDotNode : public GraphNodeBinaryBase {
 public:
  BatchDotNode(std::string name, GraphNode* X, GraphNode* Y);
  DEFINE_GRAPH_NODE_LIKE(BatchDotNode);
};

class StopGradNode : public GraphNode {
 public:
  StopGradNode(std::string name, GraphNode* X);
  DEFINE_GRAPH_NODE_LIKE(StopGradNode);
};

class BatchNormNode : public GraphNode {
 private:
  double moving_decay_ = 0;
  DEFINE_GRAPH_NODE_ATTR(BatchNormNode, moving_decay_);

 public:
  double moving_decay() const noexcept { return moving_decay_; }

 public:
  BatchNormNode(std::string name, GraphNode* X, GraphNode* gamma,
                GraphNode* beta, GraphNode* mean, GraphNode* var,
                double moving_decay = 0.9);
  DEFINE_GRAPH_NODE_LIKE(BatchNormNode);
};

/************************************************************************/
/* fast node creator */
/************************************************************************/
DEFINE_GRAPH_NODE_CREATOR(Sigmoid)
DEFINE_GRAPH_NODE_CREATOR(Tanh)
DEFINE_GRAPH_NODE_CREATOR(Relu)
DEFINE_GRAPH_NODE_CREATOR(LeakyRelu)
DEFINE_GRAPH_NODE_CREATOR(Elu)
DEFINE_GRAPH_NODE_CREATOR(Selu)
DEFINE_GRAPH_NODE_CREATOR(Gelu)
DEFINE_GRAPH_NODE_CREATOR(SoftPlus)
DEFINE_GRAPH_NODE_CREATOR(Swish)
DEFINE_GRAPH_NODE_CREATOR(Exp)
DEFINE_GRAPH_NODE_CREATOR(Log)
DEFINE_GRAPH_NODE_CREATOR(Negate)
DEFINE_GRAPH_NODE_CREATOR(Inv)
DEFINE_GRAPH_NODE_CREATOR(Reciprocal)
DEFINE_GRAPH_NODE_CREATOR(Sqrt)
DEFINE_GRAPH_NODE_CREATOR(Cbrt)
DEFINE_GRAPH_NODE_CREATOR(Square)
DEFINE_GRAPH_NODE_CREATOR(Cubic)
DEFINE_GRAPH_NODE_CREATOR(Dropout)
DEFINE_GRAPH_NODE_CREATOR(Sign)
DEFINE_GRAPH_NODE_CREATOR(Abs)
DEFINE_GRAPH_NODE_CREATOR(ClipByValue)
DEFINE_GRAPH_NODE_CREATOR(MatrixBandPart)
DEFINE_GRAPH_NODE_CREATOR(Identity)
DEFINE_GRAPH_NODE_CREATOR(Add)
DEFINE_GRAPH_NODE_CREATOR(Sub)
DEFINE_GRAPH_NODE_CREATOR(Mul)
DEFINE_GRAPH_NODE_CREATOR(Div)
DEFINE_GRAPH_NODE_CREATOR(Pow)
DEFINE_GRAPH_NODE_CREATOR(Max)
DEFINE_GRAPH_NODE_CREATOR(Min)
DEFINE_GRAPH_NODE_CREATOR(Equal)
DEFINE_GRAPH_NODE_CREATOR(Greater)
DEFINE_GRAPH_NODE_CREATOR(GreaterEqual)
DEFINE_GRAPH_NODE_CREATOR(Less)
DEFINE_GRAPH_NODE_CREATOR(LessEqual)
DEFINE_GRAPH_NODE_CREATOR(BroadcastAdd)
DEFINE_GRAPH_NODE_CREATOR(BroadcastSub)
DEFINE_GRAPH_NODE_CREATOR(BroadcastMul)
DEFINE_GRAPH_NODE_CREATOR(BroadcastDiv)
DEFINE_GRAPH_NODE_CREATOR(BroadcastPow)
DEFINE_GRAPH_NODE_CREATOR(BroadcastMax)
DEFINE_GRAPH_NODE_CREATOR(BroadcastMin)
DEFINE_GRAPH_NODE_CREATOR(BroadcastEqual)
DEFINE_GRAPH_NODE_CREATOR(BroadcastGreater)
DEFINE_GRAPH_NODE_CREATOR(BroadcastGreaterEqual)
DEFINE_GRAPH_NODE_CREATOR(BroadcastLess)
DEFINE_GRAPH_NODE_CREATOR(BroadcastLessEqual)
DEFINE_GRAPH_NODE_CREATOR(BroadcastTo)
DEFINE_GRAPH_NODE_CREATOR(BroadcastToLike)
DEFINE_GRAPH_NODE_CREATOR(Softmax)
DEFINE_GRAPH_NODE_CREATOR(Softmax2)
DEFINE_GRAPH_NODE_CREATOR(LogSoftmax)
DEFINE_GRAPH_NODE_CREATOR(Normalize2)
DEFINE_GRAPH_NODE_CREATOR(ReduceMean)
DEFINE_GRAPH_NODE_CREATOR(ReduceSum)
DEFINE_GRAPH_NODE_CREATOR(ReduceMax)
DEFINE_GRAPH_NODE_CREATOR(ReduceMin)
DEFINE_GRAPH_NODE_CREATOR(ReduceL1)
DEFINE_GRAPH_NODE_CREATOR(ReduceL2)
DEFINE_GRAPH_NODE_CREATOR(ArgMax)
DEFINE_GRAPH_NODE_CREATOR(ArgMin)
DEFINE_GRAPH_NODE_CREATOR(BatchFMInteraction)
DEFINE_GRAPH_NODE_CREATOR(BatchFMInteraction2)
DEFINE_GRAPH_NODE_CREATOR(BatchFMQuadratic)
DEFINE_GRAPH_NODE_CREATOR(BatchGroupFMQuadratic)
DEFINE_GRAPH_NODE_CREATOR(BatchGroupFMQuadratic2)

DEFINE_GRAPH_NODE_CREATOR(Conv1d)
// DEFINE_GRAPH_NODE_CREATOR(Conv2d)
// DEFINE_GRAPH_NODE_CREATOR(Conv3d)
inline GraphNode* Conv2d(std::string name, GraphNode* X, GraphNode* K,
                         int data_format, std::vector<int> strides,
                         std::vector<int> dilations,
                         std::vector<int> paddings) {
  return new Conv2dNode(std::move(name), X, K, data_format, std::move(strides),
                        std::move(dilations), std::move(paddings));
}
inline GraphNode* Conv2d(std::string name, GraphNode* X, GraphNode* K,
                         int data_format, std::vector<int> strides,
                         std::vector<int> dilations, int padding_mode,
                         std::vector<int> paddings) {
  return new Conv2dNode(std::move(name), X, K, data_format, std::move(strides),
                        std::move(dilations), padding_mode,
                        std::move(paddings));
}
inline GraphNode* Conv3d(std::string name, GraphNode* X, GraphNode* K,
                         int data_format, std::vector<int> strides,
                         std::vector<int> dilations,
                         std::vector<int> paddings) {
  return new Conv3dNode(std::move(name), X, K, data_format, std::move(strides),
                        std::move(dilations), std::move(paddings));
}
inline GraphNode* Conv3d(std::string name, GraphNode* X, GraphNode* K,
                         int data_format, std::vector<int> strides,
                         std::vector<int> dilations, int padding_mode,
                         std::vector<int> paddings) {
  return new Conv3dNode(std::move(name), X, K, data_format, std::move(strides),
                        std::move(dilations), padding_mode,
                        std::move(paddings));
}
DEFINE_GRAPH_NODE_CREATOR(MaxPool1d)
// DEFINE_GRAPH_NODE_CREATOR(MaxPool2d)
// DEFINE_GRAPH_NODE_CREATOR(MaxPool3d)
inline GraphNode* MaxPool2d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides,
                            std::vector<int> dilations,
                            std::vector<int> paddings, int ceil_mode = 0) {
  return new MaxPool2dNode(
      std::move(name), X, data_format, std::move(kernel_sizes),
      std::move(strides), std::move(dilations), std::move(paddings), ceil_mode);
}
inline GraphNode* MaxPool2d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides,
                            std::vector<int> dilations, int padding_mode,
                            std::vector<int> paddings, int ceil_mode = 0) {
  return new MaxPool2dNode(std::move(name), X, data_format,
                           std::move(kernel_sizes), std::move(strides),
                           std::move(dilations), padding_mode,
                           std::move(paddings), ceil_mode);
}
inline GraphNode* MaxPool3d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides,
                            std::vector<int> dilations,
                            std::vector<int> paddings, int ceil_mode = 0) {
  return new MaxPool3dNode(
      std::move(name), X, data_format, std::move(kernel_sizes),
      std::move(strides), std::move(dilations), std::move(paddings), ceil_mode);
}
inline GraphNode* MaxPool3d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides,
                            std::vector<int> dilations, int padding_mode,
                            std::vector<int> paddings, int ceil_mode = 0) {
  return new MaxPool3dNode(std::move(name), X, data_format,
                           std::move(kernel_sizes), std::move(strides),
                           std::move(dilations), padding_mode,
                           std::move(paddings), ceil_mode);
}
DEFINE_GRAPH_NODE_CREATOR(AvgPool1d)
// DEFINE_GRAPH_NODE_CREATOR(AvgPool2d)
// DEFINE_GRAPH_NODE_CREATOR(AvgPool3d)
inline GraphNode* AvgPool2d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides, std::vector<int> paddings,
                            int ceil_mode = 0, int count_include_pad = 0) {
  return new AvgPool2dNode(std::move(name), X, data_format,
                           std::move(kernel_sizes), std::move(strides),
                           std::move(paddings), ceil_mode, count_include_pad);
}
inline GraphNode* AvgPool2d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides, int padding_mode,
                            std::vector<int> paddings, int ceil_mode = 0,
                            int count_include_pad = 0) {
  return new AvgPool2dNode(std::move(name), X, data_format,
                           std::move(kernel_sizes), std::move(strides),
                           padding_mode, std::move(paddings), ceil_mode,
                           count_include_pad);
}
inline GraphNode* AvgPool3d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides, std::vector<int> paddings,
                            int ceil_mode = 0, int count_include_pad = 0) {
  return new AvgPool3dNode(std::move(name), X, data_format,
                           std::move(kernel_sizes), std::move(strides),
                           std::move(paddings), ceil_mode, count_include_pad);
}
inline GraphNode* AvgPool3d(std::string name, GraphNode* X, int data_format,
                            std::vector<int> kernel_sizes,
                            std::vector<int> strides, int padding_mode,
                            std::vector<int> paddings, int ceil_mode = 0,
                            int count_include_pad = 0) {
  return new AvgPool3dNode(std::move(name), X, data_format,
                           std::move(kernel_sizes), std::move(strides),
                           padding_mode, std::move(paddings), ceil_mode,
                           count_include_pad);
}

DEFINE_GRAPH_NODE_CREATOR(AbsoluteError)
DEFINE_GRAPH_NODE_CREATOR(SquareError)
DEFINE_GRAPH_NODE_CREATOR(BCELoss)
DEFINE_GRAPH_NODE_CREATOR(BCELoss2)
DEFINE_GRAPH_NODE_CREATOR(SigmoidBCELoss)
DEFINE_GRAPH_NODE_CREATOR(SigmoidBCELoss2)
DEFINE_GRAPH_NODE_CREATOR(BatchCELoss)
DEFINE_GRAPH_NODE_CREATOR(BatchCELoss2)
DEFINE_GRAPH_NODE_CREATOR(BatchSoftmaxCELoss)
DEFINE_GRAPH_NODE_CREATOR(BatchSoftmaxCELoss2)
DEFINE_GRAPH_NODE_CREATOR(FocalLoss)
DEFINE_GRAPH_NODE_CREATOR(SigmoidFocalLoss)

// DEFINE_GRAPH_NODE_CREATOR(Instance)
// use InstanceNode creator in graph_module_creator.h

// DEFINE_GRAPH_NODE_CREATOR(Variable)
// use variable scope API in variable_scope.h

// DEFINE_GRAPH_NODE_CREATOR(Constant)
inline GraphNode* ConstantScalar(std::string name, double value) {
  return new ConstantNode(std::move(name), Shape(1), value);
}
inline GraphNode* ConstantVector(std::string name, std::vector<double> values) {
  Shape shape((int)values.size());
  return new ConstantNode(std::move(name), shape, std::move(values));
}
inline GraphNode* Constant(std::string name, const Shape& shape, double value) {
  return new ConstantNode(std::move(name), shape, value);
}
inline GraphNode* Constant(std::string name, const Shape& shape,
                           std::vector<double> values) {
  return new ConstantNode(std::move(name), shape, std::move(values));
}
inline GraphNode* Constant(std::string name, const Shape& shape,
                           int initializer_type, double initializer_param1,
                           double initializer_param2) {
  return new ConstantNode(std::move(name), shape, initializer_type,
                          initializer_param1, initializer_param2);
}

DEFINE_GRAPH_NODE_CREATOR(Zeros)
DEFINE_GRAPH_NODE_CREATOR(Ones)
DEFINE_GRAPH_NODE_CREATOR(RandomNormal)
DEFINE_GRAPH_NODE_CREATOR(RandomUniform)
DEFINE_GRAPH_NODE_CREATOR(ConstantLike)
DEFINE_GRAPH_NODE_CREATOR(ZerosLike)
DEFINE_GRAPH_NODE_CREATOR(OnesLike)
DEFINE_GRAPH_NODE_CREATOR(RandomNormalLike)
DEFINE_GRAPH_NODE_CREATOR(RandomUniformLike)
DEFINE_GRAPH_NODE_CREATOR(TFEmbeddingLookup)
DEFINE_GRAPH_NODE_CREATOR(EmbeddingLookup)
DEFINE_GRAPH_NODE_CREATOR(GroupEmbeddingLookup)
DEFINE_GRAPH_NODE_CREATOR(GroupEmbeddingLookup2)
DEFINE_GRAPH_NODE_CREATOR(GEMM)
DEFINE_GRAPH_NODE_CREATOR(BatchGEMM)
DEFINE_GRAPH_NODE_CREATOR(Matmul)
DEFINE_GRAPH_NODE_CREATOR(Matmul2)

// DEFINE_GRAPH_NODE_CREATOR(FullyConnect)
inline GraphNode* FullyConnect(std::string name, GraphNode* X, GraphNode* W) {
  return new FullyConnectNode(std::move(name), X, W);
}
inline GraphNode* FullyConnect(std::string name, GraphNode* X, GraphNode* W,
                               GraphNode* b) {
  return new FullyConnectNode(std::move(name), X, W, b);
}

DEFINE_GRAPH_NODE_CREATOR(TensorDot)
DEFINE_GRAPH_NODE_CREATOR(Inner)
DEFINE_GRAPH_NODE_CREATOR(Outer)

// DEFINE_GRAPH_NODE_CREATOR(AddN)
inline GraphNode* AddN(std::string name, std::vector<GraphNode*> X) {
  return new AddNNode(std::move(name), std::move(X));
}

// DEFINE_GRAPH_NODE_CREATOR(Concat)
inline GraphNode* Concat(std::string name, std::vector<GraphNode*> X,
                         int axis = -1) {
  return new ConcatNode(std::move(name), std::move(X), axis);
}

// DEFINE_GRAPH_NODE_CREATOR(Reshape)
// DEFINE_GRAPH_NODE_CREATOR(ReshapeFast)
inline GraphNode* Reshape(std::string name, GraphNode* X, const Shape& shape) {
  // ReshapeNode is deprecated by Reshape2Node.
  return new Reshape2Node(std::move(name), X, shape);
}
inline GraphNode* ReshapeFast(std::string name, GraphNode* X,
                              const Shape& shape) {
  // ReshapeFastNode is deprecated by Reshape2FastNode.
  return new Reshape2FastNode(std::move(name), X, shape);
}

DEFINE_GRAPH_NODE_CREATOR(Reshape2)
DEFINE_GRAPH_NODE_CREATOR(Reshape2Fast)
DEFINE_GRAPH_NODE_CREATOR(ReshapeZeroCopy)
DEFINE_GRAPH_NODE_CREATOR(ExpandDim)
DEFINE_GRAPH_NODE_CREATOR(ExpandDimFast)
DEFINE_GRAPH_NODE_CREATOR(ExpandDimZeroCopy)
DEFINE_GRAPH_NODE_CREATOR(Squeeze)
DEFINE_GRAPH_NODE_CREATOR(SqueezeFast)
DEFINE_GRAPH_NODE_CREATOR(SqueezeZeroCopy)
DEFINE_GRAPH_NODE_CREATOR(Transpose)
DEFINE_GRAPH_NODE_CREATOR(Subscript)
DEFINE_GRAPH_NODE_CREATOR(Subscript2)
DEFINE_GRAPH_NODE_CREATOR(SubscriptRange)
DEFINE_GRAPH_NODE_CREATOR(LayerNorm)
DEFINE_GRAPH_NODE_CREATOR(SequenceMask)
DEFINE_GRAPH_NODE_CREATOR(Where)

// DEFINE_GRAPH_NODE_CREATOR(Tile)
inline GraphNode* Tile(std::string name, GraphNode* X, int rep) {
  return new TileNode(std::move(name), X, rep);
}
inline GraphNode* Tile(std::string name, GraphNode* X, std::vector<int> reps) {
  return new TileNode(std::move(name), X, std::move(reps));
}

DEFINE_GRAPH_NODE_CREATOR(BatchCos)
DEFINE_GRAPH_NODE_CREATOR(BatchDot)
DEFINE_GRAPH_NODE_CREATOR(StopGrad)

// DEFINE_GRAPH_NODE_CREATOR(BatchNorm)
// use BatchNorm in graph_module_creator.h

}  // namespace deepx_core
