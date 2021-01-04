// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include "arithmetic_impl.h"
#include <algorithm>  // std::min_element
#include <deque>
#include <memory>
#include <set>
#include <utility>
#include "../op/kernel/broadcast.h"

namespace deepx_core {

/************************************************************************/
/* RewriteGroupedNodesBase */
/************************************************************************/
bool RewriteGroupedNodesBase::MaySimplify(const GraphNode* node) const
    noexcept {
  return CanBeRoot(node);
}

bool RewriteGroupedNodesBase::TrySimplify(GraphNode* node) {
  NodeGroup group;
  group.root = node;
  for (auto* input : group.root->input()) {
    TryAbsorbNodeToGroup(input, &group);
  }
  bool simplified = false;
  if (!group.non_root_members.empty()) {
    if (SimplifyGroup(&group)) {
      simplified = true;
    }
  }
  return simplified;
}

void RewriteGroupedNodesBase::TryAbsorbNodeToGroup(GraphNode* node,
                                                   NodeGroup* group) {
  std::deque<GraphNode*> to_process;
  to_process.emplace_back(node);
  while (!to_process.empty()) {
    GraphNode* on_process = to_process.front();
    to_process.pop_front();

    if (Absorbable(on_process, group)) {
      group->non_root_members.emplace_back(on_process);
      for (auto* input : on_process->input()) {
        to_process.emplace_back(input);
      }
    } else {
      group->inputs.emplace_back(on_process);
    }
  }
}

/************************************************************************/
/* RewriteGroupedAddStage */
/************************************************************************/
bool RewriteGroupedAddStage::CanBeRoot(const GraphNode* node) const noexcept {
  return CanBeMember(node);
}

bool RewriteGroupedAddStage::Absorbable(const GraphNode* node,
                                        const NodeGroup* /*group*/) const
    noexcept {
  if (!CanBeMember(node)) {
    return false;
  }
  return IsSingleOutput(node);
}

bool RewriteGroupedAddStage::CanBeMember(const GraphNode* node) const noexcept {
  return (node->type_index() == typeid(AddNode) ||
          node->type_index() == typeid(AddNNode)) &&
         !IsTarget(node);
}

bool RewriteGroupedAddStage::SimplifyGroup(NodeGroup* group) {
  std::string name = NewNodeName(group->root->name());
  std::vector<GraphNode*> inputs = group->inputs;
  SortByName(&inputs);
  GraphNode* node = AddN(name, inputs);
  ctx_->item->Add(node);
  ctx_->nodes_to_simp.PushBack(node);
  ctx_->item->ReplaceInputOfAllOutputs(group->root->name(), node->name());
  return true;
}

/************************************************************************/
/* RewriteGroupedBroadcastStage */
/************************************************************************/
bool RewriteGroupedBroadcastStage::CanBeRoot(const GraphNode* node) const
    noexcept {
  if (!CanBeMember(node)) {
    return false;
  }
  return !node->shape().empty();
}

bool RewriteGroupedBroadcastStage::Absorbable(const GraphNode* node,
                                              const NodeGroup* group) const
    noexcept {
  if (node->type_index() != group->root->type_index()) {
    return false;
  }
  if (!CanBeMember(node)) {
    return false;
  }
  return IsSingleOutput(node);
}

bool RewriteGroupedBroadcastStage::CanBeMember(const GraphNode* node) const
    noexcept {
  return (node->type_index() == typeid(BroadcastAddNode) ||
          node->type_index() == typeid(BroadcastMulNode)) &&
         !IsTarget(node);
}

bool RewriteGroupedBroadcastStage::CompareByShapeAndName(
    const GraphNode* a, const GraphNode* b) noexcept {
  const Shape& a_shape = a->shape();
  const Shape& b_shape = b->shape();
  if (a_shape.rank() != b_shape.rank()) {
    return a_shape.rank() < b_shape.rank();
  }
  for (int i = 0; i < a_shape.rank(); ++i) {
    if (a_shape[i] != b_shape[i]) {
      return a_shape[i] < b_shape[i];
    }
  }
  return a->name() < b->name();
}

bool RewriteGroupedBroadcastStage::CompareStrideMoveToTarget(
    const GraphNode* target, const GraphNode* a, const GraphNode* b) noexcept {
  int a_stride_move = CountStrideMoveInBroadcast(target, a);
  int b_stride_move = CountStrideMoveInBroadcast(target, b);
  return a_stride_move < b_stride_move;
}

int RewriteGroupedBroadcastStage::CountStrideMoveInBroadcast(
    const GraphNode* a, const GraphNode* b) noexcept {
  BroadcastAux aux;
  BroadcastPrepare(a->shape(), b->shape(), &aux);

  if (aux.XY_same_shape) {
    return 0;
  }

  int stride_move = 0;
  int leading = 1;
  for (int i = 0; i < aux.Z.rank(); ++i) {
    if (aux.Z[i] < 0) {
      continue;
    }
    if (i < aux.Z.rank() - 1 || !aux.vectorization) {
      leading *= aux.Z[i];
      if (aux.Xstrides[i]) {
        stride_move += leading;
      }
      if (aux.Ystrides[i]) {
        stride_move += leading;
      }
      if (aux.Zstrides[i]) {
        stride_move += leading;
      }
    }
  }
  return stride_move;
}

bool RewriteGroupedBroadcastStage::SimplifyGroup(NodeGroup* group) {
  bool is_add = group->root->type_index() == typeid(BroadcastAddNode);
  std::vector<GraphNode*> to_combine(group->inputs.begin(),
                                     group->inputs.end());
  SortByName(&to_combine);

  int old_stride_move =
      CountStrideMoveInBroadcast(group->root->input(0), group->root->input(1));
  for (auto* member : group->non_root_members) {
    old_stride_move +=
        CountStrideMoveInBroadcast(member->input(0), member->input(1));
  }
  int new_stride_move = 0;
  int new_broadcast = 0;
  std::vector<GraphNode*> new_broadcast_nodes;
  while ((int)to_combine.size() > 1) {
    auto min = std::min_element(to_combine.begin(), to_combine.end(),
                                CompareByShapeAndName);
    GraphNode* X = *min;
    to_combine.erase(min);

    auto closest =
        std::min_element(to_combine.begin(), to_combine.end(),
                         [X](const GraphNode* a, const GraphNode* b) {
                           return CompareStrideMoveToTarget(X, a, b);
                         });
    new_stride_move += CountStrideMoveInBroadcast(X, *closest);
    GraphNode* Y = *closest;
    to_combine.erase(closest);
    std::string new_node_name =
        NewNodeName(group->root->name(), std::to_string(new_broadcast));
    GraphNode* new_node;
    if (is_add) {
      new_node = BroadcastAdd(new_node_name, X, Y);
    } else {
      new_node = BroadcastMul(new_node_name, X, Y);
    }
    new_broadcast_nodes.emplace_back(new_node);
    ctx_->item->Add(new_node);
    to_combine.emplace_back(new_node);
    ++new_broadcast;
  }
  GraphNode* new_root = *to_combine.begin();
  if (new_stride_move < old_stride_move) {
    for (auto* node : new_broadcast_nodes) {
      ctx_->nodes_to_simp.PushBack(node);
    }
    ctx_->item->ReplaceInputOfAllOutputs(group->root->name(), new_root->name());
    return true;
  } else {
    return false;
  }
}

/************************************************************************/
/* RewritePowStage */
/************************************************************************/
bool RewritePowStage::MaySimplify(const GraphNode* node) const noexcept {
  if (node->type_index() != typeid(PowNode)) {
    return false;
  }
  const GraphNode* Y = node->input(1);
  if (Y->type_index() != typeid(ConstantNode) &&
      Y->type_index() != typeid(ZerosLikeNode) &&
      Y->type_index() != typeid(OnesLikeNode)) {
    return false;
  }
  return !IsTarget(node);
}

bool RewritePowStage::TrySimplify(GraphNode* node) {
  double exponent = 0;
  const GraphNode* Y = node->input(1);
  if (Y->type_index() == typeid(ZerosLikeNode)) {
    exponent = 0;
  } else if (Y->type_index() == typeid(OnesLikeNode)) {
    exponent = 1;
  } else {
    auto* _Y = (const ConstantNode*)Y;
    if (_Y->constant_type() == ConstantNode::CONSTANT_TYPE_VALUE) {
      exponent = _Y->value();
    } else if (_Y->constant_type() == ConstantNode::CONSTANT_TYPE_VALUES) {
      const std::vector<double>& values = _Y->values();
      exponent = values[0];
      for (int i = 1; i < (int)values.size(); ++i) {
        if (exponent != values[i]) {
          return false;
        }
      }
    } else {
      return false;
    }
  }

  GraphNode* X = node->input()[0];
  GraphNode* new_node = nullptr;
  std::string name = NewNodeName(node->name());
  if (exponent == 3) {
    new_node = Cubic(name, X);
  } else if (exponent == 2) {
    new_node = Square(name, X);
  } else if (exponent == 1) {
    new_node = X;
  } else if (exponent == 0.5) {
    new_node = Sqrt(name, X);
  } else if (exponent == 0) {
    new_node = OnesLike(name, X);
  } else if (exponent == -1) {
    new_node = Reciprocal(name, X);
  } else {
    return false;
  }

  ctx_->item->Add(new_node);
  ctx_->nodes_to_simp.PushBack(new_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_node->name());
  return true;
}

/************************************************************************/
/* RewriteMaxOrMinOfMonotonicStage */
/************************************************************************/
const std::vector<std::type_index>
    RewriteMaxOrMinOfMonotonicStage::MONOTONIC_NON_INCREASING_TYPEIDS_ = {
        typeid(NegateNode), typeid(InvNode)};

const std::vector<std::type_index>
    RewriteMaxOrMinOfMonotonicStage::MONOTONIC_NON_DECREASING_TYPEIDS_ = {
        typeid(SigmoidNode),   typeid(TanhNode),     typeid(ReluNode),
        typeid(LeakyReluNode), typeid(EluNode),      typeid(SeluNode),
        typeid(GeluNode),      typeid(SoftPlusNode), typeid(SwishNode),
        typeid(ExpNode),       typeid(LogNode),      typeid(SqrtNode),
        typeid(CbrtNode),      typeid(SquareNode),   typeid(CubicNode),
        typeid(SignNode)};

bool RewriteMaxOrMinOfMonotonicStage::IsMaxPool(const GraphNode* node) const
    noexcept {
  const std::type_index& node_type_id = node->type_index();
  return node_type_id == typeid(MaxPool1dNode) ||
         node_type_id == typeid(MaxPool2dNode) ||
         node_type_id == typeid(MaxPool3dNode);
}
bool RewriteMaxOrMinOfMonotonicStage::IsMonotonicNonIncreasing(
    const GraphNode* node) const noexcept {
  for (auto& type_id : MONOTONIC_NON_INCREASING_TYPEIDS_) {
    if (node->type_index() == type_id) {
      return true;
    }
  }
  return false;
}

bool RewriteMaxOrMinOfMonotonicStage::IsMonotonicNonDecreasing(
    const GraphNode* node) const noexcept {
  for (auto& type_id : MONOTONIC_NON_DECREASING_TYPEIDS_) {
    if (node->type_index() == type_id) {
      return true;
    }
  }
  return false;
}

bool RewriteMaxOrMinOfMonotonicStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (IsTarget(node)) {
    return false;
  }
  const std::type_index& node_type_id = node->type_index();
  if (node_type_id != typeid(ArgMaxNode) &&
      node_type_id != typeid(ArgMinNode) &&
      node_type_id != typeid(ReduceMaxNode) &&
      node_type_id != typeid(ReduceMinNode) && !IsMaxPool(node)) {
    return false;
  }
  const GraphNode* X = node->input(0);
  if (IsMaxPool(node)) {
    if (IsMonotonicNonIncreasing(X)) {
      return false;
    }
  }
  if (!IsMonotonicNonIncreasing(X) && !IsMonotonicNonDecreasing(X)) {
    return false;
  }
  return !IsTarget(X) && IsSingleOutput(X);
}

bool RewriteMaxOrMinOfMonotonicStage::TrySimplify(GraphNode* node) {
  const std::type_index& node_type_id = node->type_index();
  GraphNode* input = node->input()[0];
  GraphNode* _input = input->input()[0];
  bool is_arg = false;
  GraphNode* new_node = nullptr;
  if (IsMonotonicNonDecreasing(input)) {
    ctx_->item->ReplaceInput(node->name(), input->name(), _input->name());
    if (node_type_id == typeid(ArgMaxNode) ||
        node_type_id == typeid(ArgMinNode)) {
      return true;
    }
    new_node = node;
  } else {
    std::string node_name = NewNodeName(node->name());
    if (node_type_id == typeid(ArgMaxNode)) {
      is_arg = true;
      int axis = ((ArgMaxNode*)node)->axis();
      new_node = ArgMin(node_name, _input, axis);
    } else if (node_type_id == typeid(ArgMinNode)) {
      is_arg = true;
      int axis = ((ArgMinNode*)node)->axis();
      new_node = ArgMax(node_name, _input, axis);
    } else if (node_type_id == typeid(ReduceMaxNode)) {
      int axis = ((ReduceMaxNode*)node)->axis();
      int keep_dim = ((ReduceMaxNode*)node)->keep_dim();
      new_node = ReduceMin(node_name, _input, axis, keep_dim);
    } else if (node_type_id == typeid(ReduceMinNode)) {
      int axis = ((ReduceMinNode*)node)->axis();
      int keep_dim = ((ReduceMinNode*)node)->keep_dim();
      new_node = ReduceMax(node_name, _input, axis, keep_dim);
    } else {
      return false;
    }
  }
  ctx_->item->Add(new_node);
  ctx_->nodes_to_simp.PushBack(new_node);
  ctx_->item->ReplaceInputOfAllOutputs(
      node->name(), is_arg ? new_node->name() : input->name());
  if (!is_arg) {
    ctx_->item->ReplaceInput(input->name(), _input->name(), new_node->name());
  }
  return true;
}

/************************************************************************/
/* RewriteAggregatableAddNStage */
/************************************************************************/
bool RewriteAggregatableAddNStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(AddNNode)) {
    return false;
  }
  return !IsTarget(node);
}

bool RewriteAggregatableAddNStage::TrySimplify(GraphNode* node) {
  std::unordered_map<GraphNode*, int> aggregatable;
  GetAggregatableInputs(node, &aggregatable);
  if (aggregatable.empty()) {
    return false;
  }

  std::vector<GraphNode*> new_inputs;
  for (auto* input : node->input()) {
    if (aggregatable.count(input) == 0) {
      new_inputs.emplace_back(input);
    }
  }
  for (auto& entry : aggregatable) {
    GraphNode* input = entry.first;
    int count = entry.second;
    const std::string& name = input->name();
    auto* constant =
        Constant(NewNodeName(name, "constant"), Shape(1), (double)count);
    auto* broadcastmul =
        BroadcastMul(NewNodeName(name, "broadcastmul"), input, constant);
    ctx_->item->Add(constant);
    ctx_->nodes_to_simp.PushBack(constant);
    ctx_->item->Add(broadcastmul);
    ctx_->nodes_to_simp.PushBack(broadcastmul);
    new_inputs.emplace_back(broadcastmul);
  }

  GraphNode* new_node = nullptr;
  if ((int)new_inputs.size() == 1) {
    new_node = new_inputs[0];
  } else {
    SortByName(&new_inputs);
    new_node = AddN(NewNodeName(node->name()), new_inputs);
    ctx_->item->Add(new_node);
    ctx_->nodes_to_simp.PushBack(new_node);
  }
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_node->name());
  return true;
}

void RewriteAggregatableAddNStage::GetAggregatableInputs(
    GraphNode* node, std::unordered_map<GraphNode*, int>* inputs) {
  inputs->clear();

  std::unordered_map<std::string, int> input_counts;
  for (auto& input : node->input()) {
    ++input_counts[input->name()];
  }
  if ((int)input_counts.size() == 1) {
    inputs->emplace(node->input()[0], node->input_size());
  } else {
    for (auto& entry : input_counts) {
      // operation speed: 1 addition > 1 multiply  > 2 addition.
      // input N nodes, using AddN operation needs N addition operations.
      // if dup of N nodes are the same node, replaced by
      // BroadcastMul(Mul) needs (N-dup+1) addition operations and 1 multi
      // operation, this stage can speed up initial AddN operation when dup
      // >= 3.
      if (entry.second >= 3) {
        inputs->emplace(ctx_->item->find_node(entry.first), entry.second);
      }
    }
  }
}

/************************************************************************/
/* RewriteSquareMulStage */
/************************************************************************/
bool RewriteSquareMulStage::MaySimplify(const GraphNode* node) const noexcept {
  if (node->type_index() != typeid(MulNode)) {
    return false;
  }
  const GraphNode* left = node->input(0);
  const GraphNode* right = node->input(1);
  return (left == right) && !IsTarget(node);
}

bool RewriteSquareMulStage::TrySimplify(GraphNode* node) {
  GraphNode* left = node->input()[0];
  auto* new_node = Square(NewNodeName(left->name()), left);
  ctx_->item->Add(new_node);
  ctx_->nodes_to_simp.PushBack(new_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_node->name());
  return true;
}

/************************************************************************/
/* RewriteCubicMulStage */
/************************************************************************/
bool RewriteCubicMulStage::MaySimplify(const GraphNode* node) const noexcept {
  if (node->type_index() != typeid(MulNode)) {
    return false;
  }
  return !IsTarget(node);
}

bool RewriteCubicMulStage::TrySimplify(GraphNode* node) {
  auto is_modifiable = [this](const GraphNode* node) {
    return !IsTarget(node) && IsSingleOutput(node);
  };

  GraphNode* X = node->input()[0];
  GraphNode* Y = node->input()[1];
  GraphNode* new_node = nullptr;
  bool modifiable_X = is_modifiable(X);
  bool modifiable_Y = is_modifiable(Y);
  std::unordered_map<std::string, int> candidate_factor_counts;
  for (auto* input : node->input()) {
    if (input->type_index() == typeid(MulNode)) {
      for (auto* _input : input->input()) {
        candidate_factor_counts[_input->name()] += 1;
      }
    } else if (input->type_index() == typeid(SquareNode)) {
      GraphNode* _input = input->input()[0];
      candidate_factor_counts[_input->name()] += 2;
    }
  }

  GraphNode* cubic_input = nullptr;
  GraphNode* extra_input = nullptr;
  if (candidate_factor_counts[X->name()] == 2 && modifiable_Y) {
    /*     node
     *    /   \
     *   A   mul/square
     *       /   \
     *      A ... A
     */
    cubic_input = X;
  } else if (candidate_factor_counts[Y->name()] == 2 && modifiable_X) {
    /*           node
     *           /  \
     *  mul/square   A
     *       /   \
     *      A ... A
     */
    cubic_input = Y;
  } else if (modifiable_X && modifiable_Y) {
    /*
     *          node
     *          /  \
     *  mul/square mul/square
     *
     * factor_counts : {A: 4} -> cubic_input = extra_input = A
     * factor_counts : {A: 3, B : 1} -> cubic_input = A, extra_input = B
     * other input_counts are invalid.
     */
    for (auto& entry : candidate_factor_counts) {
      int count = entry.second;
      GraphNode* input = ctx_->item->find_node(entry.first);
      if (count == 4) {
        cubic_input = input;
        extra_input = input;
      } else if (count == 3) {
        cubic_input = input;
      } else if (count == 1) {
        extra_input = input;
      }
    }
  }

  if (!cubic_input) {
    return false;
  }

  auto* new_cubic_node =
      Cubic(NewNodeName(cubic_input->name(), "cubic"), cubic_input);
  ctx_->item->Add(new_cubic_node);
  ctx_->nodes_to_simp.PushBack(new_cubic_node);

  if (extra_input) {
    new_node = Mul(NewNodeName(extra_input->name(), "mul"), extra_input,
                   new_cubic_node);
    ctx_->item->Add(new_node);
    ctx_->nodes_to_simp.PushBack(new_node);
  } else {
    new_node = new_cubic_node;
  }
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_node->name());
  return true;
}

/************************************************************************/
/* RewriteNegateStage */
/************************************************************************/
bool RewriteNegateStage::MaySimplify(const GraphNode* node) const noexcept {
  if (node->type_index() != typeid(AddNode) &&
      node->type_index() != typeid(SubNode)) {
    return false;
  }
  return !IsTarget(node);
}

bool RewriteNegateStage::TrySimplify(GraphNode* node) {
  GraphNode* X = node->input()[0];
  GraphNode* Y = node->input()[1];
  bool rewritable_negate_X = IsRewritableNegate(X);
  bool rewritable_negate_Y = IsRewritableNegate(Y);

  std::string node_name = NewNodeName(node->name());
  GraphNode* new_node = nullptr;
  if (node->type_index() == typeid(AddNode)) {
    if (rewritable_negate_X) {
      new_node = Sub(node_name, Y, X->input()[0]);
    } else if (rewritable_negate_Y) {
      new_node = Sub(node_name, X, Y->input()[0]);
    } else {
      return false;
    }
  } else {  // SubNode
    if (rewritable_negate_X && rewritable_negate_Y) {
      new_node = Sub(node_name, Y->input()[0], X->input()[0]);
    } else if (rewritable_negate_Y) {
      new_node = Add(node_name, X, Y->input()[0]);
    } else {
      return false;
    }
  }
  ctx_->item->Add(new_node);
  ctx_->nodes_to_simp.PushBack(new_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_node->name());
  return true;
}

bool RewriteNegateStage::IsRewritableNegate(const GraphNode* node) const
    noexcept {
  return (node->type_index() == typeid(NegateNode)) && !IsTarget(node) &&
         IsSingleOutput(node);
}

/************************************************************************/
/* RewriteInvStage */
/************************************************************************/
bool RewriteInvStage::MaySimplify(const GraphNode* node) const noexcept {
  if (node->type_index() != typeid(MulNode) &&
      node->type_index() != typeid(DivNode)) {
    return false;
  }
  return !IsTarget(node);
}

bool RewriteInvStage::TrySimplify(GraphNode* node) {
  GraphNode* X = node->input()[0];
  GraphNode* Y = node->input()[1];
  bool rewritable_inv_X = IsRewritableInv(X);
  bool rewritable_inv_Y = IsRewritableInv(Y);

  std::string node_name = NewNodeName(node->name());
  GraphNode* new_node = nullptr;
  if (node->type_index() == typeid(MulNode)) {
    if (rewritable_inv_X) {
      new_node = Div(node_name, Y, X->input()[0]);
    } else if (rewritable_inv_Y) {
      new_node = Div(node_name, X, Y->input()[0]);
    } else {
      return false;
    }
  } else {  // DivNode
    if (rewritable_inv_X && rewritable_inv_Y) {
      new_node = Div(node_name, Y->input()[0], X->input()[0]);
    } else if (rewritable_inv_Y) {
      new_node = Mul(node_name, X, Y->input()[0]);
    } else {
      return false;
    }
  }
  ctx_->item->Add(new_node);
  ctx_->nodes_to_simp.PushBack(new_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_node->name());
  return true;
}

bool RewriteInvStage::IsRewritableInv(const GraphNode* node) const noexcept {
  return (node->type_index() == typeid(InvNode)) && !IsTarget(node) &&
         IsSingleOutput(node);
}

/************************************************************************/
/* RewriteSuccessiveReshapeStage */
/************************************************************************/
bool RewriteSuccessiveReshapeStage::MaySimplify(const GraphNode* node) const
    noexcept {
  return node->type_index() == typeid(Reshape2Node);
}

bool RewriteSuccessiveReshapeStage::TrySimplify(GraphNode* node) {
  const GraphNode* input = node->input(0);
  while (IsRewritableReshape(input)) {
    input = input->input(0);
  }
  if (input == node->input(0)) {
    return false;
  } else {
    ctx_->item->ReplaceInput(node->name(), node->input(0)->name(),
                             input->name());
  }
  return true;
}

bool RewriteSuccessiveReshapeStage::IsRewritableReshape(
    const GraphNode* node) const noexcept {
  return (node->type_index() == typeid(Reshape2Node)) && !IsTarget(node) &&
         IsSingleOutput(node);
}

/************************************************************************/
/* FuseTransposeIntoMatmul */
/************************************************************************/
namespace {

bool IsInnerMatrixTranspose(const TransposeNode* node) noexcept {
  const Shape& axes = node->axes();
  int rank = axes.rank();
  if (rank < 2) {
    return false;
  }
  for (int i = 0; i < rank - 2; ++i) {
    if (axes[i] != i) {
      return false;
    }
  }
  return axes[rank - 2] == rank - 1 && axes[rank - 1] == rank - 2;
}

}  // namespace

bool FuseTransposeIntoMatmulOrGEMMStage::MaySimplify(
    const GraphNode* node) const noexcept {
  if ((node->type_index() != typeid(MatmulNode) &&
       node->type_index() != typeid(Matmul2Node) &&
       node->type_index() != typeid(GEMMNode) &&
       node->type_index() != typeid(BatchGEMMNode)) ||
      IsTarget(node)) {
    return false;
  }
  const GraphNode* X = node->input(0);
  const GraphNode* Y = node->input(1);
  return (X->type_index() == typeid(TransposeNode) &&
          IsInnerMatrixTranspose((const TransposeNode*)X)) ||
         (Y->type_index() == typeid(TransposeNode) &&
          IsInnerMatrixTranspose((const TransposeNode*)Y));
}

bool FuseTransposeIntoMatmulOrGEMMStage::TrySimplify(GraphNode* node) {
  GraphNode* X = node->input()[0];
  GraphNode* Y = node->input()[1];
  bool is_X_fusible = X->type_index() == typeid(TransposeNode) &&
                      IsInnerMatrixTranspose((const TransposeNode*)X);
  bool is_Y_fusible = Y->type_index() == typeid(TransposeNode) &&
                      IsInnerMatrixTranspose((const TransposeNode*)Y);
  GraphNode* new_X = is_X_fusible ? X->input()[0] : X;
  GraphNode* new_Y = is_Y_fusible ? Y->input()[0] : Y;
  int transX, transY;
  GraphNode* new_node = nullptr;
  if (node->type_index() == typeid(MatmulNode)) {
    transX = (int)is_X_fusible;
    transY = (int)is_Y_fusible;
    new_node = Matmul2(NewNodeName(node->name()), new_X, new_Y, transX, transY);
  } else if (node->type_index() == typeid(Matmul2Node)) {
    transX = is_X_fusible ? 1 - ((Matmul2Node*)node)->transX()
                          : ((Matmul2Node*)node)->transX();
    transY = is_Y_fusible ? 1 - ((Matmul2Node*)node)->transY()
                          : ((Matmul2Node*)node)->transY();
    new_node = Matmul2(NewNodeName(node->name()), new_X, new_Y, transX, transY);
  } else if (node->type_index() == typeid(GEMMNode)) {
    transX = is_X_fusible ? 1 - ((GEMMNode*)node)->transX()
                          : ((GEMMNode*)node)->transX();
    transY = is_Y_fusible ? 1 - ((GEMMNode*)node)->transY()
                          : ((GEMMNode*)node)->transY();
    new_node = GEMM(NewNodeName(node->name()), new_X, new_Y, transX, transY);
  } else {  // BatchGEMMNode
    transX = is_X_fusible ? 1 - ((BatchGEMMNode*)node)->transX()
                          : ((BatchGEMMNode*)node)->transX();
    transY = is_Y_fusible ? 1 - ((BatchGEMMNode*)node)->transY()
                          : ((BatchGEMMNode*)node)->transY();
    new_node =
        BatchGEMM(NewNodeName(node->name()), new_X, new_Y, transX, transY);
  }
  ctx_->item->Add(new_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_node->name());
  return true;
}

/************************************************************************/
/* RemoveInvolutionBase */
/************************************************************************/
const std::vector<std::type_index>
    RemoveInvolutionBase::VALUE_PRESERVING_TYPEIDS_ = {
        typeid(Reshape2Node), typeid(ExpandDimNode), typeid(SqueezeNode),
        typeid(TransposeNode)};

bool RemoveInvolutionBase::TrySimplify(GraphNode* node) {
  if (!IsRemovableValuePreserving(node->input(0))) {
    const GraphNode* input = node->input(0);
    if (MaySimplify(input) && IsSingleOutput(input)) {
      ctx_->item->ReplaceInputOfAllOutputs(node->name(),
                                           input->input(0)->name());
      return true;
    } else {
      return false;
    }
  }

  const GraphNode* preserving = node->input(0);
  while (IsRemovableValuePreserving(preserving->input(0))) {
    preserving = preserving->input(0);
  }

  const GraphNode* preserving_input = preserving->input(0);
  if (!MaySimplify(preserving_input) || !IsSingleOutput(preserving_input)) {
    return false;
  }

  ctx_->item->ReplaceInput(preserving->name(), preserving_input->name(),
                           preserving_input->input(0)->name());
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), node->input(0)->name());
  return true;
}

bool RemoveInvolutionBase::IsRemovableValuePreserving(
    const GraphNode* node) const noexcept {
  if (IsTarget(node) || !IsSingleOutput(node)) {
    return false;
  }
  for (auto& type_id : VALUE_PRESERVING_TYPEIDS_) {
    if (node->type_index() == type_id) {
      return true;
    }
  }
  return false;
}

/************************************************************************/
/* RemoveInvInvolutionStage */
/************************************************************************/
bool RemoveInvInvolutionStage::MaySimplify(const GraphNode* node) const
    noexcept {
  return node->type_index() == typeid(InvNode) && !IsTarget(node);
}

/************************************************************************/
/* RemoveNegateInvolutionStage */
/************************************************************************/
bool RemoveNegateInvolutionStage::MaySimplify(const GraphNode* node) const
    noexcept {
  return node->type_index() == typeid(NegateNode) && !IsTarget(node);
}

/************************************************************************/
/* RemoveIneffectiveAdjacentTransposeStage */
/************************************************************************/
bool RemoveIneffectiveAdjacentTransposeStage::MaySimplify(
    const GraphNode* node) const noexcept {
  if (node->type_index() != typeid(TransposeNode) || IsTarget(node)) {
    return false;
  }
  const GraphNode* input = node->input(0);
  if (input->type_index() != typeid(TransposeNode) || IsTarget(input) ||
      !IsSingleOutput(input)) {
    return false;
  }
  const Shape& axes1 = ((const TransposeNode*)node)->axes();
  const Shape& axes2 = ((const TransposeNode*)input)->axes();
  if (axes1.rank() != axes2.rank()) {
    return false;
  }
  for (int i = 0; i < axes1.rank(); ++i) {
    if (axes1[axes2[i]] != i) {
      return false;
    }
  }
  return true;
}

bool RemoveIneffectiveAdjacentTransposeStage::TrySimplify(GraphNode* node) {
  ctx_->item->ReplaceInputOfAllOutputs(node->name(),
                                       node->input(0)->input(0)->name());
  return true;
}

/************************************************************************/
/* RemoveIdempotentStage */
/************************************************************************/
bool RemoveIdempotentStage::MaySimplify(const GraphNode* node) const noexcept {
  if ((node->type_index() != typeid(IdentityNode)) &&
      (node->type_index() != typeid(StopGradNode))) {
    return false;
  }
  return !IsTarget(node);
}

bool RemoveIdempotentStage::TrySimplify(GraphNode* node) {
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), node->input(0)->name());
  return true;
}

/************************************************************************/
/* HoistCommonFactorOutOfAggregationStage */
/************************************************************************/
bool HoistCommonFactorOutOfAggregationStage::MaySimplify(
    const GraphNode* node) const noexcept {
  if ((node->type_index() != typeid(AddNode)) &&
      (node->type_index() != typeid(AddNNode))) {
    return false;
  }
  if (IsTarget(node)) {
    return false;
  }

  if (node->input_size() == 1) {
    return false;
  }
  for (auto* input : node->input()) {
    if (input->type_index() != typeid(MulNode) &&
        input->type_index() != typeid(SquareNode)) {
      return false;
    }
    if (IsTarget(input) || !IsSingleOutput(input)) {
      return false;
    }
  }
  return true;
}

bool HoistCommonFactorOutOfAggregationStage::TrySimplify(GraphNode* node) {
  const GraphNode* X0 = node->input(0);
  std::set<GraphNode*> commons(X0->input().begin(), X0->input().end());
  for (int i = 1; i < node->input_size(); ++i) {
    std::set<GraphNode*> intersection;
    for (auto* factor : node->input(i)->input()) {
      if (commons.count(factor) > 0) {
        intersection.insert(factor);
      }
    }
    if (intersection.empty()) {
      return false;
    }
    commons.swap(intersection);
  }

  GraphNode* common =
      commons.count(X0->input()[0]) ? X0->input()[0] : X0->input()[1];
  std::vector<GraphNode*> new_add_inputs;
  for (auto* input : node->input()) {
    GraphNode* other;
    if (input->type_index() == typeid(SquareNode) ||
        input->input(1) == common) {
      other = input->input()[0];
    } else {
      other = input->input()[1];
    }
    new_add_inputs.emplace_back(other);
  }
  SortByName(&new_add_inputs);

  GraphNode* add_node = nullptr;
  if ((int)new_add_inputs.size() == 2) {
    add_node = Add(NewNodeName(node->name(), "add"), new_add_inputs[0],
                   new_add_inputs[1]);
  } else {
    add_node = AddN(NewNodeName(node->name(), "addn"), new_add_inputs);
  }
  ctx_->item->Add(add_node);
  ctx_->nodes_to_simp.PushBack(add_node);

  GraphNode* mul_node = Mul(NewNodeName(node->name(), "mul"), add_node, common);
  ctx_->item->Add(mul_node);
  ctx_->nodes_to_simp.PushBack(mul_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), mul_node->name());
  return true;
}

/************************************************************************/
/* HoistCommonDenominatorOutOfAggregationStage */
/************************************************************************/
bool HoistCommonDenominatorOutOfAggregationStage::MaySimplify(
    const GraphNode* node) const noexcept {
  if ((node->type_index() != typeid(AddNode)) &&
      (node->type_index() != typeid(AddNNode))) {
    return false;
  }
  if (IsTarget(node)) {
    return false;
  }

  if (node->input_size() == 1) {
    return false;
  }
  for (auto* input : node->input()) {
    if (input->type_index() != typeid(DivNode) || IsTarget(input) ||
        !IsSingleOutput(input)) {
      return false;
    }
  }
  return true;
}

bool HoistCommonDenominatorOutOfAggregationStage::TrySimplify(GraphNode* node) {
  GraphNode* denominator = node->input(0)->input()[1];
  for (auto* input : node->input()) {
    if (denominator != input->input()[1]) {
      return false;
    }
  }

  std::vector<GraphNode*> new_add_inputs;
  for (auto* input : node->input()) {
    new_add_inputs.emplace_back(input->input()[0]);
  }
  SortByName(&new_add_inputs);

  GraphNode* add_node = nullptr;
  if ((int)new_add_inputs.size() == 2) {
    add_node = Add(NewNodeName(node->name(), "add"), new_add_inputs[0],
                   new_add_inputs[1]);
  } else {
    add_node = AddN(NewNodeName(node->name(), "addn"), new_add_inputs);
  }
  ctx_->item->Add(add_node);
  ctx_->nodes_to_simp.PushBack(add_node);

  GraphNode* div_node =
      Div(NewNodeName(node->name(), "div"), add_node, denominator);
  ctx_->item->Add(div_node);
  ctx_->nodes_to_simp.PushBack(div_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), div_node->name());
  return true;
}

/************************************************************************/
/* ArithmeticSimp */
/************************************************************************/
ArithmeticSimp::ArithmeticSimp() : Simp("arithmetic") {}

bool ArithmeticSimp::Simplify(SimpItem* mutable_item) const {
  SimpContext ctx;
  std::vector<std::unique_ptr<SimpStage>> stages;
  stages.emplace_back(new RewriteGroupedAddStage(name(), &ctx));
  stages.emplace_back(new RewriteGroupedBroadcastStage(name(), &ctx));
  stages.emplace_back(new RewritePowStage(name(), &ctx));
  stages.emplace_back(new RewriteMaxOrMinOfMonotonicStage(name(), &ctx));
  stages.emplace_back(new RewriteAggregatableAddNStage(name(), &ctx));
  stages.emplace_back(new RewriteSquareMulStage(name(), &ctx));
  stages.emplace_back(new RewriteCubicMulStage(name(), &ctx));
  stages.emplace_back(new RewriteNegateStage(name(), &ctx));
  stages.emplace_back(new RewriteInvStage(name(), &ctx));
  stages.emplace_back(new RewriteSuccessiveReshapeStage(name(), &ctx));
  stages.emplace_back(new FuseTransposeIntoMatmulOrGEMMStage(name(), &ctx));
  stages.emplace_back(new RemoveInvInvolutionStage(name(), &ctx));
  stages.emplace_back(new RemoveNegateInvolutionStage(name(), &ctx));
  stages.emplace_back(
      new RemoveIneffectiveAdjacentTransposeStage(name(), &ctx));
  stages.emplace_back(new RemoveIdempotentStage(name(), &ctx));
  stages.emplace_back(new HoistCommonFactorOutOfAggregationStage(name(), &ctx));
  stages.emplace_back(
      new HoistCommonDenominatorOutOfAggregationStage(name(), &ctx));
  SimpPipeline pipeline(&ctx, std::move(stages));

  ctx.Init(mutable_item);
  bool simplified = false;
  while (!ctx.nodes_to_simp.Empty()) {
    GraphNode* node = ctx.nodes_to_simp.PopBack();
    if (pipeline.TrySimplify(node)) {
      simplified = true;
    }
  }
  ctx.item->Prune();

  return simplified;
}

}  // namespace deepx_core
