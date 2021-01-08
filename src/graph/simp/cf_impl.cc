// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "cf_impl.h"
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/graph/tensor_map.h>
#include <deque>
#include <memory>
#include <typeindex>
#include <utility>
#include <vector>

namespace deepx_core {
namespace {

GraphNode* RunGraphToGetNewConstant(GraphNode* target,
                                    const std::string& name) {
  Graph graph;
  TensorMap param;
  DXCHECK_THROW(graph.Compile({target}, 0));

  OpContext op_context;
  op_context.Init(&graph, &param);
  DXCHECK_THROW(op_context.InitOp(std::vector<int>{0}, -1));
  op_context.InitForward();
  op_context.Forward();

  auto& Z = op_context.hidden().get<DataType::tsr_t>(target->name());
  std::vector<double> values(Z.begin(), Z.end());
  return Constant(name, target->shape(), values);
}

bool IsFoldableConstantNode(const GraphNode* node) noexcept {
  return node->type_index() == typeid(ConstantNode) &&
         (((const ConstantNode*)node)->constant_type() ==
              ConstantNode::CONSTANT_TYPE_VALUE ||
          ((const ConstantNode*)node)->constant_type() ==
              ConstantNode::CONSTANT_TYPE_VALUES);
}

}  // namespace

/************************************************************************/
/* GraphFolding */
/************************************************************************/
GraphFolding::GraphFolding(const CFConfig& config) : config_(config) {}

bool GraphFolding::FoldGraph(SimpItem* item) {
  std::vector<GraphNode*> nodes;
  item->GetTopologicalSortedNodes(&nodes);

  std::deque<GraphNode*> to_process;
  for (auto* node : nodes) {
    if (IsFoldable(node, item)) {
      to_process.emplace_back(node);
    }
  }

  bool folded = false;
  while (!to_process.empty()) {
    GraphNode* node = to_process.front();
    to_process.pop_front();

    if (!FoldNode(node, item)) {
      continue;
    }
    folded = true;
    for (auto* output : item->find_output(node->name())) {
      if (IsFoldable(output, item)) {
        to_process.emplace_back(output);
      }
    }
  }
  return folded;
}

bool GraphFolding::IsFoldable(const GraphNode* node, const SimpItem* item) const
    noexcept {
  if (node->type_index() == typeid(ConstantNode)) {
    return false;
  }
  if (item->is_target(node->name())) {
    return false;
  }

  if (node->input_size() == 0) {
    return false;
  }
  for (auto* input : node->input()) {
    if (!IsFoldableConstantNode(input)) {
      return false;
    }
  }

  int output_bytes =
      (int)(sizeof(DataType::float_t) * node->shape().total_dim());
  return output_bytes <= config_.max_constant_bytes;
}

bool GraphFolding::FoldNode(GraphNode* node, SimpItem* item) {
  GraphNode* constant = RunGraphToGetNewConstant(
      node,
      item->NewNodeName(node->name(), {"constant_folding", "fold_graph"}));
  item->Add(constant);
  item->ReplaceInputOfAllOutputs(node->name(), constant->name());
  return true;
}

/************************************************************************/
/* RemoveIdenticalNodeBase */
/************************************************************************/
bool RemoveIdenticalNodeBase::TrySimplify(GraphNode* node) {
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), node->input(0)->name());
  return true;
}

/************************************************************************/
/* RemoveIdenticalTransposeStage */
/************************************************************************/
bool RemoveIdenticalTransposeStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(TransposeNode) || IsTarget(node)) {
    return false;
  }
  const auto* transpose = (const TransposeNode*)node;
  const Shape& axes = transpose->axes();
  for (int i = 0; i < axes.rank(); ++i) {
    if (i != axes[i]) {
      return false;
    }
  }
  return true;
}

/************************************************************************/
/* RemoveIdenticalTileStage */
/************************************************************************/
bool RemoveIdenticalTileStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(TileNode) || IsTarget(node)) {
    return false;
  }
  const auto* tile = (const TileNode*)node;
  for (auto rep : tile->reps()) {
    if (rep != 1) {
      return false;
    }
  }
  return true;
}

/************************************************************************/
/* RemoveIdenticalSubscriptRangeStage */
/************************************************************************/
bool RemoveIdenticalSubscriptRangeStage::MaySimplify(
    const GraphNode* node) const noexcept {
  if (node->type_index() != typeid(SubscriptRangeNode) || IsTarget(node)) {
    return false;
  }
  const auto* subscript_range = (const SubscriptRangeNode*)node;
  const GraphNode* X = subscript_range->input(0);
  int axis = subscript_range->axis();
  int begin_index = subscript_range->begin_index();
  int end_index = subscript_range->end_index();
  return begin_index == 0 && end_index == X->shape()[axis];
}

/************************************************************************/
/* RemoveIdenticalReshapeStage */
/************************************************************************/
bool RemoveIdenticalReshapeStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(Reshape2Node) || IsTarget(node)) {
    return false;
  }

  const auto* reshape2 = (const Reshape2Node*)node;
  const Shape& new_shape = reshape2->new_shape();
  const Shape& input_shape = reshape2->input(0)->shape();
  if (new_shape.rank() != input_shape.rank()) {
    return false;
  }
  for (int i = 0; i < new_shape.rank(); ++i) {
    int dim = new_shape[i];
    int input_dim = input_shape[i];
    if (dim > 0 && input_dim > 0 && dim != input_dim) {
      return false;
    }
  }
  return true;
}

/************************************************************************/
/* DivToReciprocalMulFoldingStage */
/************************************************************************/
bool DivToReciprocalMulFoldingStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(DivNode) || IsTarget(node)) {
    return false;
  }
  const GraphNode* Y = node->input(1);
  return IsFoldableConstantNode(Y) && !IsTarget(Y) && IsSingleOutput(Y);
}

bool DivToReciprocalMulFoldingStage::TrySimplify(GraphNode* node) {
  GraphNode* X = node->input()[0];
  auto* Y = (ConstantNode*)(node->input()[1]);
  GraphNode* inv_Y = nullptr;
  if (Y->constant_type() == ConstantNode::CONSTANT_TYPE_VALUE) {
    inv_Y = Constant(NewNodeName(Y->name()), Y->shape(), 1 / Y->value());
  } else {
    std::vector<double> values;
    values.reserve(Y->values().size());
    for (auto v : Y->values()) {
      values.emplace_back(1 / v);
    }
    inv_Y = Constant(NewNodeName(Y->name()), Y->shape(), values);
  }
  ctx_->item->Add(inv_Y);
  ctx_->nodes_to_simp.PushBack(inv_Y);

  GraphNode* new_mul_node = Mul(NewNodeName(node->name()), X, inv_Y);
  ctx_->item->Add(new_mul_node);
  ctx_->nodes_to_simp.PushBack(new_mul_node);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_mul_node->name());
  return true;
}

/************************************************************************/
/* PartialAddNFoldingStage */
/************************************************************************/
bool PartialAddNFoldingStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(AddNNode) || IsTarget(node)) {
    return false;
  }
  int foldable = 0;
  for (auto* input : node->input()) {
    if (IsFoldableAddNInput(input)) {
      ++foldable;
    }
  }
  return foldable > 1 && foldable < node->input_size();
}

bool PartialAddNFoldingStage::TrySimplify(GraphNode* node) {
  std::vector<GraphNode*> foldable;
  std::vector<GraphNode*> unfoldable;
  for (auto* input : node->input()) {
    if (IsFoldableAddNInput(input)) {
      foldable.emplace_back(input);
    } else {
      unfoldable.emplace_back(input);
    }
  }

  GraphNode* folded_addn =
      AddN(NewNodeName(node->name(), "folded_addn"), foldable);
  ctx_->item->Add(folded_addn);
  ctx_->nodes_to_simp.PushBack(folded_addn);

  GraphNode* constant = RunGraphToGetNewConstant(
      folded_addn, NewNodeName(node->name(), "new_constant"));
  ctx_->item->Add(constant);
  ctx_->nodes_to_simp.PushBack(constant);
  unfoldable.emplace_back(constant);
  SortByName(&unfoldable);

  GraphNode* new_addn = AddN(NewNodeName(node->name(), "new_addn"), unfoldable);
  ctx_->item->Add(new_addn);
  ctx_->nodes_to_simp.PushBack(new_addn);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_addn->name());
  return true;
}

bool PartialAddNFoldingStage::IsFoldableAddNInput(const GraphNode* node) const
    noexcept {
  return IsFoldableConstantNode(node) && !IsTarget(node) &&
         IsSingleOutput(node);
}

/************************************************************************/
/* PartialConcatFoldingStage */
/************************************************************************/
bool PartialConcatFoldingStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(ConcatNode) || IsTarget(node) ||
      node->input_size() < 2) {
    return false;
  }

  // If all inputs are foldable, GraphFolding will process node.
  bool all_constant = true;
  for (auto* input : node->input()) {
    if (!IsFoldableConcatInput(input)) {
      all_constant = false;
    }
  }
  if (all_constant) {
    return false;
  }

  for (int i = 1; i < node->input_size(); ++i) {
    if (IsFoldableConcatInput(node->input(i - 1)) &&
        IsFoldableConcatInput(node->input(i))) {
      return true;
    }
  }
  return false;
}

bool PartialConcatFoldingStage::TrySimplify(GraphNode* node) {
  int axis = ((ConcatNode*)node)->axis();
  std::vector<GraphNode*> new_inputs;
  int folded_concat_count = 0;
  int first = 0;
  while (first < node->input_size()) {
    while (first < node->input_size() &&
           !IsFoldableConcatInput(node->input(first))) {
      new_inputs.emplace_back(node->input()[first]);
      ++first;
    }
    int last = first + 1;
    while (last < node->input_size() &&
           IsFoldableConcatInput(node->input(last))) {
      ++last;
    }
    if (last - first > 1) {
      ++folded_concat_count;
      GraphNode* folded_concat =
          Concat(NewNodeName(node->name(), std::to_string(folded_concat_count)),
                 std::vector<GraphNode*>(node->input().begin() + first,
                                         node->input().begin() + last),
                 axis);
      ctx_->item->Add(folded_concat);
      ctx_->nodes_to_simp.PushBack(folded_concat);

      GraphNode* constant = RunGraphToGetNewConstant(
          folded_concat,
          NewNodeName(node->name(),
                      "new_constant_" + std::to_string(folded_concat_count)));
      ctx_->item->Add(constant);
      ctx_->nodes_to_simp.PushBack(constant);
      new_inputs.emplace_back(constant);
    } else {
      new_inputs.emplace_back(node->input()[first]);
    }
    first = last;
  }

  GraphNode* new_concat =
      Concat(NewNodeName(node->name(), "new_concat"), new_inputs, axis);
  ctx_->item->Add(new_concat);
  ctx_->nodes_to_simp.PushBack(new_concat);
  ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_concat->name());
  return true;
}

bool PartialConcatFoldingStage::IsFoldableConcatInput(
    const GraphNode* node) const noexcept {
  return IsFoldableConstantNode(node) && !IsTarget(node) &&
         IsSingleOutput(node);
}

/************************************************************************/
/* ArithmeticOperationsFoldingStage */
/************************************************************************/
bool ArithmeticOperationsFoldingStage::MaySimplify(const GraphNode* node) const
    noexcept {
  if (node->type_index() != typeid(AddNode) &&
      node->type_index() != typeid(SubNode) &&
      node->type_index() != typeid(MulNode) &&
      node->type_index() != typeid(DivNode)) {
    return false;
  }
  return !IsTarget(node);
}

bool ArithmeticOperationsFoldingStage::TrySimplify(GraphNode* node) {
  auto all_equal_to_value = [](const GraphNode* n, double value) {
    if (n->type_index() == typeid(ConstantNode)) {
      const auto* constant = (const ConstantNode*)n;
      if (constant->constant_type() == ConstantNode::CONSTANT_TYPE_VALUE) {
        return constant->value() == value;
      } else if (constant->constant_type() ==
                 ConstantNode::CONSTANT_TYPE_VALUES) {
        for (auto v : constant->values()) {
          if (v != value) {
            return false;
          }
        }
        return true;
      }
    }
    return false;
  };

  GraphNode* X = node->input()[0];
  GraphNode* Y = node->input()[1];
  bool is_add = node->type_index() == typeid(AddNode);
  bool is_sub = node->type_index() == typeid(SubNode);
  bool is_mul = node->type_index() == typeid(MulNode);
  bool is_div = node->type_index() == typeid(DivNode);
  bool X_is_zero =
      X->type_index() == typeid(ZerosLikeNode) || all_equal_to_value(X, 0);
  bool X_is_one =
      X->type_index() == typeid(OnesLikeNode) || all_equal_to_value(X, 1);
  bool Y_is_zero =
      Y->type_index() == typeid(ZerosLikeNode) || all_equal_to_value(Y, 0);
  bool Y_is_one =
      Y->type_index() == typeid(OnesLikeNode) || all_equal_to_value(Y, 1);

  GraphNode* replacement;
  if ((X_is_zero && is_add) || (X_is_one && is_mul)) {
    // 0 + Y or 1 * Y
    replacement = Y;
  } else if (X_is_zero && (is_mul || is_div)) {
    // 0 * Y or 0 / Y
    replacement = ZerosLike(NewNodeName(Y->name(), "ZerosLike"), Y);
    ctx_->item->Add(replacement);
    ctx_->nodes_to_simp.PushBack(replacement);
  } else if (X_is_zero && is_sub) {
    // 0 - Y
    replacement = Negate(NewNodeName(Y->name(), "Negate"), Y);
    ctx_->item->Add(replacement);
    ctx_->nodes_to_simp.PushBack(replacement);
  } else if (X_is_one && is_div) {
    // 1 / Y
    replacement = Inv(NewNodeName(Y->name(), "Inv"), Y);
    ctx_->item->Add(replacement);
    ctx_->nodes_to_simp.PushBack(replacement);
  } else if ((Y_is_zero && (is_add || is_sub)) ||
             (Y_is_one && (is_mul || is_div))) {
    // X + 0, X - 0, X * 1 or X / 1
    replacement = X;
  } else if (Y_is_zero && is_mul) {
    // X * 0
    replacement = ZerosLike(NewNodeName(X->name(), "ZerosLike"), X);
    ctx_->item->Add(replacement);
    ctx_->nodes_to_simp.PushBack(replacement);
  } else {
    return false;
  }

  ctx_->item->ReplaceInputOfAllOutputs(node->name(), replacement->name());
  return true;
}

/************************************************************************/
/* ConstantPushDownStage */
/************************************************************************/
namespace {

bool IsConstant(const GraphNode* node) noexcept {
  return node->type_index() == typeid(ConstantNode);
}

bool IsAddOrSub(const GraphNode* node) noexcept {
  return node->type_index() == typeid(AddNode) ||
         node->type_index() == typeid(SubNode);
}

bool IsMulOrDiv(const GraphNode* node) noexcept {
  return node->type_index() == typeid(MulNode) ||
         node->type_index() == typeid(DivNode);
}

bool IsPositive(const GraphNode* node) noexcept {
  return node->type_index() == typeid(AddNode) ||
         node->type_index() == typeid(MulNode);
}

}  // namespace

/************************************************************************/
/* ConstantPushDownStage */
/************************************************************************/
/* Do transformation like the following(just an example).
 *    +            +      = parent
 *   / \          / \
 *  C   +    ->  X   +    = children
 *     / \          / \
 *    X   Y        C   Y  = leaves
 * where C is constant, X is non-constant, Y may be constant or non-constant.
 * Here at least one of X and Y must be non-constant, otherwise the X and Y are
 * already foldable. This transformation push down constant node down in the
 * tree to allow subsequent simplification, like constant folding etc.
 *
 * Naming abbreviations.
 * P: parent node.
 * C: constant child node.
 * O: the other child node.
 * X: the non-constant leaf node.
 * Y: the other leaf node.
 */
bool ConstantPushDownStage::MaySimplify(const GraphNode* node) const noexcept {
  if (!IsAddOrSub(node) && !(IsMulOrDiv(node))) {
    return false;
  }
  if (IsTarget(node)) {
    return false;
  }
  if (!IsConstant(node->input(0)) && !IsConstant(node->input(1))) {
    return false;
  }
  const GraphNode* O =
      IsConstant(node->input(0)) ? node->input(1) : node->input(0);
  if (IsTarget(O) || !IsSingleOutput(O)) {
    return false;
  }
  if (!IsAddOrSub(O) && !IsMulOrDiv(O)) {
    return false;
  }
  if (IsConstant(O->input(0)) && IsConstant(O->input(1))) {
    return false;
  }
  bool add_sub_combination = IsAddOrSub(node) && IsAddOrSub(O);
  bool mul_div_combination = IsMulOrDiv(node) && IsMulOrDiv(O);
  if (!add_sub_combination && !mul_div_combination) {
    return false;
  }
  return true;
}

bool ConstantPushDownStage::TrySimplify(GraphNode* node) {
  GraphNode* P = node;
  bool C_is_left = IsConstant(P->input()[0]);
  GraphNode* C = C_is_left ? P->input()[0] : P->input()[1];
  GraphNode* O = C_is_left ? P->input()[1] : P->input()[0];
  bool X_is_left = !IsConstant(O->input(0));
  GraphNode* X = X_is_left ? O->input()[0] : O->input()[1];
  GraphNode* Y = X_is_left ? O->input()[1] : O->input()[0];
  bool P_is_positive = IsPositive(P);
  bool O_is_positive = IsPositive(O);
  if (P_is_positive && O_is_positive) {
    /*
     *    +(*)          +(*)          +(*)          +(*)
     *   / \           / \           / \           / \
     *  C   +(*)  ->  X   +(*) or  +(*) C  ->    +(*) X
     *     / \           / \      / \           / \
     *    X   Y         C   Y    X   Y         C   Y
     */
    ctx_->item->ReplaceInput(P->name(), C->name(), X->name());
    ctx_->item->ReplaceInput(O->name(), X->name(), C->name());
  } else {
    /* We can rewrite this subtree according the the sign of each of terms C, X
     * and Y by generalizing the concept sign to mul(positive) and div(negative)
     * operations. We can rebuild C and Y to get new child subtree named by
     * O and then rebuild X and O to get new subtree.
     */
    bool add_sub_combination = IsAddOrSub(node) && IsAddOrSub(O);
    auto make_new_node = [add_sub_combination](const std::string& new_node_name,
                                               GraphNode* operand1,
                                               GraphNode* operand2,
                                               bool positive) {
      if (add_sub_combination) {
        if (positive) {
          return Add(new_node_name, operand1, operand2);
        } else {
          return Sub(new_node_name, operand1, operand2);
        }
      } else {
        if (positive) {
          return Mul(new_node_name, operand1, operand2);
        } else {
          return Div(new_node_name, operand1, operand2);
        }
      }
    };

    bool left_leaf_is_positive, right_leaf_is_positive;
    if (C_is_left) {
      left_leaf_is_positive = P_is_positive;
      right_leaf_is_positive = !(P_is_positive ^ O_is_positive);
    } else {
      left_leaf_is_positive = true;
      right_leaf_is_positive = O_is_positive;
    }

    bool C_is_positive = C_is_left || P_is_positive;
    bool X_is_positive =
        X_is_left ? left_leaf_is_positive : right_leaf_is_positive;
    bool Y_is_positive =
        X_is_left ? right_leaf_is_positive : left_leaf_is_positive;
    bool C_Y_same_sign = !(C_is_positive ^ Y_is_positive);

    // C + Y, C - Y, Y - C
    GraphNode* operand1 = C_Y_same_sign || C_is_positive ? C : Y;
    GraphNode* operand2 = C_Y_same_sign || C_is_positive ? Y : C;
    std::string new_O_name = NewNodeName(O->name());
    GraphNode* new_O =
        make_new_node(new_O_name, operand1, operand2, C_Y_same_sign);
    ctx_->item->Add(new_O);
    ctx_->nodes_to_simp.PushBack(new_O);

    std::string new_node_name = NewNodeName(node->name());
    GraphNode* new_P;
    if (X_is_positive) {
      // X - (C + Y), X + (C - Y), X + (Y - C)
      new_P = make_new_node(new_node_name, X, new_O, !C_Y_same_sign);
    } else {
      // (C + Y) - X, (C - Y) - X, (Y - C) - X
      new_P = make_new_node(new_node_name, new_O, X, false);
    }
    ctx_->item->Add(new_P);
    ctx_->nodes_to_simp.PushBack(new_O);
    ctx_->item->ReplaceInputOfAllOutputs(node->name(), new_P->name());
  }

  return true;
}

/************************************************************************/
/* CFSimp */
/************************************************************************/
CFSimp::CFSimp(const CFConfig& config)
    : Simp("constant_folding"), config_(config) {}

bool CFSimp::Simplify(SimpItem* mutable_item) const {
  SimpContext ctx;
  std::vector<std::unique_ptr<SimpStage>> stages;
  stages.emplace_back(new RemoveIdenticalTransposeStage(name(), &ctx));
  stages.emplace_back(new RemoveIdenticalTileStage(name(), &ctx));
  stages.emplace_back(new DivToReciprocalMulFoldingStage(name(), &ctx));
  stages.emplace_back(new PartialAddNFoldingStage(name(), &ctx));
  stages.emplace_back(new PartialConcatFoldingStage(name(), &ctx));
  stages.emplace_back(new ArithmeticOperationsFoldingStage(name(), &ctx));
  stages.emplace_back(new ConstantPushDownStage(name(), &ctx));
  if (config_.use_static_shape) {
    stages.emplace_back(new RemoveIdenticalSubscriptRangeStage(name(), &ctx));
    stages.emplace_back(new RemoveIdenticalReshapeStage(name(), &ctx));
  }
  SimpPipeline pipeline(&ctx, std::move(stages));

  GraphFolding graph_folding(config_);

  bool simplified = false;
  for (;;) {
    bool _simplified = graph_folding.FoldGraph(mutable_item);
    ctx.Init(mutable_item);
    while (!ctx.nodes_to_simp.Empty()) {
      GraphNode* node = ctx.nodes_to_simp.PopBack();
      if (pipeline.TrySimplify(node)) {
        _simplified = true;
      }
    }
    if (!_simplified) {
      break;
    }
    simplified = true;
    ctx.item->Prune();
  }

  return simplified;
}

}  // namespace deepx_core
