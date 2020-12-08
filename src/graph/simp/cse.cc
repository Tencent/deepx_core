// Copyright 2020 the deepx authors.
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include "cse.h"
#include <algorithm>  // std::sort
#include <unordered_set>
#include <vector>

namespace deepx_core {
namespace {

bool IsInputCommutative(const GraphNode* node) noexcept {
  return node->type_index() == typeid(MulNode) ||
         node->type_index() == typeid(BroadcastMulNode) ||
         node->type_index() == typeid(AddNode) ||
         node->type_index() == typeid(BroadcastAddNode) ||
         node->type_index() == typeid(AddNNode);
}

bool IsEquivalent(const GraphNode* a, const GraphNode* b) {
  if (a->input_size() != b->input_size()) {
    return false;
  }

  if (a->shape() != b->shape()) {
    return false;
  }

  if (a->type_index() != b->type_index()) {
    return false;
  }

  if (a->type_index() == typeid(InstanceNode) ||
      a->type_index() == typeid(VariableNode) ||
      a->type_index() == typeid(RandomNormalNode) ||
      a->type_index() == typeid(RandomUniformNode) ||
      a->type_index() == typeid(RandomNormalLikeNode) ||
      a->type_index() == typeid(RandomUniformLikeNode)) {
    return false;
  }

  if (a->type_index() == typeid(ConstantNode) &&
      ((const ConstantNode*)a)->constant_type() ==
          ConstantNode::CONSTANT_TYPE_INITIALIZER) {
    return false;
  }

  if (a->type_index() == typeid(ConstantLikeNode) &&
      ((const ConstantLikeNode*)a)->constant_type() ==
          ConstantLikeNode::CONSTANT_TYPE_INITIALIZER) {
    return false;
  }

  if (IsInputCommutative(a)) {
    std::vector<GraphNode*> a_input(a->input().begin(), a->input().end());
    std::sort(a_input.begin(), a_input.end());
    std::vector<GraphNode*> b_input(b->input().begin(), b->input().end());
    std::sort(b_input.begin(), b_input.end());
    if (a_input != b_input) {
      return false;
    }
  } else if (a->input() != b->input()) {
    return false;
  }

  return a->IsAttrEqual(b);
}

}  // namespace

CSESimp::CSESimp() : Simp("common_subexpression_elimination") {}

bool CSESimp::Simplify(SimpItem* item) const {
  std::vector<GraphNode*> sorted;
  item->GetTopologicalSortedNodes(&sorted);

  bool simplified = false;
  std::unordered_set<GraphNode*> processed;
  for (size_t i = 0; i < sorted.size(); ++i) {
    GraphNode* node = sorted[i];
    if (processed.count(node) > 0) {
      continue;
    }
    processed.insert(node);
    if (item->is_target(node->name())) {
      continue;
    }
    for (size_t j = i + 1; j < sorted.size(); ++j) {
      GraphNode* candidate = sorted[j];
      if (processed.count(candidate) > 0 ||
          item->is_target(candidate->name())) {
        continue;
      }
      if (IsEquivalent(node, candidate)) {
        simplified = true;
        processed.insert(candidate);
        item->ReplaceInputOfAllOutputs(candidate->name(), node->name());
      }
    }
  }
  item->Prune();
  return simplified;
}

}  // namespace deepx_core
