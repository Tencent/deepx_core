// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "simp_test.h"

namespace deepx_core {

void SimpTestBase::SimplifyTwice() {
  Simplify();
  Simplify();
}

void SimpTestBase::AssertNodesDeleted(const std::vector<GraphNode*>& nodes) {
  for (auto* node : nodes) {
    ASSERT_EQ(item.find_node(node->name()), null_node);
  }
}

void SimpTestBase::AssertTypeEQ(const std::string& node_name,
                                const std::type_index& type_index) {
  ASSERT_EQ(item.find_node(node_name)->type_index(), type_index);
}

void SimpTestBase::AssertInputsEQ(const std::string& node_name,
                                  const std::vector<std::string>& input_names) {
  GraphNode* node = item.find_node(node_name);
  ASSERT_NE(node, null_node);
  ASSERT_EQ(node->input_size(), (int)input_names.size());
  if (node->input_size() == 0) {
    return;
  }

  std::string actual_input_names(node->input(0)->name());
  std::string expect_input_names(input_names[0]);
  for (int i = 1; i < (int)input_names.size(); ++i) {
    actual_input_names.append(" " + node->input(i)->name());
    expect_input_names.append(" " + input_names[i]);
  }
  ASSERT_EQ(actual_input_names, expect_input_names);
}

void SimpStageTestBase::Simplify() {
  bool simplified;
  do {
    ctx.Init(&item);
    simplified = false;
    while (!ctx.nodes_to_simp.Empty()) {
      GraphNode* node = ctx.nodes_to_simp.PopBack();
      if (stage->MaySimplify(node)) {
        if (stage->TrySimplify(node)) {
          simplified = true;
        }
      }
    }
    item.Prune();
  } while (simplified);
}

std::string SimpStageTestBase::ScopedName(const std::string& name,
                                          int rep) const noexcept {
  std::string scope;
  while (rep-- > 0) {
    scope += stage->simp_name() + "/" + stage->stage_name() + "/";
  }
  return scope + name;
}

}  // namespace deepx_core
