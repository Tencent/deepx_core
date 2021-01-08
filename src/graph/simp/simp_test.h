// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <deepx_core/graph/graph.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>
#include "simp_item.h"
#include "simp_stage.h"

namespace deepx_core {

class SimpTestBase : public testing::Test {
 protected:
  const GraphNode* null_node = nullptr;
  Graph graph;
  SimpItem item;
  std::string simp_name;
  SimpContext ctx;

 protected:
  void SimplifyTwice();
  virtual void Simplify() = 0;
  void AssertNodesDeleted(const std::vector<GraphNode*>& nodes);
  void AssertTypeEQ(const std::string& node_name,
                    const std::type_index& type_index);
  void AssertInputsEQ(const std::string& node_name,
                      const std::vector<std::string>& input_names);
};

class SimpStageTestBase : public SimpTestBase {
 protected:
  std::unique_ptr<SimpStage> stage;

 protected:
  void Simplify() override;
  std::string ScopedName(const std::string& name, int rep = 1) const noexcept;
};

}  // namespace deepx_core
