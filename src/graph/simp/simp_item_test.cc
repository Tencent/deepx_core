// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "simp_item.h"
#include <gtest/gtest.h>

namespace deepx_core {

class SimpItemTest : public testing::Test {
 protected:
  const GraphNode* null_node = nullptr;
  const GraphTarget* null_target = nullptr;
  Graph graph;
  SimpItem item;
  Shape shape;

 protected:
  void SetUp() override {
    shape.resize(2, 3);

    auto* X1 = Constant("X1", shape, 1);
    auto* X2 = ReduceMean("X2", X1, 0, 1);
    auto* X3 = ReduceSum("X3", X1, 0, 1);
    auto* X4 = Add("X4", X1, X3);
    auto* X5 = Sub("X5", X2, X3);
    ASSERT_TRUE(graph.Compile({X4, X5}, 1));
    item.FromGraph(graph);
  }

  static void GetOutputNames(const std::unordered_set<GraphNode*>& outputs,
                             std::unordered_set<std::string>* output_names) {
    output_names->clear();
    for (auto* output : outputs) {
      output_names->emplace(output->name());
    }
  }
};

TEST_F(SimpItemTest, Add) {
  item.FromGraph(graph);

  item.Add(item.find_node("X1"));
  auto* X6 = new AddNode("X6", item.find_node("X1"), item.find_node("X3"));
  item.Add(X6);
  ASSERT_GT(item.find_output("X1").count(item.find_node("X6")), 0u);
  ASSERT_GT(item.find_output("X3").count(item.find_node("X6")), 0u);
}

TEST_F(SimpItemTest, ReplaceInput) {
  auto* X1 = Constant("X1", shape, 1);
  auto* X2 = ReduceMean("X2", X1, 0, 1);
  auto* X3 = Sub("X3", X2, X2);
  ASSERT_TRUE(graph.Compile({X3}, 1));
  item.FromGraph(graph);
  auto* X4 = new ReduceMinNode("X4", item.find_node("X1"));
  item.Add(X4);

  ASSERT_FALSE(item.ReplaceInput("X0", "X2", "X4"));
  ASSERT_FALSE(item.ReplaceInput("X3", "X0", "X4"));
  ASSERT_FALSE(item.ReplaceInput("X3", "X2", "X0"));
  ASSERT_TRUE(item.ReplaceInput("X3", "X2", "X4"));
  auto& X3_input = item.find_node("X3")->input();
  ASSERT_EQ(0,
            std::count(X3_input.begin(), X3_input.end(), item.find_node("X2")));
  ASSERT_EQ(2,
            std::count(X3_input.begin(), X3_input.end(), item.find_node("X4")));

  auto& X2_output = item.find_output("X2");
  ASSERT_EQ(0, (int)X2_output.count(item.find_node("X3")));
}

TEST_F(SimpItemTest, ReplaceInput_throw) {
  auto* X1 = Constant("X1", shape, 1);
  auto* X2 = ReduceMean("X2", X1, 0, 1);
  auto* X3 = ReduceMean("X3", X2, 0, 1);
  ASSERT_TRUE(graph.Compile({X3}, 1));
  item.FromGraph(graph);
  ASSERT_ANY_THROW(item.ReplaceInput("X2", "X1", "X3"));
}

TEST_F(SimpItemTest, ReplaceInputOfAllOutputs) {
  auto* X1 = Constant("X1", Shape(2, 3), 1);
  auto* X2 = ReduceMean("X2", X1, 0, 1);
  auto* X3 = Sub("X3", X2, X2);
  ASSERT_TRUE(graph.Compile({X3}, 1));
  item.FromGraph(graph);
  auto* X4 = new ReduceMinNode("X4", item.find_node("X1"));
  item.Add(X4);

  ASSERT_FALSE(item.ReplaceInputOfAllOutputs("X0", "X1"));
  ASSERT_FALSE(item.ReplaceInputOfAllOutputs("X1", "X0"));
  ASSERT_TRUE(item.ReplaceInputOfAllOutputs("X2", "X4"));

  auto& X3_input = item.find_node("X3")->input();
  ASSERT_EQ(0,
            std::count(X3_input.begin(), X3_input.end(), item.find_node("X2")));
  ASSERT_EQ(2,
            std::count(X3_input.begin(), X3_input.end(), item.find_node("X4")));

  auto& X2_output = item.find_output("X2");
  ASSERT_EQ(0, (int)X2_output.count(item.find_node("X3")));
}

TEST_F(SimpItemTest, ReplaceInputOfAllOutputs_throw) {
  auto* X1 = Constant("X1", shape, 1);
  auto* X2 = ReduceMean("X2", X1, 0, 1);
  auto* X3 = ReduceMean("X3", X2, 0, 1);
  ASSERT_TRUE(graph.Compile({X3}, 1));
  item.FromGraph(graph);
  ASSERT_ANY_THROW(item.ReplaceInputOfAllOutputs("X1", "X3"));
}

TEST_F(SimpItemTest, Prune) {
  auto* X1 = Constant("X1", shape, 1);
  auto* X3 = Constant("X3", shape, 1);
  auto* X4 = ReduceMean("X4", X1, 0, 1);
  auto* X5 = AddN("X5", {X1, X3, X1});
  ASSERT_TRUE(graph.Compile({X4, X5}, 1));
  item.FromGraph(graph);
  ASSERT_TRUE(item.ReplaceInput("X5", "X3", "X1"));

  ASSERT_NE(item.find_node("X3"), null_node);
  item.Prune();
  ASSERT_EQ(item.find_node("X3"), null_node);
}

TEST_F(SimpItemTest, GetTopologicalSortedNodes) {
  auto* X1 = Constant("X1", shape, 1);
  auto* X3 = ReduceMean("X3", X1, 0, 1);
  auto* X2 = Add("X2", X1, X3);
  auto* X4 = ReduceSum("X4", X1, 0, 1);
  auto* X6 = AddN("X6", {X4, X4, X4});
  auto* X5 = Sub("X5", X6, X3);
  ASSERT_TRUE(graph.Compile({X2, X5}, 1));
  item.FromGraph(graph);

  std::vector<GraphNode*> sorted;
  item.GetTopologicalSortedNodes(&sorted);
  ASSERT_EQ(6, (int)sorted.size());
  ASSERT_EQ(sorted[0], item.find_node("X1"));
  ASSERT_EQ(sorted[1], item.find_node("X3"));
  ASSERT_EQ(sorted[2], item.find_node("X4"));
  ASSERT_EQ(sorted[3], item.find_node("X2"));
  ASSERT_EQ(sorted[4], item.find_node("X6"));
  ASSERT_EQ(sorted[5], item.find_node("X5"));
}

TEST_F(SimpItemTest, GetTopologicalSortedNodes_reverse) {
  auto* X1 = Constant("X1", shape, 1);
  auto* X3 = ReduceMean("X3", X1, 0, 1);
  auto* X2 = Add("X2", X1, X3);
  auto* X4 = ReduceSum("X4", X1, 0, 1);
  auto* X6 = AddN("X6", {X4, X4, X4});
  auto* X5 = Sub("X5", X6, X3);
  ASSERT_TRUE(graph.Compile({X2, X5}, 1));
  item.FromGraph(graph);

  std::vector<GraphNode*> sorted;
  item.GetTopologicalSortedNodes(&sorted, true);
  ASSERT_EQ(6, (int)sorted.size());
  ASSERT_EQ(sorted[0], item.find_node("X5"));
  ASSERT_EQ(sorted[1], item.find_node("X6"));
  ASSERT_EQ(sorted[2], item.find_node("X2"));
  ASSERT_EQ(sorted[3], item.find_node("X4"));
  ASSERT_EQ(sorted[4], item.find_node("X3"));
  ASSERT_EQ(sorted[5], item.find_node("X1"));
}

TEST_F(SimpItemTest, NewNodeName) {
  auto* X1 = Constant("X1", shape, 1);
  auto* X1_new = ReduceMean("X1_new", X1, 0, 1);
  ASSERT_TRUE(graph.Compile({X1_new}, 1));
  item.FromGraph(graph);

  std::string name;
  ASSERT_EQ("X1_new_new", item.NewNodeName("X1"));
  ASSERT_EQ("X2", item.NewNodeName("X2"));
  ASSERT_EQ("A/X1", item.NewNodeName("X1", {"A"}));
  ASSERT_EQ("A/B/X1", item.NewNodeName("X1", {"A", "B"}));
  ASSERT_EQ("A/p_X1", item.NewNodeName("X1", {"A"}, "p"));
  ASSERT_EQ("A/p_X1_s", item.NewNodeName("X1", {"A"}, "p", "s"));
}

TEST_F(SimpItemTest, FromGraph) {
  ASSERT_EQ(2, item.target_size());
  EXPECT_TRUE(item.is_target("X4"));
  EXPECT_TRUE(item.is_target("X5"));
  EXPECT_FALSE(item.is_target("X1"));
  EXPECT_FALSE(item.is_target("X2"));
  EXPECT_FALSE(item.is_target("X3"));

  ASSERT_EQ(5, item.node_size());
  EXPECT_NE(item.find_node("X1"), null_node);
  EXPECT_NE(item.find_node("X2"), null_node);
  EXPECT_NE(item.find_node("X3"), null_node);
  EXPECT_NE(item.find_node("X4"), null_node);
  EXPECT_NE(item.find_node("X5"), null_node);

  std::unordered_set<std::string> X1_output_names, X2_output_names,
      X3_output_names, X4_output_names, X5_output_names;
  GetOutputNames(item.find_output("X1"), &X1_output_names);
  GetOutputNames(item.find_output("X2"), &X2_output_names);
  GetOutputNames(item.find_output("X3"), &X3_output_names);
  GetOutputNames(item.find_output("X4"), &X4_output_names);
  GetOutputNames(item.find_output("X5"), &X5_output_names);
  ASSERT_EQ(3, (int)X1_output_names.size());
  ASSERT_GT(X1_output_names.count("X2"), 0u);
  ASSERT_GT(X1_output_names.count("X3"), 0u);
  ASSERT_GT(X1_output_names.count("X4"), 0u);
  ASSERT_EQ(1, (int)X2_output_names.size());
  ASSERT_GT(X2_output_names.count("X5"), 0u);
  ASSERT_EQ(2, (int)X3_output_names.size());
  ASSERT_GT(X3_output_names.count("X4"), 0u);
  ASSERT_GT(X3_output_names.count("X5"), 0u);
  ASSERT_EQ(0, (int)X4_output_names.size());
  ASSERT_EQ(0, (int)X5_output_names.size());
}

TEST_F(SimpItemTest, ToGraph) {
  item.ToGraph(&graph);

  const GraphNode* _X1 = graph.find_node("X1");
  const GraphNode* _X2 = graph.find_node("X2");
  const GraphNode* _X3 = graph.find_node("X3");
  EXPECT_EQ(_X1->input_fork(), 0);
  EXPECT_EQ(_X2->output_size(), 1);
  EXPECT_EQ(_X2->input_fork(), 1);
  EXPECT_EQ(_X3->output_size(), 2);
  EXPECT_EQ(_X3->input_fork(), 1);

  ASSERT_EQ(graph.target_size(), 2);
  EXPECT_EQ(graph.target(0).name(), "X4");
  EXPECT_EQ(graph.target(0).forward_name(),
            std::vector<std::string>({"X1", "X3", "X4"}));
  EXPECT_EQ(graph.target(1).name(), "X5");
  EXPECT_EQ(graph.target(1).forward_name(),
            std::vector<std::string>({"X1", "X2", "X3", "X5"}));
  EXPECT_EQ(graph.find_target("X1"), null_target);
  EXPECT_EQ(graph.find_target("X2"), null_target);
  EXPECT_EQ(graph.find_target("X3"), null_target);
  EXPECT_EQ(graph.find_target("X4"), &graph.target(0));
  EXPECT_EQ(graph.find_target("X5"), &graph.target(1));

  ASSERT_EQ(5, (int)graph.name_2_node().size());
  EXPECT_NE(graph.find_node("X1"), null_node);
  EXPECT_NE(graph.find_node("X2"), null_node);
  EXPECT_NE(graph.find_node("X3"), null_node);
  EXPECT_NE(graph.find_node("X4"), null_node);
  EXPECT_NE(graph.find_node("X5"), null_node);
}

}  // namespace deepx_core
