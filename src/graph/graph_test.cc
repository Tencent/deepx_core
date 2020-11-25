// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/graph/graph.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace deepx_core {

class GraphTest : public testing::Test {
 protected:
  Graph graph;
  const GraphTarget* null = nullptr;
};

TEST_F(GraphTest, Compile_duplicate_name) {
  VariableNode X1("X", Shape(2, 3));
  ReduceMeanNode X2("X", &X1, 1, 1);

  EXPECT_TRUE(X1.IsValidName());
  EXPECT_TRUE(X2.IsValidName());
  ASSERT_FALSE(graph.Compile({&X2}, 0));
}

TEST_F(GraphTest, Compile_1_target_invalid_name) {
  VariableNode X1("X1", Shape(2, 3));
  VariableNode X2("X2", Shape(2, 3));
  AddNode X3("?", &X1, &X2);
  ReduceMeanNode X4("", &X3, 1, 1);

  EXPECT_TRUE(X1.IsValidName());
  EXPECT_TRUE(X2.IsValidName());
  EXPECT_FALSE(X3.IsValidName());
  EXPECT_FALSE(X4.IsValidName());
  ASSERT_TRUE(graph.Compile({&X4}, 0));

  EXPECT_TRUE(X1.IsValidName());
  EXPECT_TRUE(X2.IsValidName());
  EXPECT_TRUE(X3.IsValidName());
  EXPECT_TRUE(X4.IsValidName());
}

TEST_F(GraphTest, Compile_2_target_invalid_name) {
  VariableNode X1("X1", Shape(2, 3));
  VariableNode X2("X2", Shape(2, 3));
  AddNode X3("?", &X1, &X2);
  MulNode X4("?", &X1, &X2);
  ReduceMeanNode X5("", &X3, 1, 1);
  ReduceMeanNode X6("", &X4, 1, 1);

  EXPECT_TRUE(X1.IsValidName());
  EXPECT_TRUE(X2.IsValidName());
  EXPECT_FALSE(X3.IsValidName());
  EXPECT_FALSE(X4.IsValidName());
  EXPECT_FALSE(X5.IsValidName());
  EXPECT_FALSE(X6.IsValidName());
  ASSERT_TRUE(graph.Compile({&X5, &X6}, 0));

  EXPECT_TRUE(X1.IsValidName());
  EXPECT_TRUE(X2.IsValidName());
  EXPECT_TRUE(X3.IsValidName());
  EXPECT_TRUE(X4.IsValidName());
  EXPECT_TRUE(X5.IsValidName());
  EXPECT_TRUE(X6.IsValidName());
}

TEST_F(GraphTest, Compile_1_target) {
  VariableNode X1("X1", Shape(2, 3));
  ReduceMeanNode X2("X2", &X1, 1, 1);
  ASSERT_TRUE(graph.Compile({&X2}, 0));

  EXPECT_EQ(X1.need_grad(), 1);
  EXPECT_EQ(X1.output_size(), 1);
  EXPECT_EQ(X1.input_fork(), 0);
  EXPECT_FALSE(X1.is_target());
  EXPECT_EQ(X2.need_grad(), 1);
  EXPECT_EQ(X2.output_size(), 0);
  EXPECT_EQ(X2.input_fork(), 0);
  EXPECT_TRUE(X2.is_target());

  ASSERT_EQ(graph.target_size(), 1);
  EXPECT_EQ(graph.target(0).name(), X2.name());
  EXPECT_EQ(graph.target(0).forward_name(),
            std::vector<std::string>({X1.name(), X2.name()}));

  EXPECT_EQ(graph.find_target(X1.name()), null);
  EXPECT_EQ(graph.find_target(X2.name()), &graph.target(0));
  EXPECT_EQ(graph.find_node(X1.name()), &X1);
  EXPECT_EQ(graph.find_node(X2.name()), &X2);

  EXPECT_EQ(graph.find_target(X1.node_id()), null);
  EXPECT_EQ(graph.find_target(X2.node_id()), &graph.target(0));
  EXPECT_EQ(graph.find_node(X1.node_id()), &X1);
  EXPECT_EQ(graph.find_node(X2.node_id()), &X2);
}

TEST_F(GraphTest, Compile_2_target_on_heap) {
  auto* X1 = new VariableNode("X1", Shape(2, 3));
  auto* X2 = new ReduceMeanNode("X2", X1, 1, 1);
  ASSERT_TRUE(graph.Compile({X1, X2}, 1));

  EXPECT_EQ(X1->need_grad(), 1);
  EXPECT_EQ(X1->output_size(), 1);
  EXPECT_EQ(X1->input_fork(), 0);
  EXPECT_TRUE(X1->is_target());
  EXPECT_EQ(X2->need_grad(), 1);
  EXPECT_EQ(X2->output_size(), 0);
  EXPECT_EQ(X2->input_fork(), 0);
  EXPECT_TRUE(X2->is_target());

  ASSERT_EQ(graph.target_size(), 2);
  EXPECT_EQ(graph.target(0).name(), X1->name());
  EXPECT_EQ(graph.target(0).forward_name(),
            std::vector<std::string>({X1->name()}));
  EXPECT_EQ(graph.target(1).name(), X2->name());
  EXPECT_EQ(graph.target(1).forward_name(),
            std::vector<std::string>({X1->name(), X2->name()}));

  EXPECT_EQ(graph.find_target(X1->name()), &graph.target(0));
  EXPECT_EQ(graph.find_target(X2->name()), &graph.target(1));
  EXPECT_EQ(graph.find_node(X1->name()), X1);
  EXPECT_EQ(graph.find_node(X2->name()), X2);

  EXPECT_EQ(graph.find_target(X1->node_id()), &graph.target(0));
  EXPECT_EQ(graph.find_target(X2->node_id()), &graph.target(1));
  EXPECT_EQ(graph.find_node(X1->node_id()), X1);
  EXPECT_EQ(graph.find_node(X2->node_id()), X2);
}

TEST_F(GraphTest, Compile_2_target_on_heap_fork) {
  auto* X1 = new VariableNode("X1", Shape(2, 3));
  auto* X2 = new ReduceMeanNode("X2", X1, 0, 1);
  auto* X3 = new ReduceSumNode("X3", X1, 0, 1);
  auto* X4 = new AddNode("X4", X2, X3);
  auto* X5 = new SubNode("X5", X2, X3);
  ASSERT_TRUE(graph.Compile({X4, X5}, 1));

  EXPECT_EQ(X1->need_grad(), 1);
  EXPECT_EQ(X1->output_size(), 2);
  EXPECT_EQ(X1->input_fork(), 0);
  EXPECT_FALSE(X1->is_target());
  EXPECT_EQ(X2->need_grad(), 1);
  EXPECT_EQ(X2->output_size(), 2);
  EXPECT_EQ(X2->input_fork(), 1);
  EXPECT_FALSE(X2->is_target());
  EXPECT_EQ(X3->need_grad(), 1);
  EXPECT_EQ(X3->output_size(), 2);
  EXPECT_EQ(X3->input_fork(), 1);
  EXPECT_FALSE(X3->is_target());
  EXPECT_EQ(X4->need_grad(), 1);
  EXPECT_EQ(X4->output_size(), 0);
  EXPECT_EQ(X4->input_fork(), 1);
  EXPECT_TRUE(X4->is_target());
  EXPECT_EQ(X5->need_grad(), 1);
  EXPECT_EQ(X5->output_size(), 0);
  EXPECT_EQ(X5->input_fork(), 1);
  EXPECT_TRUE(X5->is_target());

  ASSERT_EQ(graph.target_size(), 2);
  EXPECT_EQ(graph.target(0).name(), X4->name());
  EXPECT_EQ(graph.target(0).forward_name(),
            std::vector<std::string>(
                {X1->name(), X2->name(), X3->name(), X4->name()}));
  EXPECT_EQ(graph.target(1).name(), X5->name());
  EXPECT_EQ(graph.target(1).forward_name(),
            std::vector<std::string>(
                {X1->name(), X2->name(), X3->name(), X5->name()}));

  EXPECT_EQ(graph.find_target(X1->name()), null);
  EXPECT_EQ(graph.find_target(X2->name()), null);
  EXPECT_EQ(graph.find_target(X3->name()), null);
  EXPECT_EQ(graph.find_target(X4->name()), &graph.target(0));
  EXPECT_EQ(graph.find_target(X5->name()), &graph.target(1));
  EXPECT_EQ(graph.find_node(X1->name()), X1);
  EXPECT_EQ(graph.find_node(X2->name()), X2);
  EXPECT_EQ(graph.find_node(X3->name()), X3);
  EXPECT_EQ(graph.find_node(X4->name()), X4);
  EXPECT_EQ(graph.find_node(X5->name()), X5);

  EXPECT_EQ(graph.find_target(X1->node_id()), null);
  EXPECT_EQ(graph.find_target(X2->node_id()), null);
  EXPECT_EQ(graph.find_target(X3->node_id()), null);
  EXPECT_EQ(graph.find_target(X4->node_id()), &graph.target(0));
  EXPECT_EQ(graph.find_target(X5->node_id()), &graph.target(1));
  EXPECT_EQ(graph.find_node(X1->node_id()), X1);
  EXPECT_EQ(graph.find_node(X2->node_id()), X2);
  EXPECT_EQ(graph.find_node(X3->node_id()), X3);
  EXPECT_EQ(graph.find_node(X4->node_id()), X4);
  EXPECT_EQ(graph.find_node(X5->node_id()), X5);
}

TEST_F(GraphTest, Compile_2_targets_on_heap_fork_no_grad) {
  auto* X1 = new ConstantNode("X1", Shape(2, 3), 0);
  auto* X2 = new ReduceMeanNode("X2", X1, 0, 1);
  auto* X3 = new ReduceSumNode("X3", X1, 0, 1);
  auto* X4 = new AddNode("X4", X2, X3);
  auto* X5 = new SubNode("X5", X2, X3);
  ASSERT_TRUE(graph.Compile({X4, X5}, 1));

  EXPECT_EQ(X1->need_grad(), 0);
  EXPECT_EQ(X1->output_size(), 2);
  EXPECT_EQ(X1->input_fork(), 0);
  EXPECT_FALSE(X1->is_target());
  EXPECT_EQ(X2->need_grad(), 0);
  EXPECT_EQ(X2->output_size(), 2);
  EXPECT_EQ(X2->input_fork(), 1);
  EXPECT_FALSE(X2->is_target());
  EXPECT_EQ(X3->need_grad(), 0);
  EXPECT_EQ(X3->output_size(), 2);
  EXPECT_EQ(X3->input_fork(), 1);
  EXPECT_FALSE(X3->is_target());
  EXPECT_EQ(X4->need_grad(), 0);
  EXPECT_EQ(X4->output_size(), 0);
  EXPECT_EQ(X4->input_fork(), 1);
  EXPECT_TRUE(X4->is_target());
  EXPECT_EQ(X5->need_grad(), 0);
  EXPECT_EQ(X5->output_size(), 0);
  EXPECT_EQ(X5->input_fork(), 1);
  EXPECT_TRUE(X4->is_target());

  ASSERT_EQ(graph.target_size(), 2);
  EXPECT_EQ(graph.target(0).name(), X4->name());
  EXPECT_EQ(graph.target(0).forward_name(),
            std::vector<std::string>(
                {X1->name(), X2->name(), X3->name(), X4->name()}));
  EXPECT_EQ(graph.target(1).name(), X5->name());
  EXPECT_EQ(graph.target(1).forward_name(),
            std::vector<std::string>(
                {X1->name(), X2->name(), X3->name(), X5->name()}));

  EXPECT_EQ(graph.find_target(X1->name()), null);
  EXPECT_EQ(graph.find_target(X2->name()), null);
  EXPECT_EQ(graph.find_target(X3->name()), null);
  EXPECT_EQ(graph.find_target(X4->name()), &graph.target(0));
  EXPECT_EQ(graph.find_target(X5->name()), &graph.target(1));
  EXPECT_EQ(graph.find_node(X1->name()), X1);
  EXPECT_EQ(graph.find_node(X2->name()), X2);
  EXPECT_EQ(graph.find_node(X3->name()), X3);
  EXPECT_EQ(graph.find_node(X4->name()), X4);
  EXPECT_EQ(graph.find_node(X5->name()), X5);

  EXPECT_EQ(graph.find_target(X1->node_id()), null);
  EXPECT_EQ(graph.find_target(X2->node_id()), null);
  EXPECT_EQ(graph.find_target(X3->node_id()), null);
  EXPECT_EQ(graph.find_target(X4->node_id()), &graph.target(0));
  EXPECT_EQ(graph.find_target(X5->node_id()), &graph.target(1));
  EXPECT_EQ(graph.find_node(X1->node_id()), X1);
  EXPECT_EQ(graph.find_node(X2->node_id()), X2);
  EXPECT_EQ(graph.find_node(X3->node_id()), X3);
  EXPECT_EQ(graph.find_node(X4->node_id()), X4);
  EXPECT_EQ(graph.find_node(X5->node_id()), X5);
}

TEST_F(GraphTest, WriteRead) {
  VariableNode X1("X1", Shape(2, 3));
  ReduceMeanNode X2("X2", &X1, 1, 1);
  Graph read_graph;
  ASSERT_TRUE(graph.Compile({&X2}, 0));

  OutputStringStream os;
  InputStringStream is;

  ASSERT_TRUE(graph.Write(os));
  is.SetView(os.GetBuf());
  EXPECT_FALSE(read_graph.compiled());
  ASSERT_TRUE(read_graph.Read(is));
  ASSERT_TRUE(read_graph.compiled());

  ASSERT_EQ(graph.target_size(), read_graph.target_size());
  for (int i = 0; i < graph.target_size(); ++i) {
    EXPECT_EQ(graph.target(i).name(), read_graph.target(i).name());
    EXPECT_EQ(graph.target(i).forward_name(),
              read_graph.target(i).forward_name());
  }
}

}  // namespace deepx_core
