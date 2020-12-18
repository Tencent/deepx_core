// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/ol_store.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>

namespace deepx_core {

class OLStoreTest : public testing::Test, public DataType {
 protected:
  VariableNode* W1node = nullptr;
  VariableNode* W2node = nullptr;
  VariableNode* W3node = nullptr;
  Graph graph;
  TensorMap param;

 protected:
  void InitGraph() {
    W1node = new VariableNode("W1", Shape(2, 3));
    W2node = new VariableNode("W2", Shape(0, 2), TENSOR_TYPE_SRM);
    W3node = new VariableNode("W3", Shape(0, 1), TENSOR_TYPE_SRM);
    ASSERT_TRUE(graph.Compile({W1node, W2node, W3node}, 1));
  }

  void InitParam() {
    auto& W1 = param.insert<tsr_t>(W1node->name());
    W1.resize(W1node->shape());
    W1.ones();

    auto& W2 = param.insert<srm_t>(W2node->name());
    W2.set_col(W2node->shape()[1]);
    W2.get_row_no_init(1);
    W2.get_row_no_init(2);

    auto& W3 = param.insert<srm_t>(W3node->name());
    W3.set_col(W3node->shape()[1]);
    W3.get_scalar_no_init(0);
    W3.get_scalar_no_init(2);
    W3.get_scalar_no_init(4);
  }

  void UpdateParam() {
    auto& W2 = param.get<srm_t>(W2node->name());
    W2.get_row_no_init(0)[0] = 1;
    W2.get_row_no_init(0)[1] = 1;
    W2.get_row_no_init(1)[0] = 2;
    W2.get_row_no_init(1)[1] = 2;
    W2.get_row_no_init(2)[0] = 4;
    W2.get_row_no_init(2)[1] = 4;

    auto& W3 = param.get<srm_t>(W3node->name());
    W3.get_scalar_no_init(2) = 2;
    W3.get_scalar_no_init(4) = 4;
  }

  void TestCollect(freq_t update_threshold, float_t distance_threshold,
                   const id_set_t& expected_id_set) {
    InitGraph();
    InitParam();

    OLStore ol_store;
    ol_store.set_update_threshold(update_threshold);
    ol_store.set_distance_threshold(distance_threshold);
    ol_store.Init(&graph, &param);
    ol_store.InitParam();

    {
      TensorMap _param;
      _param.insert<srm_t>(W2node->name()) =
          srm_t{{0, 1, 2}, {{1, 1}, {1, 1}, {1, 1}}};
      ol_store.Update(&_param);
      ol_store.Update(&_param);
    }

    {
      TensorMap _param;
      _param.insert<srm_t>(W3node->name()) = srm_t{{2, 4}, {{1}, {1}}};
      ol_store.Update(&_param);
      ol_store.Update(&_param);
      ol_store.Update(&_param);
    }

    UpdateParam();

    // W2:
    //   0: update=2, not exists -> {1, 1}
    //   1: update=2, {0, 0} -> {2, 2}, distance=2.8284271
    //   2: update=2, {0, 0} -> {4, 4}, distance=5.6568542
    // W3:
    //   2, update=3, {0} -> {2}, distance=2
    //   4, update=3, {0} -> {4}, distance=4

    id_set_t id_set = ol_store.Collect();
    EXPECT_EQ(id_set, expected_id_set);
  }
};

TEST_F(OLStoreTest, Collect_update_threshold0_distance_threshold0) {
  TestCollect(0, 0, id_set_t({0, 1, 2, 4}));
}

TEST_F(OLStoreTest, Collect_update_threshold0_distance_threshold10) {
  TestCollect(0, 10, id_set_t({0, 1, 2, 4}));
}

TEST_F(OLStoreTest, Collect_update_threshold2_distance_threshold0) {
  TestCollect(2, 0, id_set_t({0, 1, 2, 4}));
}

TEST_F(OLStoreTest, Collect_update_threshold2_distance_threshold10) {
  TestCollect(2, 10, id_set_t({0, 2, 4}));
}

TEST_F(OLStoreTest, Collect_update_threshold3_distance_threshold0) {
  TestCollect(3, 0, id_set_t({0, 1, 2, 4}));
}

TEST_F(OLStoreTest, Collect_update_threshold3_distance_threshold10) {
  TestCollect(3, 10, id_set_t({0}));
}

TEST_F(OLStoreTest, Collect_update_threshold4_distance_threshold0) {
  TestCollect(4, 0, id_set_t({0, 1, 2, 4}));
}

TEST_F(OLStoreTest, Collect_update_threshold4_distance_threshold3) {
  TestCollect(4, 3, id_set_t({0, 2, 4}));
}

TEST_F(OLStoreTest, Collect_update_threshold4_distance_threshold5) {
  TestCollect(4, 5, id_set_t({0, 2}));
}

}  // namespace deepx_core
