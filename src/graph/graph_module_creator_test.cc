// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/group_config.h>
#include <deepx_core/graph/graph_module_creator.h>
#include <deepx_core/graph/instance_reader.h>
#include <deepx_core/graph/variable_scope.h>
#include "op/op_test.h"

namespace deepx_core {

/************************************************************************/
/* embedding creator */
/************************************************************************/
class EmbeddingCreatorBackwardTest : public testing::Test, public DataType {
 protected:
  const uint16_t GROUP_ID1 = 1;
  const uint16_t GROUP_ID2 = 2;
  const uint16_t GROUP_ID3 = 3;
  const csr_t X_{{0, 1, 4, 6, 7, 10, 14},
                 {ll_sparse_tensor_t::make_feature_id(GROUP_ID1, 1),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID2, 2),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID3, 3),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID1, 4),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID2, 5),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID3, 6),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID1, 7),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID2, 1),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID3, 2),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID1, 3),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID2, 4),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID3, 5),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID1, 6),
                  ll_sparse_tensor_t::make_feature_id(GROUP_ID2, 7)},
                 {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3}};
  std::vector<GroupConfigItem3> items;

 protected:
  void SetUp() override {
    GroupConfigItem3 item;
    item.group_id = GROUP_ID1;
    item.embedding_row = 10;
    item.embedding_col = 4;
    items.emplace_back(item);
    item.group_id = GROUP_ID2;
    item.embedding_row = 10;
    item.embedding_col = 4;
    items.emplace_back(item);
    item.group_id = GROUP_ID3;
    item.embedding_row = 10;
    item.embedding_col = 4;
    items.emplace_back(item);
  }

 protected:
  void Test(GraphNode* Z) {
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>(X_NAME) = X_;
    };
    ReleaseVariable();
    CheckOpBackward(Z, 1, nullptr, nullptr, inst_initializer);
  }

  void TestGroupEmbeddingLookup_sparse(GraphNode* Z) {
    auto post_param_initializer = [this](std::default_random_engine& engine,
                                         TensorMap* param) {
      auto& W1 = param->get<srm_t>("ZW" + std::to_string(GROUP_ID1));
      auto& W2 = param->get<srm_t>("ZW" + std::to_string(GROUP_ID2));
      auto& W3 = param->get<srm_t>("ZW" + std::to_string(GROUP_ID3));
      for (size_t i = 0; i < X_.col_size(); ++i) {
        W1.get_row(engine, X_.col(i));
        W2.get_row(engine, X_.col(i));
        W3.get_row(engine, X_.col(i));
      }
    };
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>(X_NAME) = X_;
    };
    ReleaseVariable();
    CheckOpBackward(Z, 1, nullptr, post_param_initializer, inst_initializer);
  }

  void TestGroupEmbeddingLookup2_sparse(GraphNode* Z) {
    auto post_param_initializer = [this](std::default_random_engine& engine,
                                         TensorMap* param) {
      auto& W = param->get<srm_t>("ZW");
      for (size_t i = 0; i < X_.col_size(); ++i) {
        W.get_row(engine, X_.col(i));
      }
    };
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>(X_NAME) = X_;
    };
    ReleaseVariable();
    CheckOpBackward(Z, 1, nullptr, post_param_initializer, inst_initializer);
  }
};

TEST_F(EmbeddingCreatorBackwardTest, WideGroupEmbeddingLookup_sparse0) {
  auto* Z = WideGroupEmbeddingLookup("Z", GetX(), items, 0);
  Test(Z);
}

TEST_F(EmbeddingCreatorBackwardTest, WideGroupEmbeddingLookup_sparse1) {
  auto* Z = WideGroupEmbeddingLookup("Z", GetX(), items, 1);
  TestGroupEmbeddingLookup_sparse(Z);
}

TEST_F(EmbeddingCreatorBackwardTest, WideGroupEmbeddingLookup2_sparse0) {
  auto* Z = WideGroupEmbeddingLookup2("Z", GetX(), items, 0);
  Test(Z);
}

TEST_F(EmbeddingCreatorBackwardTest, WideGroupEmbeddingLookup2_sparse1) {
  auto* Z = WideGroupEmbeddingLookup2("Z", GetX(), items, 1);
  TestGroupEmbeddingLookup2_sparse(Z);
}

TEST_F(EmbeddingCreatorBackwardTest, DeepGroupEmbeddingLookup_sparse0) {
  auto* Z = DeepGroupEmbeddingLookup("Z", GetX(), items, 0);
  Test(Z);
}

TEST_F(EmbeddingCreatorBackwardTest, DeepGroupEmbeddingLookup_sparse1) {
  auto* Z = DeepGroupEmbeddingLookup("Z", GetX(), items, 1);
  TestGroupEmbeddingLookup_sparse(Z);
}

TEST_F(EmbeddingCreatorBackwardTest, DeepGroupEmbeddingLookup2_sparse0) {
  auto* Z = DeepGroupEmbeddingLookup2("Z", GetX(), items, 0);
  Test(Z);
}

TEST_F(EmbeddingCreatorBackwardTest, DeepGroupEmbeddingLookup2_sparse1) {
  auto* Z = DeepGroupEmbeddingLookup2("Z", GetX(), items, 1);
  TestGroupEmbeddingLookup2_sparse(Z);
}

/************************************************************************/
/* building block creator */
/************************************************************************/
class BuildingBlockCreatorForwardTest : public testing::Test,
                                        public DataType {};

class BuildingBlockCreatorBackwardTest : public testing::Test {};

TEST_F(BuildingBlockCreatorBackwardTest, StackedFullyConnect_sigmoid) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Z = StackedFullyConnect("Z", X, {3, 4}, "sigmoid");
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, StackedFullyConnect_tanh) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Z = StackedFullyConnect("Z", X, {3, 4}, "tanh");
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, FullyConnect) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Z = FullyConnect("Z", X, 4);
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, AddBias) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Z = AddBias("Z", X);
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, SelfAttention) {
  auto* X = GetVariableRandn("X", Shape(2, 3, 4));
  auto* Z = SelfAttention("Z", X, 5);
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, CrossNet) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Z = CrossNet("Z", X, 3);
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, CIN) {
  auto* X = GetVariableRandn("X", Shape(2, 3, 4));
  auto* Z = CIN("Z", X, {8, 8});
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, RNNCell) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Hout = RNNCell("Z", X, nullptr, 4);
  ReleaseVariable();
  CheckOpBackward(Hout, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, RNNCell_Hin) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Hin = GetVariableRandn("Hin", Shape(2, 4));
  auto* Hout = RNNCell("Z", X, Hin, 4);
  ReleaseVariable();
  CheckOpBackward(Hout, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, LSTMCell_Cout) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto CHout = LSTMCell("Z", X, {}, 4, 1);
  ReleaseVariable();
  CheckOpBackward(CHout[0], 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, LSTMCell_Hout) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto CHout = LSTMCell("Z", X, {}, 4, 2);
  ReleaseVariable();
  CheckOpBackward(CHout[1], 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, LSTMCell_CHin_Cout) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Cin = GetVariableRandn("Cin", Shape(2, 4));
  auto* Hin = GetVariableRandn("Hin", Shape(2, 4));
  auto CHout = LSTMCell("Z", X, {Cin, Hin}, 4, 1);
  ReleaseVariable();
  CheckOpBackward(CHout[0], 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, LSTMCell_CHin_Hout) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Cin = GetVariableRandn("Cin", Shape(2, 4));
  auto* Hin = GetVariableRandn("Hin", Shape(2, 4));
  auto CHout = LSTMCell("Z", X, {Cin, Hin}, 4, 2);
  ReleaseVariable();
  CheckOpBackward(CHout[1], 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, GRUCell) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Hout = GRUCell("Z", X, nullptr, 4);
  ReleaseVariable();
  CheckOpBackward(Hout, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, GRUCell_Hin) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto* Hin = GetVariableRandn("Hin", Shape(2, 4));
  auto* Hout = GRUCell("Z", X, Hin, 4);
  ReleaseVariable();
  CheckOpBackward(Hout, 1);
}

TEST_F(BuildingBlockCreatorForwardTest, Split_1) {
  auto* X = Constant("X", Shape(2, 3),
                     {0, 1, 2,  //
                      3, 4, 5});
  auto S = Split("S", X, 1, 3);
  auto* Z = Concat("Z", S, 1);
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5}};
  CheckOpForward(Z, 1, expected_Z);
}

TEST_F(BuildingBlockCreatorForwardTest, Split_2) {
  auto* X = Constant("X", Shape(2, 3),
                     {0, 1, 2,  //
                      3, 4, 5});
  auto S = Split("S", X, 1, {1, 2});
  auto* Z = Concat("Z", S, 1);
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5}};
  CheckOpForward(Z, 1, expected_Z);
}

TEST_F(BuildingBlockCreatorBackwardTest, Split_1) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto S = Split("S", X, 1, 3);
  auto* Z = Concat("Z", S, 1);
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

TEST_F(BuildingBlockCreatorBackwardTest, Split_2) {
  auto* X = GetVariableRandn("X", Shape(2, 3));
  auto S = Split("S", X, 1, {1, 2});
  auto* Z = Concat("Z", S, 1);
  ReleaseVariable();
  CheckOpBackward(Z, 1);
}

}  // namespace deepx_core
