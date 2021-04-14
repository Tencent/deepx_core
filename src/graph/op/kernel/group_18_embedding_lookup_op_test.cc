// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class Group18EmbeddingLookupBaseTest : public testing::Test, public DataType {
 protected:
  const int GROUP_ID1 = 3;
  const int GROUP_ID2 = 4;
  const int GROUP_ID3 = 5;
  const std::vector<int> GROUP_IDS = {GROUP_ID1, GROUP_ID2, GROUP_ID3};
  const csr_t X_{{0, 1, 4, 6, 7, 10, 14},
                 {ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID1, 1),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID2, 2),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID3, 3),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID1, 4),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID2, 5),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID3, 6),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID1, 7),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID2, 1),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID3, 2),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID1, 3),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID2, 4),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID3, 5),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID1, 6),
                  ll_sparse_tensor_t::group_18_make_feature_id(GROUP_ID2, 7)},
                 {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3}};
};

/************************************************************************/
/* Group18EmbeddingLookup */
/************************************************************************/
class Group18EmbeddingLookupBackwardTest
    : public Group18EmbeddingLookupBaseTest {};

TEST_F(Group18EmbeddingLookupBackwardTest, Group18EmbeddingLookup_TSR) {
  InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
  VariableNode W1("W1", Shape(10, 1), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode W2("W2", Shape(10, 2), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode W3("W3", Shape(10, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  Group18EmbeddingLookupNode Z("Z", &X, {&W1, &W2, &W3}, GROUP_IDS);
  auto inst_initializer = [this](Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
}

TEST_F(Group18EmbeddingLookupBackwardTest, Group18EmbeddingLookup_SRM) {
  InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
  VariableNode W1("W1", Shape(10, 1), TENSOR_TYPE_SRM,
                  TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode W2("W2", Shape(10, 2), TENSOR_TYPE_SRM,
                  TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode W3("W3", Shape(10, 4), TENSOR_TYPE_SRM,
                  TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  Group18EmbeddingLookupNode Z("Z", &X, {&W1, &W2, &W3}, GROUP_IDS);
  auto post_param_initializer = [this](std::default_random_engine& engine,
                                       TensorMap* param) {
    auto& W1 = param->get<srm_t>("W1");
    auto& W2 = param->get<srm_t>("W2");
    auto& W3 = param->get<srm_t>("W3");
    for (size_t i = 0; i < X_.col_size(); ++i) {
      W1.get_row(engine, X_.col(i));
      W2.get_row(engine, X_.col(i));
      W3.get_row(engine, X_.col(i));
    }
  };
  auto inst_initializer = [this](Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  CheckOpBackward(&Z, 0, nullptr, post_param_initializer, inst_initializer);
}

/************************************************************************/
/* Group18EmbeddingLookup2 */
/************************************************************************/
class Group18EmbeddingLookup2BackwardTest
    : public Group18EmbeddingLookupBaseTest {
 protected:
  void TestTSR(const Shape& Wshape) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode W("W", Wshape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    Group18EmbeddingLookup2Node Z("Z", &X, &W, GROUP_IDS);
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
  }

  void TestSRM(const Shape& Wshape) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode W("W", Wshape, TENSOR_TYPE_SRM, TENSOR_INITIALIZER_TYPE_RANDN,
                   0, 1);
    Group18EmbeddingLookup2Node Z("Z", &X, &W, GROUP_IDS);
    auto post_param_initializer = [this](std::default_random_engine& engine,
                                         TensorMap* param) {
      auto& W = param->get<srm_t>("W");
      for (size_t i = 0; i < X_.col_size(); ++i) {
        W.get_row(engine, X_.col(i));
      }
    };
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, post_param_initializer, inst_initializer);
  }
};

TEST_F(Group18EmbeddingLookup2BackwardTest, Group18EmbeddingLookup2_TSR_Wcol1) {
  TestTSR(Shape(10, 1));
}

TEST_F(Group18EmbeddingLookup2BackwardTest, Group18EmbeddingLookup2_TSR_Wcol4) {
  TestTSR(Shape(10, 4));
}

TEST_F(Group18EmbeddingLookup2BackwardTest, Group18EmbeddingLookup2_SRM_Wcol1) {
  TestSRM(Shape(0, 1));
}

TEST_F(Group18EmbeddingLookup2BackwardTest, Group18EmbeddingLookup2_SRM_Wcol4) {
  TestSRM(Shape(0, 4));
}

}  // namespace deepx_core
