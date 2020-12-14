// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class EmbeddingLookupForwardTest : public testing::Test, public DataType {
 protected:
  const csr_t X_{{0, 1, 4, 6, 7, 10, 14},
                 {1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7},
                 {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3}};

 protected:
  void TestTSR(const Shape& Wshape, const tsr_t& expected_Z) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode W("W", Wshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    EmbeddingLookupNode Z("Z", &X, &W);
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
  }

  void TestSRM(const Shape& Wshape, const tsr_t& expected_Z) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode W("W", Wshape, TENSOR_TYPE_SRM);
    EmbeddingLookupNode Z("Z", &X, &W);
    auto post_param_initializer = [](std::default_random_engine& /*engine*/,
                                     TensorMap* param) {
      auto& W = param->get<srm_t>("W");
      for (int i = 0; i < W.col(); ++i) {
        W.get_row_no_init(1)[i] = 1;
        W.get_row_no_init(2)[i] = 2;
        W.get_row_no_init(3)[i] = 3;
        W.get_row_no_init(4)[i] = 4;
        W.get_row_no_init(5)[i] = 5;
        W.get_row_no_init(6)[i] = 6;
        W.get_row_no_init(7)[i] = 7;
      }
    };
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpForward(&Z, 0, expected_Z, nullptr, post_param_initializer,
                   inst_initializer);
  }
};

TEST_F(EmbeddingLookupForwardTest, EmbeddingLookup_TSR_Wcol1) {
  tsr_t expected_Z{1, 9, 17, 14, 12, 66};
  expected_Z.reshape(6, 1);
  TestTSR(Shape(10, 1), expected_Z);
}

TEST_F(EmbeddingLookupForwardTest, EmbeddingLookup_TSR_Wcol4) {
  tsr_t expected_Z{{4, 5, 6, 7},      //
                   {36, 39, 42, 45},  //
                   {68, 71, 74, 77},  //
                   {56, 58, 60, 62},  //
                   {48, 54, 60, 66},  //
                   {264, 276, 288, 300}};
  TestTSR(Shape(10, 4), expected_Z);
}

TEST_F(EmbeddingLookupForwardTest, EmbeddingLookup_SRM_Wcol1) {
  tsr_t expected_Z{1, 9, 17, 14, 12, 66};
  expected_Z.reshape(6, 1);
  TestSRM(Shape(0, 1), expected_Z);
}

TEST_F(EmbeddingLookupForwardTest, EmbeddingLookup_SRM_Wcol4) {
  tsr_t expected_Z{{1, 1, 1, 1},      //
                   {9, 9, 9, 9},      //
                   {17, 17, 17, 17},  //
                   {14, 14, 14, 14},  //
                   {12, 12, 12, 12},  //
                   {66, 66, 66, 66}};
  TestSRM(Shape(0, 4), expected_Z);
}

class EmbeddingLookupBackwardTest : public EmbeddingLookupForwardTest {
 protected:
  void TestTSR(const Shape& Wshape) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode W("W", Wshape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    EmbeddingLookupNode Z("Z", &X, &W);
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
  }

  void TestSRM(const Shape& Wshape) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode W("W", Wshape, TENSOR_TYPE_SRM, TENSOR_INITIALIZER_TYPE_RANDN,
                   0, 1);
    EmbeddingLookupNode Z("Z", &X, &W);
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

TEST_F(EmbeddingLookupBackwardTest, EmbeddingLookup_TSR_Wcol1) {
  TestTSR(Shape(10, 1));
}

TEST_F(EmbeddingLookupBackwardTest, EmbeddingLookup_TSR_Wcol4) {
  TestTSR(Shape(10, 4));
}

TEST_F(EmbeddingLookupBackwardTest, EmbeddingLookup_SRM_Wcol1) {
  TestSRM(Shape(0, 1));
}

TEST_F(EmbeddingLookupBackwardTest, EmbeddingLookup_SRM_Wcol4) {
  TestSRM(Shape(0, 4));
}

}  // namespace deepx_core
