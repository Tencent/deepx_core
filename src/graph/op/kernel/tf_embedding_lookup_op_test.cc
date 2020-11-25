// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class TFEmbeddingLookupForwardTest : public testing::Test, public DataType {
 protected:
  const tsri_t X_{{0, 1}, {13, 14}};

 protected:
  void TestTSR(const Shape& Wshape, const tsr_t& expected_Z) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_TSRI);
    VariableNode W("W", Wshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    TFEmbeddingLookupNode Z("Z", &X, &W);
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<tsri_t>("X") = X_;
    };
    CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
  }

  void TestSRM(const Shape& Wshape, const tsr_t& expected_Z) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_TSRI);
    VariableNode W("W", Wshape, TENSOR_TYPE_SRM);
    TFEmbeddingLookupNode Z("Z", &X, &W);
    auto post_param_initializer = [](std::default_random_engine& /*engine*/,
                                     TensorMap* param) {
      auto& W = param->get<srm_t>("W");
      for (int i = 0; i < W.col(); ++i) {
        W.get_row_no_init(0)[i] = 0;
        W.get_row_no_init(1)[i] = 1;
        W.get_row_no_init(13)[i] = 13;
        W.get_row_no_init(14)[i] = 14;
      }
    };
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<tsri_t>("X") = X_;
    };
    CheckOpForward(&Z, 0, expected_Z, nullptr, post_param_initializer,
                   inst_initializer);
  }
};

TEST_F(TFEmbeddingLookupForwardTest, TFEmbeddingLookup_TSR_Wcol1) {
  tsr_t expected_Z{0, 1,  //
                   3, 4};
  expected_Z.reshape(2, 2, 1);
  TestTSR(Shape(10, 1), expected_Z);
}

TEST_F(TFEmbeddingLookupForwardTest, TFEmbeddingLookup_TSR_Wcol4) {
  tsr_t expected_Z{0,  1,  2,  3,   //
                   4,  5,  6,  7,   //
                   12, 13, 14, 15,  //
                   16, 17, 18, 19};
  expected_Z.reshape(2, 2, 4);
  TestTSR(Shape(10, 4), expected_Z);
}

TEST_F(TFEmbeddingLookupForwardTest, TFEmbeddingLookup_SRM_Wcol1) {
  tsr_t expected_Z{0, 1,  //
                   13, 14};
  expected_Z.reshape(2, 2, 1);
  TestSRM(Shape(0, 1), expected_Z);
}

TEST_F(TFEmbeddingLookupForwardTest, TFEmbeddingLookup_SRM_Wcol4) {
  tsr_t expected_Z{0,  0,  0,  0,   //
                   1,  1,  1,  1,   //
                   13, 13, 13, 13,  //
                   14, 14, 14, 14};
  expected_Z.reshape(2, 2, 4);
  TestSRM(Shape(0, 4), expected_Z);
}

class TFEmbeddingLookupBackwardTest : public TFEmbeddingLookupForwardTest {
 protected:
  void TestTSR(const Shape& Wshape) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_TSRI);
    VariableNode W("W", Wshape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    TFEmbeddingLookupNode Z("Z", &X, &W);
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<tsri_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
  }

  void TestSRM(const Shape& Wshape) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_TSRI);
    VariableNode W("W", Wshape, TENSOR_TYPE_SRM, TENSOR_INITIALIZER_TYPE_RANDN,
                   0, 1);
    TFEmbeddingLookupNode Z("Z", &X, &W);
    auto post_param_initializer = [this](std::default_random_engine& engine,
                                         TensorMap* param) {
      auto& W = param->get<srm_t>("W");
      for (int i = 0; i < X_.total_dim(); ++i) {
        W.get_row(engine, X_.data(i));
      }
    };
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<tsri_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, post_param_initializer, inst_initializer);
  }
};

TEST_F(TFEmbeddingLookupBackwardTest, TFEmbeddingLookup_TSR_Wcol1) {
  TestTSR(Shape(10, 1));
}

TEST_F(TFEmbeddingLookupBackwardTest, TFEmbeddingLookup_TSR_Wcol4) {
  TestTSR(Shape(10, 4));
}

TEST_F(TFEmbeddingLookupBackwardTest, TFEmbeddingLookup_SRM_Wcol1) {
  TestSRM(Shape(0, 1));
}

TEST_F(TFEmbeddingLookupBackwardTest, TFEmbeddingLookup_SRM_Wcol4) {
  TestSRM(Shape(0, 4));
}

}  // namespace deepx_core
