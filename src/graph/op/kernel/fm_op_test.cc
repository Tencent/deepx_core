// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

/************************************************************************/
/* BatchFMInteraction */
/************************************************************************/
class BatchFMInteractionForwardTest : public testing::Test, public DataType {};

TEST_F(BatchFMInteractionForwardTest, BatchFMInteraction) {
  ConstantNode X("X", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
  BatchFMInteractionNode Z("Z", &X);
  tsr_t expected_Z{0,   5,   12,  21,   //
                   0,   9,   20,  33,   //
                   32,  45,  60,  77,   //
                   192, 221, 252, 285,  //
                   240, 273, 308, 345,  //
                   320, 357, 396, 437};
  expected_Z.reshape(2, 3, 4);
  CheckOpForward(&Z, 0, expected_Z);
}

class BatchFMInteractionBackwardTest : public testing::Test {};

TEST_F(BatchFMInteractionBackwardTest, BatchFMInteraction) {
  VariableNode X("X", Shape(2, 4, 5), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchFMInteractionNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

/************************************************************************/
/* BatchFMInteraction2 */
/************************************************************************/
class BatchFMInteraction2ForwardTest : public testing::Test, public DataType {};

TEST_F(BatchFMInteraction2ForwardTest, BatchFMInteraction2) {
  ConstantNode X("X", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
  ConstantNode Y("Y", Shape(2, 1, 4), TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
  BatchFMInteraction2Node Z("Z", &X, &Y);
  tsr_t expected_Z{0,  1,   4,   9,    //
                   0,  5,   12,  21,   //
                   0,  9,   20,  33,   //
                   48, 65,  84,  105,  //
                   64, 85,  108, 133,  //
                   80, 105, 132, 161};
  expected_Z.reshape(2, 3, 4);
  CheckOpForward(&Z, 0, expected_Z);
}

class BatchFMInteraction2BackwardTest : public testing::Test {};

TEST_F(BatchFMInteraction2BackwardTest, BatchFMInteraction2) {
  VariableNode X("X", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchFMInteraction2Node Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

/************************************************************************/
/* BatchFMQuadratic */
/************************************************************************/
class BatchFMQuadraticForwardTest : public testing::Test, public DataType {
 protected:
  const csr_t X_{{0, 1, 4, 6, 7, 10, 14},
                 {1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7},
                 {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3}};

 protected:
  void Test(const Shape& Vshape, const tsr_t& expected_Z) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode V("V", Vshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    BatchFMQuadraticNode Z("Z", &X, &V);
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
  }
};

TEST_F(BatchFMQuadraticForwardTest, BatchFMQuadratic_Vcol1) {
  tsr_t expected_Z{0, 26, 60, 0, 44, 1611};
  expected_Z.reshape(6, 1);
  Test(Shape(10, 1), expected_Z);
}

TEST_F(BatchFMQuadraticForwardTest, BatchFMQuadratic_Vcol4) {
  tsr_t expected_Z{0, 2138, 4396, 0, 4136, 118116};
  expected_Z.reshape(6, 1);
  Test(Shape(10, 4), expected_Z);
}

class BatchFMQuadraticBackwardTest : public BatchFMQuadraticForwardTest {
 protected:
  void Test(const Shape& Vshape) {
    InstanceNode X("X", Shape(-1, 0), TENSOR_TYPE_CSR);
    VariableNode V("V", Vshape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    BatchFMQuadraticNode Z("Z", &X, &V);
    auto inst_initializer = [this](Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
  }
};

TEST_F(BatchFMQuadraticBackwardTest, BatchFMQuadratic_Vcol1) {
  Test(Shape(10, 1));
}

TEST_F(BatchFMQuadraticBackwardTest, BatchFMQuadratic_Vcol4) {
  Test(Shape(10, 4));
}

/************************************************************************/
/* BatchGroupFMQuadratic */
/************************************************************************/
class BatchGroupFMQuadraticForwardTest : public testing::Test,
                                         public DataType {};

TEST_F(BatchGroupFMQuadraticForwardTest, BatchGroupFMQuadratic) {
  ConstantNode X("X", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
  BatchGroupFMQuadraticNode Z("Z", &X);
  tsr_t expected_Z{314, 3626};
  expected_Z.reshape(2, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

class BatchGroupFMQuadraticBackwardTest : public testing::Test {};

TEST_F(BatchGroupFMQuadraticBackwardTest, BatchGroupFMQuadratic) {
  VariableNode X("X", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchGroupFMQuadraticNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

/************************************************************************/
/* BatchGroupFMQuadratic2 */
/************************************************************************/
class BatchGroupFMQuadratic2ForwardTest : public testing::Test,
                                          public DataType {};

TEST_F(BatchGroupFMQuadratic2ForwardTest, BatchGroupFMQuadratic) {
  ConstantNode X("X", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
  BatchGroupFMQuadratic2Node Z("Z", &X);
  tsr_t expected_Z{{32, 59, 92, 131},  //
                   {752, 851, 956, 1067}};
  CheckOpForward(&Z, 0, expected_Z);
}

class BatchGroupFMQuadratic2BackwardTest : public testing::Test {};

TEST_F(BatchGroupFMQuadratic2BackwardTest, BatchGroupFMQuadratic2) {
  VariableNode X("X", Shape(2, 3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchGroupFMQuadratic2Node Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
