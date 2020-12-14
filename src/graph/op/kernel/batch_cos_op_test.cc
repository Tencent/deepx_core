// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class BatchCosForwardTest : public testing::Test, public DataType {};

TEST_F(BatchCosForwardTest, BatchCos_1) {
  ConstantNode X("X", Shape(2, 3),
                 {0, 1, 2,  //
                  3, 4, 5});
  ConstantNode Y("Y", Shape(2, 3),
                 {1, 1, 1,  //
                  1, 1, 1});
  BatchCosNode Z("Z", &X, &Y);
  tsr_t expected_Z{(float_t)0.7745967, (float_t)0.9797959};
  expected_Z.reshape(2, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(BatchCosForwardTest, BatchCos_2) {
  ConstantNode X("X", Shape(2, 3),
                 {0, 1, 2,  //
                  3, 4, 5});
  ConstantNode Y("Y", Shape(2, 3),
                 {0, 0, 0,  //
                  0, 0, 0});
  BatchCosNode Z("Z", &X, &Y);
  tsr_t expected_Z{0, 0};
  expected_Z.reshape(2, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

class BatchCosBackwardTest : public testing::Test {};

TEST_F(BatchCosBackwardTest, BatchCos) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchCosNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
