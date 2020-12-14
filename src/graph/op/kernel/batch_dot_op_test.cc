// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class BatchDotForwardTest : public testing::Test, public DataType {};

TEST_F(BatchDotForwardTest, BatchDot) {
  ConstantNode X("X", Shape(2, 3),
                 {0, 1, 2,  //
                  3, 4, 5});
  ConstantNode Y("Y", Shape(2, 3),
                 {0, 1, 2,  //
                  3, 4, 5});
  BatchDotNode Z("Z", &X, &Y);
  tsr_t expected_Z{5, 50};
  expected_Z.reshape(2, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

class BatchDotBackwardTest : public testing::Test {};

TEST_F(BatchDotBackwardTest, BatchDot) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchDotNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
