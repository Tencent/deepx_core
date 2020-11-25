// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class BinaryElementWiseBackwardTest : public testing::Test {};

TEST_F(BinaryElementWiseBackwardTest, Add) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  AddNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(BinaryElementWiseBackwardTest, Sub) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  SubNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(BinaryElementWiseBackwardTest, Mul) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  MulNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(BinaryElementWiseBackwardTest, Div) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  DivNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);

  Y.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(BinaryElementWiseBackwardTest, Pow) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  VariableNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  PowNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
