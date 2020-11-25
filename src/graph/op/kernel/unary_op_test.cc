// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class UnaryElementWiseForwardTest : public testing::Test, public DataType {};

TEST_F(UnaryElementWiseForwardTest, ClipByValue) {
  ConstantNode X("X", Shape(2, 3), {3, 2, 1, 0, -1, -2});
  ClipByValueNode Z("Z", &X, -1, 2);
  tsr_t expected_Z{{2, 2, 1},  //
                   {0, -1, -1}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(UnaryElementWiseForwardTest, MatrixBandPart) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  MatrixBandPartNode Z("Z", &X, -1, 1);
  tsr_t expected_Z{{0, 1, 0},  //
                   {3, 4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

class UnaryElementWiseBackwardTest : public testing::Test {};

TEST_F(UnaryElementWiseBackwardTest, Sigmoid) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  SigmoidNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Tanh) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  TanhNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Relu) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  ReluNode Z("Z", &X);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, LeakyRelu) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  LeakyReluNode Z("Z", &X, 0.5);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Elu) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  EluNode Z("Z", &X, 0.5);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Selu) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  SeluNode Z("Z", &X, 0.5, 0.5);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Gelu) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  GeluNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, SoftPlus) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  SoftPlusNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Swish) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  SwishNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Exp) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ExpNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Log) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  LogNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Negate) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  NegateNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Inv) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  InvNode Z("Z", &X);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Sqrt) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  SqrtNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Cbrt) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  CbrtNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Square) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  SquareNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Cubic) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  CubicNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Dropout) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  DropoutNode Z("Z", &X, 0.5);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Abs) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  AbsNode Z("Z", &X);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, ClipByValue) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  ClipByValueNode Z("Z", &X, -3, 3);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, MatrixBandPart) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  MatrixBandPartNode Z("Z", &X, -1, 1);
  CheckOpBackward(&Z, 0);
}

TEST_F(UnaryElementWiseBackwardTest, Identity) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  IdentityNode Z("Z", &X);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
