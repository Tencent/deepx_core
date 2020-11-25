// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class LossBackwardTest : public testing::Test {};

TEST_F(LossBackwardTest, AbsoluteError) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 3, 4);
  AbsoluteErrorNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
  Y.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -4, -3);
  CheckOpBackward(&Z, 0);

  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
  Y.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -4, -3);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, SquareError) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  SquareErrorNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, BCELoss) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 0.1, 0.9);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, -1, 1);
  BCELossNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, BCELoss2) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 0.1, 0.9);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 0, 1);
  BCELoss2Node Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, SigmoidBCELoss) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, -1, 1);
  SigmoidBCELossNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, SigmoidBCELoss2) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 0, 1);
  SigmoidBCELoss2Node Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, BatchCELoss) {
  int batch = 30;
  int m = 5;
  VariableNode X("X", Shape(batch, m), TENSOR_INITIALIZER_TYPE_CONSTANT,
                 1.0 / m, 0);
  ConstantNode Y("Y", Shape(batch, 1), TENSOR_INITIALIZER_TYPE_RAND_INT, 0, m);
  BatchCELossNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, BatchCELoss2) {
  int batch = 30;
  int m = 5;
  VariableNode X("X", Shape(batch, m), TENSOR_INITIALIZER_TYPE_CONSTANT,
                 1.0 / m, 0);
  ConstantNode Y("Y", Shape(batch, m), TENSOR_INITIALIZER_TYPE_CONSTANT,
                 1.0 / m, 0);
  BatchCELoss2Node Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, BatchSoftmaxCELoss) {
  int batch = 30;
  int m = 5;
  VariableNode X("X", Shape(batch, m), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantNode Y("Y", Shape(batch, 1), TENSOR_INITIALIZER_TYPE_RAND_INT, 0, m);
  BatchSoftmaxCELossNode Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, BatchSoftmaxCELoss2) {
  int batch = 30;
  int m = 5;
  VariableNode X("X", Shape(batch, m), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantNode Y("Y", Shape(batch, m), TENSOR_INITIALIZER_TYPE_CONSTANT,
                 1.0 / m, 0);
  BatchSoftmaxCELoss2Node Z("Z", &X, &Y);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, FocalLoss) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, 0.1, 0.9);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, -1, 1);
  FocalLossNode Z("Z", &X, &Y, 0.5, 2);
  CheckOpBackward(&Z, 0);
}

TEST_F(LossBackwardTest, SigmoidFocalLoss) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantNode Y("Y", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RAND, -1, 1);
  SigmoidFocalLossNode Z("Z", &X, &Y, 0.5, 2);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
