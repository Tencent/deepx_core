// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class ConstantForwardTest : public testing::Test, public DataType {};

TEST_F(ConstantForwardTest, Zeros) {
  ConstantNode X1("X1", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  ZerosNode X2("X2", Shape(2, 3));
  AddNode Z("Z", &X1, &X2);
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ConstantForwardTest, Ones) {
  ConstantNode X1("X1", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  OnesNode X2("X2", Shape(2, 3));
  AddNode Z("Z", &X1, &X2);
  tsr_t expected_Z{{1, 2, 3},  //
                   {4, 5, 6}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ConstantForwardTest, ZerosLike) {
  ConstantNode X1("X1", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  ZerosLikeNode X2("X2", &X1);
  AddNode Z("Z", &X1, &X2);
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ConstantForwardTest, OnesLike) {
  ConstantNode X1("X1", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  OnesLikeNode X2("X2", &X1);
  AddNode Z("Z", &X1, &X2);
  tsr_t expected_Z{{1, 2, 3},  //
                   {4, 5, 6}};
  CheckOpForward(&Z, 0, expected_Z);
}

class ConstantBackwardTest : public testing::Test {};

TEST_F(ConstantBackwardTest, Constant) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantNode X2("X2", Shape(2, 3), 0.5);
  ConstantNode X3("X3", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  AddNode Z1("Z1", &X1, &X2);
  AddNode Z("Z", &Z1, &X3);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, Zeros) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ZerosNode X2("X2", Shape(2, 3));
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, Ones) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  OnesNode X2("X2", Shape(2, 3));
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, RandomNormal) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  RandomNormalNode X2("X2", Shape(2, 3), 0, 1);
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, RandomUniform) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  RandomUniformNode X2("X2", Shape(2, 3), 0, 1);
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, ConstantLike) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ConstantLikeNode X2("X2", &X1, 0.5);
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, ZerosLike) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  ZerosLikeNode X2("X2", &X1);
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, OnesLike) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  OnesLikeNode X2("X2", &X1);
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, RandomNormalLike) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  RandomNormalLikeNode X2("X2", &X1, 0, 1);
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

TEST_F(ConstantBackwardTest, RandomUniformLike) {
  VariableNode X1("X1", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  RandomUniformLikeNode X2("X2", &X1, 0, 1);
  AddNode Z("Z", &X1, &X2);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
