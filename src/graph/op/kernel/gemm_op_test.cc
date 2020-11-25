// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class GEMMBackwardTest : public testing::Test {};

TEST_F(GEMMBackwardTest, GEMM_transX0_transY0) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  GEMMNode Z("Z", &X, &Y, 0, 0);
  CheckOpBackward(&Z, 0);
}

TEST_F(GEMMBackwardTest, GEMM_transX1_transY0) {
  VariableNode X("X", Shape(3, 2), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  GEMMNode Z("Z", &X, &Y, 1, 0);
  CheckOpBackward(&Z, 0);
}

TEST_F(GEMMBackwardTest, GEMM_transX0_transY1) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(4, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  GEMMNode Z("Z", &X, &Y, 0, 1);
  CheckOpBackward(&Z, 0);
}

TEST_F(GEMMBackwardTest, GEMM_transX1_transY1) {
  VariableNode X("X", Shape(3, 2), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(4, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  GEMMNode Z("Z", &X, &Y, 1, 1);
  CheckOpBackward(&Z, 0);
}

class BatchGEMMBackwardTest : public testing::Test {};

TEST_F(BatchGEMMBackwardTest, BatchGEMM_transX0_transY0) {
  VariableNode X("X", Shape(5, 2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(5, 3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchGEMMNode Z("Z", &X, &Y, 0, 0);
  CheckOpBackward(&Z, 0);
}

TEST_F(BatchGEMMBackwardTest, BatchGEMM_transX1_transY0) {
  VariableNode X("X", Shape(5, 3, 2), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(5, 3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchGEMMNode Z("Z", &X, &Y, 1, 0);
  CheckOpBackward(&Z, 0);
}

TEST_F(BatchGEMMBackwardTest, BatchGEMM_transX0_transY1) {
  VariableNode X("X", Shape(5, 2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(5, 4, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchGEMMNode Z("Z", &X, &Y, 0, 1);
  CheckOpBackward(&Z, 0);
}

TEST_F(BatchGEMMBackwardTest, BatchGEMM_transX1_transY1) {
  VariableNode X("X", Shape(5, 3, 2), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", Shape(5, 4, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  BatchGEMMNode Z("Z", &X, &Y, 1, 1);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
