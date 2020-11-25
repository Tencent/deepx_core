// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class FullyConnectBackwardTest : public testing::Test {};

TEST_F(FullyConnectBackwardTest, FullyConnect) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode W("W", Shape(3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  FullyConnectNode Z("Y", &X, &W);
  CheckOpBackward(&Z, 0);
}

TEST_F(FullyConnectBackwardTest, FullyConnect_b) {
  VariableNode X("X", Shape(2, 3), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode W("W", Shape(3, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode b("b", Shape(1, 4), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  FullyConnectNode Z("Y", &X, &W, &b);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
