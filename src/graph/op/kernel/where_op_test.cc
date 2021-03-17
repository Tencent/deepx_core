// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class WhereForwardTest : public testing::Test, public DataType {};

TEST_F(WhereForwardTest, Where) {
  Shape shape(5);
  ConstantNode C("C", shape, {0, 1, 0, 1, 0});
  ConstantNode X("X", shape, {0, 1, 2, 3, 4});
  ConstantNode Y("Y", shape, {5, 6, 7, 8, 9});
  WhereNode Z("Z", &C, &X, &Y);
  tsr_t expected_Z{5, 1, 7, 3, 9};
  CheckOpForward(&Z, 0, expected_Z);
}

class WhereBackwardTest : public testing::Test {};

TEST_F(WhereBackwardTest, Where) {
  Shape shape(5);
  ConstantNode C("C", shape, {0, 1, 0, 1, 0});
  VariableNode X("X", shape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode Y("Y", shape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  WhereNode Z("Z", &C, &X, &Y);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
