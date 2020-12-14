// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class AddNForwardTest : public testing::Test, public DataType {};

TEST_F(AddNForwardTest, AddN_Xshape23) {
  ConstantNode X1("X1", Shape(2, 3),
                  {0, 1, 2,  //
                   3, 4, 5});
  ConstantNode X2("X2", Shape(2, 3),
                  {0, 1, 2,  //
                   3, 4, 5});
  AddNNode Z("Z", {&X1, &X2});
  tsr_t expected_Z{{0, 2, 4},  //
                   {6, 8, 10}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(AddNForwardTest, AddN_Xshape234) {
  ConstantNode X1("X1", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                         4,  5,  6,  7,   //
                                         8,  9,  10, 11,  //
                                         12, 13, 14, 15,  //
                                         16, 17, 18, 19,  //
                                         20, 21, 22, 23});
  ConstantNode X2("X2", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                         4,  5,  6,  7,   //
                                         8,  9,  10, 11,  //
                                         12, 13, 14, 15,  //
                                         16, 17, 18, 19,  //
                                         20, 21, 22, 23});
  AddNNode Z("Z", {&X1, &X2});
  tsr_t expected_Z{0,  2,  4,  6,   //
                   8,  10, 12, 14,  //
                   16, 18, 20, 22,  //
                   24, 26, 28, 30,  //
                   32, 34, 36, 38,  //
                   40, 42, 44, 46};
  expected_Z.reshape(2, 3, 4);
  CheckOpForward(&Z, 0, expected_Z);
}

class AddNBackwardTest : public testing::Test {
 protected:
  const std::vector<Shape> SHAPES = {Shape(1), Shape(2, 3), Shape(2, 3, 4),
                                     Shape(2, 3, 4, 5)};
};

TEST_F(AddNBackwardTest, AddN) {
  for (const Shape& shape : SHAPES) {
    VariableNode X1("X1", shape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode X2("X2", shape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    AddNNode Z("Z", {&X1, &X2});
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
