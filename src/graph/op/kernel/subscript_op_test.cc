// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class SubscriptForwardTest : public testing::Test, public DataType {};

TEST_F(SubscriptForwardTest, Subscript_Xshape23_axis0_index0) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  SubscriptNode Z("Z", &X, 0, 0);
  tsr_t expected_Z{0, 1, 2};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptForwardTest, Subscript_Xshape23_axis1_index1) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  SubscriptNode Z("Z", &X, 1, 1);
  tsr_t expected_Z{1, 4};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptForwardTest, Subscript_Xshape234_axis0_index0) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  SubscriptNode Z("Z", &X, 0, 0);
  tsr_t expected_Z{{0, 1, 2, 3},  //
                   {4, 5, 6, 7},  //
                   {8, 9, 10, 11}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptForwardTest, Subscript_Xshape234_axis1_index1) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  SubscriptNode Z("Z", &X, 1, 1);
  tsr_t expected_Z{{4, 5, 6, 7},  //
                   {16, 17, 18, 19}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptForwardTest, Subscript_Xshape234_axis2_index2) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  SubscriptNode Z("Z", &X, 2, 2);
  tsr_t expected_Z{{2, 6, 10},  //
                   {14, 18, 22}};
  CheckOpForward(&Z, 0, expected_Z);
}

class SubscriptBackwardTest : public testing::Test {
 protected:
  const std::vector<std::tuple<Shape, int, int>> SHAPE_AXIS_INDEX_TUPLES = {
      std::make_tuple(Shape(2, 3), 0, 0),
      std::make_tuple(Shape(2, 3), 0, 1),
      std::make_tuple(Shape(2, 3), 1, 0),
      std::make_tuple(Shape(2, 3), 1, 1),
      std::make_tuple(Shape(2, 3), 1, 2),
      std::make_tuple(Shape(2, 3, 4), 0, 0),
      std::make_tuple(Shape(2, 3, 4), 0, 1),
      std::make_tuple(Shape(2, 3, 4), 1, 0),
      std::make_tuple(Shape(2, 3, 4), 1, 1),
      std::make_tuple(Shape(2, 3, 4), 1, 2),
      std::make_tuple(Shape(2, 3, 4), 2, 0),
      std::make_tuple(Shape(2, 3, 4), 2, 1),
      std::make_tuple(Shape(2, 3, 4), 2, 2),
      std::make_tuple(Shape(2, 3, 4), 2, 3)};
};

TEST_F(SubscriptBackwardTest, Subscript) {
  for (const auto& entry : SHAPE_AXIS_INDEX_TUPLES) {
    VariableNode X("X", std::get<0>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    SubscriptNode Z("Z", &X, std::get<1>(entry), std::get<2>(entry));
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
