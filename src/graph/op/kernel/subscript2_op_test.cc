// Copyright 2021 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class Subscript2ForwardTest : public testing::Test, public DataType {};

TEST_F(Subscript2ForwardTest, Subscript2_Xshape23_Yshape3_axis0) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  ConstantNode Y("Y", Shape(3), {0, 1, 1});
  Subscript2Node Z("Z", &X, &Y, 0);
  tsr_t expected_Z{0, 4, 5};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(Subscript2ForwardTest, Subscript2_Xshape23_Yshape2_axis1) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  ConstantNode Y("Y", Shape(2), {1, 2});
  Subscript2Node Z("Z", &X, &Y, 1);
  tsr_t expected_Z{1, 5};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(Subscript2ForwardTest, Subscript2_Xshape234_Yshape34_axis0) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  ConstantNode Y("Y", Shape(3, 4),
                 {0, 1, 0, 1,  //
                  0, 0, 1, 1,  //
                  1, 1, 0, 0});
  Subscript2Node Z("Z", &X, &Y, 0);
  tsr_t expected_Z{{0, 13, 2, 15},  //
                   {4, 5, 18, 19},  //
                   {20, 21, 10, 11}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(Subscript2ForwardTest, Subscript2_Xshape234_Yshape24_axis1) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  ConstantNode Y("Y", Shape(2, 4),
                 {0, 1, 0, 1,  //
                  1, 2, 1, 2});
  Subscript2Node Z("Z", &X, &Y, 1);
  tsr_t expected_Z{{0, 5, 2, 7},  //
                   {16, 21, 18, 23}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(Subscript2ForwardTest, Subscript2_Xshape234_Yshape23_axis2) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  ConstantNode Y("Y", Shape(2, 3),
                 {0, 1, 2,  //
                  1, 2, 3});
  Subscript2Node Z("Z", &X, &Y, 2);
  tsr_t expected_Z{{0, 5, 10},  //
                   {13, 18, 23}};
  CheckOpForward(&Z, 0, expected_Z);
}

class Subscript2BackwardTest : public testing::Test {
 protected:
  const std::vector<std::tuple<Shape, Shape, int>> SHAPE_AXIS_TUPLES = {
      std::make_tuple(Shape(2, 3), Shape(3), 0),
      std::make_tuple(Shape(2, 3), Shape(2), 1),
      std::make_tuple(Shape(2, 3, 4), Shape(3, 4), 0),
      std::make_tuple(Shape(2, 3, 4), Shape(2, 4), 1),
      std::make_tuple(Shape(2, 3, 4), Shape(2, 3), 2)};
};

TEST_F(Subscript2BackwardTest, Subscript2) {
  for (const auto& entry : SHAPE_AXIS_TUPLES) {
    const Shape& Xshape = std::get<0>(entry);
    const Shape& Yshape = std::get<1>(entry);
    int axis = std::get<2>(entry);
    VariableNode X("X", Xshape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ConstantNode Y("Y", Yshape, TENSOR_INITIALIZER_TYPE_RAND_INT, 0,
                   Xshape[axis]);
    Subscript2Node Z("Z", &X, &Y, axis);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
