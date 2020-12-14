// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class ConcatForwardTest : public testing::Test, public DataType {};

TEST_F(ConcatForwardTest, Concat_X1shape23_X2shape33_axis0) {
  ConstantNode X1("X1", Shape(2, 3),
                  {0, 1, 2,  //
                   3, 4, 5});
  ConstantNode X2("X2", Shape(3, 3),
                  {0, 1, 2,  //
                   3, 4, 5,  //
                   6, 7, 8});
  ConcatNode Z("Z", {&X1, &X2}, 0);
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5},  //
                   {0, 1, 2},  //
                   {3, 4, 5},  //
                   {6, 7, 8}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ConcatForwardTest, Concat_X1shape23_X2shape24_axis1) {
  ConstantNode X1("X1", Shape(2, 3),
                  {0, 1, 2,  //
                   3, 4, 5});
  ConstantNode X2("X2", Shape(2, 4),
                  {0, 1, 2, 3,  //
                   4, 5, 6, 7});
  ConcatNode Z("Z", {&X1, &X2}, 1);
  tsr_t expected_Z{{0, 1, 2, 0, 1, 2, 3},  //
                   {3, 4, 5, 4, 5, 6, 7}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ConcatForwardTest, Concat_X1shape234_X2shape134_axis0) {
  ConstantNode X1("X1", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                         4,  5,  6,  7,   //
                                         8,  9,  10, 11,  //
                                         12, 13, 14, 15,  //
                                         16, 17, 18, 19,  //
                                         20, 21, 22, 23});
  ConstantNode X2("X2", Shape(1, 3, 4),
                  {0, 1, 2, 3,  //
                   4, 5, 6, 7,  //
                   8, 9, 10, 11});
  ConcatNode Z("Z", {&X1, &X2}, 0);
  tsr_t expected_Z{0,  1,  2,  3,   //
                   4,  5,  6,  7,   //
                   8,  9,  10, 11,  //
                   12, 13, 14, 15,  //
                   16, 17, 18, 19,  //
                   20, 21, 22, 23,  //
                   0,  1,  2,  3,   //
                   4,  5,  6,  7,   //
                   8,  9,  10, 11};
  expected_Z.reshape(3, 3, 4);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ConcatForwardTest, Concat_X1shape234_X2shape214_axis1) {
  ConstantNode X1("X1", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                         4,  5,  6,  7,   //
                                         8,  9,  10, 11,  //
                                         12, 13, 14, 15,  //
                                         16, 17, 18, 19,  //
                                         20, 21, 22, 23});
  ConstantNode X2("X2", Shape(2, 1, 4),
                  {0, 1, 2, 3,  //
                   4, 5, 6, 7});
  ConcatNode Z("Z", {&X1, &X2}, 1);
  tsr_t expected_Z{0,  1,  2,  3,   //
                   4,  5,  6,  7,   //
                   8,  9,  10, 11,  //
                   0,  1,  2,  3,   //
                   12, 13, 14, 15,  //
                   16, 17, 18, 19,  //
                   20, 21, 22, 23,  //
                   4,  5,  6,  7};
  expected_Z.reshape(2, 4, 4);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ConcatForwardTest, Concat_X1shape234_X2shape231_axis2) {
  ConstantNode X1("X1", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                         4,  5,  6,  7,   //
                                         8,  9,  10, 11,  //
                                         12, 13, 14, 15,  //
                                         16, 17, 18, 19,  //
                                         20, 21, 22, 23});
  ConstantNode X2("X2", Shape(2, 3, 1),
                  {0, 1, 2,  //
                   3, 4, 5});
  ConcatNode Z("Z", {&X1, &X2}, 2);
  tsr_t expected_Z{0,  1,  2,  3,  0,  //
                   4,  5,  6,  7,  1,  //
                   8,  9,  10, 11, 2,  //
                   12, 13, 14, 15, 3,  //
                   16, 17, 18, 19, 4,  //
                   20, 21, 22, 23, 5};
  expected_Z.reshape(2, 3, 5);
  CheckOpForward(&Z, 0, expected_Z);
}

class ConcatBackwardTest : public testing::Test {
 protected:
  const std::vector<std::tuple<Shape, Shape, int>> SHAPE_AXIS_TUPLES = {
      std::make_tuple(Shape(2, 3), Shape(2, 3), 0),
      std::make_tuple(Shape(2, 3), Shape(2, 4), 1),
      std::make_tuple(Shape(2, 3), Shape(3, 3), 0),
      std::make_tuple(Shape(2, 3), Shape(2, 3), 1),
      std::make_tuple(Shape(2, 3, 4), Shape(3, 3, 4), 0),
      std::make_tuple(Shape(2, 3, 4), Shape(2, 4, 4), 1),
      std::make_tuple(Shape(2, 3, 4), Shape(2, 3, 5), -1),
      std::make_tuple(Shape(2, 3, 4, 5), Shape(2, 3, 5, 5), 2),
      std::make_tuple(Shape(2, 3, 4, 5), Shape(2, 3, 4, 6), -1)};
};

TEST_F(ConcatBackwardTest, Concat) {
  for (const auto& entry : SHAPE_AXIS_TUPLES) {
    VariableNode X1("X1", std::get<0>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                    1);
    VariableNode X2("X2", std::get<1>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                    1);
    ConcatNode Z("Z", {&X1, &X2}, std::get<2>(entry));
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
