// Copyright 2019 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class SubscriptRangeForwardTest : public testing::Test, public DataType {};

TEST_F(SubscriptRangeForwardTest, SubscriptRange_Xshape2_axis0_index0to2) {
  ConstantNode X("X", Shape(2), {0, 1});
  SubscriptRangeNode Z("Z", &X, 0, 0, 2);
  tsr_t expected_Z{0, 1};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptRangeForwardTest, SubscriptRange_Xshape2_axis0_index1to2) {
  ConstantNode X("X", Shape(2), {0, 1});
  SubscriptRangeNode Z("Z", &X, 0, 1, 2);
  tsr_t expected_Z{1};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptRangeForwardTest, SubscriptRange_Xshape23_axis1_index1to3) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  SubscriptRangeNode Z("Z", &X, 1, 1, 3);
  tsr_t expected_Z{{1, 2}, {4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptRangeForwardTest, SubscriptRange_Xshape23_axis1_index2to3) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  SubscriptRangeNode Z("Z", &X, 1, 2, 3);
  tsr_t expected_Z{2, 5};
  expected_Z.reshape(2, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptRangeForwardTest, SubscriptRange_Xshape234_axis2_index2to4) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  SubscriptRangeNode Z("Z", &X, 2, 2, 4);
  tsr_t expected_Z{2,  3,   //
                   6,  7,   //
                   10, 11,  //
                   14, 15,  //
                   18, 19,  //
                   22, 23};
  expected_Z.reshape(2, 3, 2);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SubscriptRangeForwardTest, SubscriptRange_Xshape234_axis2_index3to4) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  SubscriptRangeNode Z("Z", &X, 2, 3, 4);
  tsr_t expected_Z{3,  7,  11,  //
                   15, 19, 23};
  expected_Z.reshape(2, 3, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

class SubscriptRangeBackwardTest : public testing::Test {
 protected:
  const std::vector<std::tuple<Shape, int, std::pair<int, int>>>
      SHAPE_AXIS_INDEX_TUPLES = {
          std::make_tuple(Shape(2), 0, std::make_pair(0, 1)),
          std::make_tuple(Shape(2), 0, std::make_pair(0, 2)),
          std::make_tuple(Shape(2), 0, std::make_pair(1, 2)),
          std::make_tuple(Shape(2, 3), 0, std::make_pair(0, 1)),
          std::make_tuple(Shape(2, 3), 0, std::make_pair(0, 2)),
          std::make_tuple(Shape(2, 3), 0, std::make_pair(1, 2)),
          std::make_tuple(Shape(2, 3), 1, std::make_pair(0, 1)),
          std::make_tuple(Shape(2, 3), 1, std::make_pair(0, 2)),
          std::make_tuple(Shape(2, 3), 1, std::make_pair(0, 3)),
          std::make_tuple(Shape(2, 3), 1, std::make_pair(1, 2)),
          std::make_tuple(Shape(2, 3), 1, std::make_pair(1, 3)),
          std::make_tuple(Shape(2, 3), 1, std::make_pair(2, 3)),
          std::make_tuple(Shape(2, 3, 4), 0, std::make_pair(0, 1)),
          std::make_tuple(Shape(2, 3, 4), 0, std::make_pair(0, 2)),
          std::make_tuple(Shape(2, 3, 4), 0, std::make_pair(1, 2)),
          std::make_tuple(Shape(2, 3, 4), 1, std::make_pair(0, 1)),
          std::make_tuple(Shape(2, 3, 4), 1, std::make_pair(0, 2)),
          std::make_tuple(Shape(2, 3, 4), 1, std::make_pair(0, 3)),
          std::make_tuple(Shape(2, 3, 4), 1, std::make_pair(1, 2)),
          std::make_tuple(Shape(2, 3, 4), 1, std::make_pair(1, 3)),
          std::make_tuple(Shape(2, 3, 4), 1, std::make_pair(2, 3)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(0, 1)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(0, 2)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(0, 3)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(0, 4)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(1, 2)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(1, 3)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(1, 4)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(2, 3)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(2, 4)),
          std::make_tuple(Shape(2, 3, 4), 2, std::make_pair(3, 4))};
};

TEST_F(SubscriptRangeBackwardTest, SubscriptRange) {
  for (const auto& entry : SHAPE_AXIS_INDEX_TUPLES) {
    VariableNode X("X", std::get<0>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    SubscriptRangeNode Z("Z", &X, std::get<1>(entry), std::get<2>(entry).first,
                         std::get<2>(entry).second);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
