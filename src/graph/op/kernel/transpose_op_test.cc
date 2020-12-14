// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class TransposeForwardTest : public testing::Test, public DataType {
 protected:
  static void Test(const Shape& Xshape, const Shape& axes,
                   const tsr_t& expected_Z) {
    ConstantNode X("X", Xshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    TransposeNode Z("Z", &X, axes);
    CheckOpForward(&Z, 0, expected_Z);
  }
};

TEST_F(TransposeForwardTest, Transpose_Xshape23_axes01) {
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5}};
  Test(Shape(2, 3), Shape(0, 1), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape23_axes10) {
  tsr_t expected_Z{{0, 3},  //
                   {1, 4},  //
                   {2, 5}};
  Test(Shape(2, 3), Shape(1, 0), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape234_axes012) {
  tsr_t expected_Z{0,  1,  2,  3,   //
                   4,  5,  6,  7,   //
                   8,  9,  10, 11,  //
                   12, 13, 14, 15,  //
                   16, 17, 18, 19,  //
                   20, 21, 22, 23};
  expected_Z.reshape(2, 3, 4);
  Test(Shape(2, 3, 4), Shape(0, 1, 2), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape234_axes021) {
  tsr_t expected_Z{0,  4,   //
                   8,  1,   //
                   5,  9,   //
                   2,  6,   //
                   10, 3,   //
                   7,  11,  //
                   12, 16,  //
                   20, 13,  //
                   17, 21,  //
                   14, 18,  //
                   22, 15,  //
                   19, 23};
  expected_Z.reshape(2, 4, 3);
  Test(Shape(2, 3, 4), Shape(0, 2, 1), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape234_axes102) {
  tsr_t expected_Z{0,  1,  2,  3,   //
                   12, 13, 14, 15,  //
                   4,  5,  6,  7,   //
                   16, 17, 18, 19,  //
                   8,  9,  10, 11,  //
                   20, 21, 22, 23};
  expected_Z.reshape(3, 2, 4);
  Test(Shape(2, 3, 4), Shape(1, 0, 2), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape234_axes120) {
  tsr_t expected_Z{0,  12, 1,   //
                   13, 2,  14,  //
                   3,  15, 4,   //
                   16, 5,  17,  //
                   6,  18, 7,   //
                   19, 8,  20,  //
                   9,  21, 10,  //
                   22, 11, 23};
  expected_Z.reshape(3, 4, 2);
  Test(Shape(2, 3, 4), Shape(1, 2, 0), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape234_axes201) {
  tsr_t expected_Z{0,  4,  8,   //
                   12, 16, 20,  //
                   1,  5,  9,   //
                   13, 17, 21,  //
                   2,  6,  10,  //
                   14, 18, 22,  //
                   3,  7,  11,  //
                   15, 19, 23};
  expected_Z.reshape(4, 2, 3);
  Test(Shape(2, 3, 4), Shape(2, 0, 1), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape234_axes210) {
  tsr_t expected_Z{0,  12,  //
                   4,  16,  //
                   8,  20,  //
                   1,  13,  //
                   5,  17,  //
                   9,  21,  //
                   2,  14,  //
                   6,  18,  //
                   10, 22,  //
                   3,  15,  //
                   7,  19,  //
                   11, 23};
  expected_Z.reshape(4, 3, 2);
  Test(Shape(2, 3, 4), Shape(2, 1, 0), expected_Z);
}

TEST_F(TransposeForwardTest, Transpose_Xshape2345_axes2310) {
  tsr_t expected_Z{0,  60, 20, 80, 40, 100,  //
                   1,  61, 21, 81, 41, 101,  //
                   2,  62, 22, 82, 42, 102,  //
                   3,  63, 23, 83, 43, 103,  //
                   4,  64, 24, 84, 44, 104,  //
                   5,  65, 25, 85, 45, 105,  //
                   6,  66, 26, 86, 46, 106,  //
                   7,  67, 27, 87, 47, 107,  //
                   8,  68, 28, 88, 48, 108,  //
                   9,  69, 29, 89, 49, 109,  //
                   10, 70, 30, 90, 50, 110,  //
                   11, 71, 31, 91, 51, 111,  //
                   12, 72, 32, 92, 52, 112,  //
                   13, 73, 33, 93, 53, 113,  //
                   14, 74, 34, 94, 54, 114,  //
                   15, 75, 35, 95, 55, 115,  //
                   16, 76, 36, 96, 56, 116,  //
                   17, 77, 37, 97, 57, 117,  //
                   18, 78, 38, 98, 58, 118,  //
                   19, 79, 39, 99, 59, 119};
  expected_Z.reshape(4, 5, 3, 2);
  Test(Shape(2, 3, 4, 5), Shape(2, 3, 1, 0), expected_Z);
}

class TransposeBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, Shape>> SHAPE_AXES_PAIRS = {
      {Shape(2), Shape(0)},
      {Shape(2, 3), Shape(0, 1)},
      {Shape(2, 3), Shape(1, 0)},
      {Shape(2, 3, 4), Shape(0, 1, 2)},
      {Shape(2, 3, 4), Shape(0, 2, 1)},
      {Shape(2, 3, 4), Shape(1, 0, 2)},
      {Shape(2, 3, 4), Shape(1, 2, 0)},
      {Shape(2, 3, 4), Shape(2, 0, 1)},
      {Shape(2, 3, 4), Shape(2, 1, 0)},
      {Shape(2, 3, 4, 5), Shape(2, 3, 0, 1)}};
};

TEST_F(TransposeBackwardTest, Transpose) {
  for (const auto& entry : SHAPE_AXES_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    TransposeNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
