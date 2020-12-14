// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"
#include "broadcast.h"

namespace deepx_core {

/************************************************************************/
/* Broadcast */
/************************************************************************/
class BroadcastForwardTest : public testing::Test, public DataType {
 protected:
  static void TestBroadcastAdd(const Shape& Xshape, const Shape& Yshape,
                               const tsr_t& expected_Z) {
    ConstantNode X("X", Xshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    ConstantNode Y("Y", Yshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    BroadcastAddNode Z("Z", &X, &Y);
    CheckOpForward(&Z, 0, expected_Z);
  }
};

TEST_F(BroadcastForwardTest, BroadcastAdd_Xshape23_Yshape1) {
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5}};
  TestBroadcastAdd(Shape(2, 3), Shape(1), expected_Z);
}

TEST_F(BroadcastForwardTest, BroadcastAdd_Xshape23_Yshape13) {
  tsr_t expected_Z{{0, 2, 4},  //
                   {3, 5, 7}};
  TestBroadcastAdd(Shape(2, 3), Shape(1, 3), expected_Z);
}

TEST_F(BroadcastForwardTest, BroadcastAdd_Xshape213_Yshape1) {
  tsr_t expected_Z{0, 1, 2,  //
                   3, 4, 5};
  expected_Z.reshape(2, 1, 3);
  TestBroadcastAdd(Shape(2, 1, 3), Shape(1), expected_Z);
}

TEST_F(BroadcastForwardTest, BroadcastAdd_Xshape234_Yshape131) {
  tsr_t expected_Z{0,  1,  2,  3,   //
                   5,  6,  7,  8,   //
                   10, 11, 12, 13,  //
                   12, 13, 14, 15,  //
                   17, 18, 19, 20,  //
                   22, 23, 24, 25};
  expected_Z.reshape(2, 3, 4);
  TestBroadcastAdd(Shape(2, 3, 4), Shape(1, 3, 1), expected_Z);
}

TEST_F(BroadcastForwardTest, BroadcastAdd_Xshape234_Yshape234) {
  tsr_t expected_Z{0,  2,  4,  6,   //
                   8,  10, 12, 14,  //
                   16, 18, 20, 22,  //
                   24, 26, 28, 30,  //
                   32, 34, 36, 38,  //
                   40, 42, 44, 46};
  expected_Z.reshape(2, 3, 4);
  TestBroadcastAdd(Shape(2, 3, 4), Shape(2, 3, 4), expected_Z);
}

TEST_F(BroadcastForwardTest, BroadcastAdd_Xshape2345_Yshape45) {
  tsr_t expected_Z{0,   2,   4,   6,   8,    //
                   10,  12,  14,  16,  18,   //
                   20,  22,  24,  26,  28,   //
                   30,  32,  34,  36,  38,   //
                   20,  22,  24,  26,  28,   //
                   30,  32,  34,  36,  38,   //
                   40,  42,  44,  46,  48,   //
                   50,  52,  54,  56,  58,   //
                   40,  42,  44,  46,  48,   //
                   50,  52,  54,  56,  58,   //
                   60,  62,  64,  66,  68,   //
                   70,  72,  74,  76,  78,   //
                   60,  62,  64,  66,  68,   //
                   70,  72,  74,  76,  78,   //
                   80,  82,  84,  86,  88,   //
                   90,  92,  94,  96,  98,   //
                   80,  82,  84,  86,  88,   //
                   90,  92,  94,  96,  98,   //
                   100, 102, 104, 106, 108,  //
                   110, 112, 114, 116, 118,  //
                   100, 102, 104, 106, 108,  //
                   110, 112, 114, 116, 118,  //
                   120, 122, 124, 126, 128,  //
                   130, 132, 134, 136, 138};
  expected_Z.reshape(2, 3, 4, 5);
  TestBroadcastAdd(Shape(2, 3, 4, 5), Shape(4, 5), expected_Z);
}

TEST_F(BroadcastForwardTest, BroadcastAdd_Xshape2141_Yshape1315) {
  tsr_t expected_Z{0,  1,  2,  3,  4,   //
                   1,  2,  3,  4,  5,   //
                   2,  3,  4,  5,  6,   //
                   3,  4,  5,  6,  7,   //
                   5,  6,  7,  8,  9,   //
                   6,  7,  8,  9,  10,  //
                   7,  8,  9,  10, 11,  //
                   8,  9,  10, 11, 12,  //
                   10, 11, 12, 13, 14,  //
                   11, 12, 13, 14, 15,  //
                   12, 13, 14, 15, 16,  //
                   13, 14, 15, 16, 17,  //
                   4,  5,  6,  7,  8,   //
                   5,  6,  7,  8,  9,   //
                   6,  7,  8,  9,  10,  //
                   7,  8,  9,  10, 11,  //
                   9,  10, 11, 12, 13,  //
                   10, 11, 12, 13, 14,  //
                   11, 12, 13, 14, 15,  //
                   12, 13, 14, 15, 16,  //
                   14, 15, 16, 17, 18,  //
                   15, 16, 17, 18, 19,  //
                   16, 17, 18, 19, 20,  //
                   17, 18, 19, 20, 21};
  expected_Z.reshape(2, 3, 4, 5);
  TestBroadcastAdd(Shape(2, 1, 4, 1), Shape(1, 3, 1, 5), expected_Z);
}

class BroadcastBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, Shape>> SHAPE_PAIRS = {
      {Shape(1), Shape(1)},
      {Shape(2), Shape(1)},
      {Shape(1), Shape(2)},
      {Shape(2, 3), Shape(1)},
      {Shape(2, 3), Shape(2, 1)},
      {Shape(2, 3), Shape(1, 3)},
      {Shape(2, 3), Shape(2, 3)},
      {Shape(2, 3, 4), Shape(1)},
      {Shape(2, 3, 4), Shape(4)},
      {Shape(2, 3, 4), Shape(3, 4)},
      {Shape(2, 3, 4), Shape(1, 3, 1)},
      {Shape(2, 3, 4), Shape(2, 1, 4)},
      {Shape(2, 3, 4), Shape(2, 3, 4)},
      {Shape(2, 3, 4, 5), Shape(1)},
      {Shape(2, 3, 4, 5), Shape(5)},
      {Shape(2, 3, 4, 5), Shape(4, 5)},
      {Shape(2, 3, 4, 5), Shape(3, 4, 5)},
      {Shape(2, 3, 4, 5), Shape(1, 3, 1, 5)},
      {Shape(2, 3, 4, 5), Shape(2, 1, 4, 1)},
      {Shape(2, 3, 4, 5), Shape(2, 3, 4, 5)},
      {Shape(2, 1, 4, 1), Shape(1, 3, 1, 5)},
      {Shape(1, 3, 1, 5), Shape(2, 1, 4, 1)}};
};

TEST_F(BroadcastBackwardTest, BroadcastAdd) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    BroadcastAddNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(BroadcastBackwardTest, BroadcastSub) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    BroadcastSubNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(BroadcastBackwardTest, BroadcastMul) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    BroadcastMulNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(BroadcastBackwardTest, BroadcastDiv) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
    BroadcastDivNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);

    Y.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(BroadcastBackwardTest, BroadcastPow) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
    BroadcastPowNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);
  }
}

/************************************************************************/
/* BroadcastTo */
/************************************************************************/
class BroadcastToForwardTest : public testing::Test, public DataType {
 protected:
  static void Test(const Shape& Xshape, const Shape& Yshape,
                   const tsr_t& expected_Z) {
    ConstantNode X("X", Xshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    BroadcastToNode Z("Z", &X, Yshape);
    CheckOpForward(&Z, 0, expected_Z);
  }
};

TEST_F(BroadcastToForwardTest, BroadcastTo_Xshape1_Yshape23) {
  tsr_t expected_Z{{0, 0, 0},  //
                   {0, 0, 0}};
  Test(Shape(1), Shape(2, 3), expected_Z);
}

TEST_F(BroadcastToForwardTest, BroadcastTo_Xshape13_Yshape23) {
  tsr_t expected_Z{{0, 1, 2},  //
                   {0, 1, 2}};
  Test(Shape(1, 3), Shape(2, 3), expected_Z);
}

TEST_F(BroadcastToForwardTest, BroadcastTo_Xshape1_Yshape213) {
  tsr_t expected_Z{0, 0, 0,  //
                   0, 0, 0};
  expected_Z.reshape(2, 1, 3);
  Test(Shape(1), Shape(2, 1, 3), expected_Z);
}

TEST_F(BroadcastToForwardTest, BroadcastTo_Xshape131_Yshape234) {
  tsr_t expected_Z{0, 0, 0, 0,  //
                   1, 1, 1, 1,  //
                   2, 2, 2, 2,  //
                   0, 0, 0, 0,  //
                   1, 1, 1, 1,  //
                   2, 2, 2, 2};
  expected_Z.reshape(2, 3, 4);
  Test(Shape(1, 3, 1), Shape(2, 3, 4), expected_Z);
}

TEST_F(BroadcastToForwardTest, BroadcastTo_Xshape234_Yshape234) {
  tsr_t expected_Z{0,  1,  2,  3,   //
                   4,  5,  6,  7,   //
                   8,  9,  10, 11,  //
                   12, 13, 14, 15,  //
                   16, 17, 18, 19,  //
                   20, 21, 22, 23};
  expected_Z.reshape(2, 3, 4);
  Test(Shape(2, 3, 4), Shape(2, 3, 4), expected_Z);
}

TEST_F(BroadcastToForwardTest, BroadcastTo_Xshape45_Yshape2345) {
  tsr_t expected_Z{0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19};
  expected_Z.reshape(2, 3, 4, 5);
  Test(Shape(4, 5), Shape(2, 3, 4, 5), expected_Z);
}

class BroadcastToBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, Shape>> SHAPE_PAIRS = {
      {Shape(1), Shape(1)},
      {Shape(1), Shape(2)},
      {Shape(1), Shape(2, 3)},
      {Shape(2, 1), Shape(2, 3)},
      {Shape(1, 3), Shape(2, 3)},
      {Shape(2, 3), Shape(2, 3)},
      {Shape(1), Shape(2, 3, 4)},
      {Shape(4), Shape(2, 3, 4)},
      {Shape(3, 4), Shape(2, 3, 4)},
      {Shape(1, 3, 1), Shape(2, 3, 4)},
      {Shape(2, 1, 4), Shape(2, 3, 4)},
      {Shape(2, 3, 4), Shape(2, 3, 4)},
      {Shape(1), Shape(2, 3, 4, 5)},
      {Shape(5), Shape(2, 3, 4, 5)},
      {Shape(4, 5), Shape(2, 3, 4, 5)},
      {Shape(3, 4, 5), Shape(2, 3, 4, 5)},
      {Shape(1, 3, 1, 5), Shape(2, 3, 4, 5)},
      {Shape(2, 1, 4, 1), Shape(2, 3, 4, 5)},
      {Shape(2, 3, 4, 5), Shape(2, 3, 4, 5)}};
};

TEST_F(BroadcastToBackwardTest, BroadcastTo) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    BroadcastToNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

/************************************************************************/
/* BroadcastToLike */
/************************************************************************/
class BroadcastToLikeForwardTest : public testing::Test, public DataType {
 protected:
  static void Test(const Shape& Xshape, const Shape& Yshape,
                   const tsr_t& expected_Z) {
    ConstantNode X("X", Xshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    ConstantNode Y("Y", Yshape, TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
    BroadcastToLikeNode Z("Z", &X, &Y);
    CheckOpForward(&Z, 0, expected_Z);
  }
};

TEST_F(BroadcastToLikeForwardTest, BroadcastToLike_Xshape1_Yshape23) {
  tsr_t expected_Z{{0, 0, 0},  //
                   {0, 0, 0}};
  Test(Shape(1), Shape(2, 3), expected_Z);
}

TEST_F(BroadcastToLikeForwardTest, BroadcastToLike_Xshape13_Yshape23) {
  tsr_t expected_Z{{0, 1, 2},  //
                   {0, 1, 2}};
  Test(Shape(1, 3), Shape(2, 3), expected_Z);
}

TEST_F(BroadcastToLikeForwardTest, BroadcastToLike_Xshape1_Yshape213) {
  tsr_t expected_Z{0, 0, 0,  //
                   0, 0, 0};
  expected_Z.reshape(2, 1, 3);
  Test(Shape(1), Shape(2, 1, 3), expected_Z);
}

TEST_F(BroadcastToLikeForwardTest, BroadcastToLike_Xshape131_Yshape234) {
  tsr_t expected_Z{0, 0, 0, 0,  //
                   1, 1, 1, 1,  //
                   2, 2, 2, 2,  //
                   0, 0, 0, 0,  //
                   1, 1, 1, 1,  //
                   2, 2, 2, 2};
  expected_Z.reshape(2, 3, 4);
  Test(Shape(1, 3, 1), Shape(2, 3, 4), expected_Z);
}

TEST_F(BroadcastToLikeForwardTest, BroadcastToLike_Xshape234_Yshape234) {
  tsr_t expected_Z{0,  1,  2,  3,   //
                   4,  5,  6,  7,   //
                   8,  9,  10, 11,  //
                   12, 13, 14, 15,  //
                   16, 17, 18, 19,  //
                   20, 21, 22, 23};
  expected_Z.reshape(2, 3, 4);
  Test(Shape(2, 3, 4), Shape(2, 3, 4), expected_Z);
}

TEST_F(BroadcastToLikeForwardTest, BroadcastToLike_Xshape45_Yshape2345) {
  tsr_t expected_Z{0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19,  //
                   0,  1,  2,  3,  4,   //
                   5,  6,  7,  8,  9,   //
                   10, 11, 12, 13, 14,  //
                   15, 16, 17, 18, 19};
  expected_Z.reshape(2, 3, 4, 5);
  Test(Shape(4, 5), Shape(2, 3, 4, 5), expected_Z);
}

class BroadcastToLikeBackwardTest : public BroadcastToBackwardTest {};

TEST_F(BroadcastToLikeBackwardTest, BroadcastToLike) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    BroadcastToLikeNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
