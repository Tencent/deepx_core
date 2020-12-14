// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class TensorDotForwardTest : public testing::Test, public DataType {
 protected:
  static void Test(const Shape& Xshape, const Shape& Yshape, int axes_n,
                   const tsr_t& expected_Z) {
    ConstantNode X("X", Xshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    ConstantNode Y("Y", Yshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    TensorDotNode Z("Z", &X, &Y, axes_n);
    CheckOpForward(&Z, 0, expected_Z);
  }

  static void Test(const Shape& Xshape, const Shape& Yshape, const Shape& Xaxes,
                   const Shape& Yaxes, const tsr_t& expected_Z) {
    ConstantNode X("X", Xshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    ConstantNode Y("Y", Yshape, TENSOR_INITIALIZER_TYPE_ARANGE, 0, 0);
    TensorDotNode Z("Z", &X, &Y, Xaxes, Yaxes);
    CheckOpForward(&Z, 0, expected_Z);
  }
};

TEST_F(TensorDotForwardTest, TensorDot_Xshape2_Yshape2_axes_n0) {
  tsr_t expected_Z{{0, 0}, {0, 1}};
  Test(Shape(2), Shape(2), 0, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape2_Yshape2_axes_n1) {
  tsr_t expected_Z{1};
  Test(Shape(2), Shape(2), 1, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape3_Yshape3_axes_n0) {
  tsr_t expected_Z{{0, 0, 0},  //
                   {0, 1, 2},  //
                   {0, 2, 4}};
  Test(Shape(3), Shape(3), 0, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape3_Yshape3_axes_n1) {
  tsr_t expected_Z{5};
  Test(Shape(3), Shape(3), 1, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape22_Yshape22_axes_n0) {
  tsr_t expected_Z{0, 0,  //
                   0, 0,  //
                   0, 1,  //
                   2, 3,  //
                   0, 2,  //
                   4, 6,  //
                   0, 3,  //
                   6, 9};
  expected_Z.reshape(2, 2, 2, 2);
  Test(Shape(2, 2), Shape(2, 2), 0, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape22_Yshape22_axes_n1) {
  tsr_t expected_Z{{2, 3}, {6, 11}};
  Test(Shape(2, 2), Shape(2, 2), 1, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape22_Yshape22_axes_n2) {
  tsr_t expected_Z{14};
  Test(Shape(2, 2), Shape(2, 2), 2, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape23_Yshape32_Xaxes01_Yaxes10) {
  tsr_t expected_Z{50};
  Test(Shape(2, 3), Shape(3, 2), Shape(0, 1), Shape(1, 0), expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape23_Yshape32_Xaxes10_Yaxes01) {
  tsr_t expected_Z{50};
  Test(Shape(2, 3), Shape(3, 2), Shape(1, 0), Shape(0, 1), expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape123_Yshape234_axes_n2) {
  tsr_t expected_Z{220, 235, 250, 265};
  expected_Z.reshape(1, 4);
  Test(Shape(1, 2, 3), Shape(2, 3, 4), 2, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape123_Yshape234_Xaxes2_Yaxes1) {
  tsr_t expected_Z{20,  23,  26,  29,  //
                   56,  59,  62,  65,  //
                   56,  68,  80,  92,  //
                   200, 212, 224, 236};
  expected_Z.reshape(1, 2, 2, 4);
  Test(Shape(1, 2, 3), Shape(2, 3, 4), Shape(2), Shape(1), expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape123_Yshape234_Xaxes12_Yaxes01) {
  tsr_t expected_Z{220, 235, 250, 265};
  expected_Z.reshape(1, 4);
  Test(Shape(1, 2, 3), Shape(2, 3, 4), Shape(1, 2), Shape(0, 1), expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape1234_Yshape2345_axes_n3) {
  tsr_t expected_Z{21620, 21896, 22172, 22448, 22724};
  expected_Z.reshape(1, 5);
  Test(Shape(1, 2, 3, 4), Shape(2, 3, 4, 5), 3, expected_Z);
}

TEST_F(TensorDotForwardTest, TensorDot_Xshape1234_Yshape2345_Xaxes12_Yaxes01) {
  tsr_t expected_Z{4400, 4460, 4520, 4580, 4640,  //
                   4700, 4760, 4820, 4880, 4940,  //
                   5000, 5060, 5120, 5180, 5240,  //
                   5300, 5360, 5420, 5480, 5540,  //
                   4700, 4766, 4832, 4898, 4964,  //
                   5030, 5096, 5162, 5228, 5294,  //
                   5360, 5426, 5492, 5558, 5624,  //
                   5690, 5756, 5822, 5888, 5954,  //
                   5000, 5072, 5144, 5216, 5288,  //
                   5360, 5432, 5504, 5576, 5648,  //
                   5720, 5792, 5864, 5936, 6008,  //
                   6080, 6152, 6224, 6296, 6368,  //
                   5300, 5378, 5456, 5534, 5612,  //
                   5690, 5768, 5846, 5924, 6002,  //
                   6080, 6158, 6236, 6314, 6392,  //
                   6470, 6548, 6626, 6704, 6782};
  expected_Z.reshape(1, 4, 4, 5);
  Test(Shape(1, 2, 3, 4), Shape(2, 3, 4, 5), Shape(1, 2), Shape(0, 1),
       expected_Z);
}

TEST_F(TensorDotForwardTest,
       TensorDot_Xshape1234_Yshape2345_Xaxes123_Yaxes012) {
  tsr_t expected_Z{21620, 21896, 22172, 22448, 22724};
  expected_Z.reshape(1, 5);
  Test(Shape(1, 2, 3, 4), Shape(2, 3, 4, 5), Shape(1, 2, 3), Shape(0, 1, 2),
       expected_Z);
}

class TensorDotBackwardTest : public testing::Test {
 protected:
  const std::vector<std::tuple<Shape, Shape, int>> SHAPE_AXES_N_TUPLES = {
      std::make_tuple(Shape(2), Shape(2), 0),
      std::make_tuple(Shape(2), Shape(3), 0),
      std::make_tuple(Shape(2), Shape(2), 1),
      std::make_tuple(Shape(2, 2), Shape(2, 2), 0),
      std::make_tuple(Shape(2, 2), Shape(2, 2), 1),
      std::make_tuple(Shape(2, 2), Shape(2, 2), 2),
      std::make_tuple(Shape(2, 3), Shape(3, 2), 0),
      std::make_tuple(Shape(2, 3), Shape(3, 2), 1),
      std::make_tuple(Shape(1, 2, 3), Shape(2, 3, 4), 2),
      std::make_tuple(Shape(1, 2, 3, 4), Shape(2, 3, 4, 5), 3)};
  const std::vector<std::tuple<Shape, Shape, Shape, Shape>> SHAPE_AXES_TUPLES =
      {std::make_tuple(Shape(2, 3), Shape(3, 2), Shape(0), Shape(1)),
       std::make_tuple(Shape(2, 3), Shape(3, 2), Shape(1), Shape(0)),
       std::make_tuple(Shape(2, 3, 4), Shape(4, 3, 2), Shape(0), Shape(2)),
       std::make_tuple(Shape(2, 3, 4), Shape(4, 3, 2), Shape(0, 1),
                       Shape(2, 1)),
       std::make_tuple(Shape(2, 3, 4), Shape(4, 3, 2), Shape(0, 2),
                       Shape(2, 0)),
       std::make_tuple(Shape(2, 3, 4), Shape(4, 3, 2), Shape(1, 0, 2),
                       Shape(1, 2, 0)),
       std::make_tuple(Shape(2, 3, 4, 5), Shape(1, 2, 3, 4), Shape(2, 1),
                       Shape(-1, -2)),
       std::make_tuple(Shape(2, 3, 4, 5), Shape(1, 2, 3, 4), Shape(0, 1, 2),
                       Shape(-3, -2, -1))};
};

TEST_F(TensorDotBackwardTest, TensorDot_axes_n) {
  for (const auto& entry : SHAPE_AXES_N_TUPLES) {
    VariableNode X("X", std::get<0>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    VariableNode Y("Y", std::get<1>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    TensorDotNode Z("Z", &X, &Y, std::get<2>(entry));
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(TensorDotBackwardTest, TensorDot_Xaxes_Yaxes) {
  for (const auto& entry : SHAPE_AXES_TUPLES) {
    VariableNode X("X", std::get<0>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    VariableNode Y("Y", std::get<1>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    TensorDotNode Z("Z", &X, &Y, std::get<2>(entry), std::get<3>(entry));
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
