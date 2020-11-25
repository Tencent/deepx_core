// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class ReshapeForwardTest : public testing::Test, public DataType {};

TEST_F(ReshapeForwardTest, Reshape) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  ReshapeNode Z("Z", &X, Shape(3, 2));
  tsr_t expected_Z{{0, 1},  //
                   {2, 3},  //
                   {4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ReshapeForwardTest, ReshapeFast) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  ReshapeFastNode Z("Z", &X, Shape(3, 2));
  tsr_t expected_Z{{0, 1},  //
                   {2, 3},  //
                   {4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ReshapeForwardTest, Reshape2) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  Reshape2Node Z("Z", &X, Shape(3, 2));
  tsr_t expected_Z{{0, 1},  //
                   {2, 3},  //
                   {4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ReshapeForwardTest, Reshape2Fast) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  Reshape2FastNode Z("Z", &X, Shape(3, 2));
  tsr_t expected_Z{{0, 1},  //
                   {2, 3},  //
                   {4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

class ExpandDimForwardTest : public testing::Test, public DataType {};

TEST_F(ExpandDimForwardTest, ExpandDim_Xshape3_axis0) {
  ConstantNode X("X", Shape(3), {0, 1, 2});
  ExpandDimNode Z("Z", &X, 0);
  tsr_t expected_Z{0, 1, 2};
  expected_Z.reshape(1, 3);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ExpandDimForwardTest, ExpandDim_Xshape2_axis1) {
  ConstantNode X("X", Shape(2), {0, 1});
  ExpandDimNode Z("Z", &X, 1);
  tsr_t expected_Z{0, 1};
  expected_Z.reshape(2, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ExpandDimForwardTest, ExpandDimFast_Xshape3_axis0) {
  ConstantNode X("X", Shape(3), {0, 1, 2});
  ExpandDimFastNode Z("Z", &X, 0);
  tsr_t expected_Z{0, 1, 2};
  expected_Z.reshape(1, 3);
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(ExpandDimForwardTest, ExpandDimFast_Xshape2_axis1) {
  ConstantNode X("X", Shape(2), {0, 1});
  ExpandDimFastNode Z("Z", &X, 1);
  tsr_t expected_Z{0, 1};
  expected_Z.reshape(2, 1);
  CheckOpForward(&Z, 0, expected_Z);
}

class SqueezeForwardTest : public testing::Test, public DataType {};

TEST_F(SqueezeForwardTest, Squeeze_Xshape13_axis0) {
  ConstantNode X("X", Shape(1, 3), {0, 1, 2});
  SqueezeNode Z("Z", &X, 0);
  tsr_t expected_Z{0, 1, 2};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SqueezeForwardTest, Squeeze_Xshape21_axis1) {
  ConstantNode X("X", Shape(2, 1), {0, 1});
  SqueezeNode Z("Z", &X, 1);
  tsr_t expected_Z{0, 1};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SqueezeForwardTest, SqueezeFast_Xshape13_axis0) {
  ConstantNode X("X", Shape(1, 3), {0, 1, 2});
  SqueezeFastNode Z("Z", &X, 0);
  tsr_t expected_Z{0, 1, 2};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SqueezeForwardTest, SqueezeFast_Xshape21_axis1) {
  ConstantNode X("X", Shape(2, 1), {0, 1});
  SqueezeFastNode Z("Z", &X, 1);
  tsr_t expected_Z{0, 1};
  CheckOpForward(&Z, 0, expected_Z);
}

class ReshapeBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, Shape>> SHAPE_PAIRS = {
      {Shape(1), Shape(1)},          {Shape(1), Shape(-1)},
      {Shape(2, 3), Shape(3, 2)},    {Shape(2, 3), Shape(2, -1)},
      {Shape(2, 3), Shape(-1, 3)},   {Shape(2, 3, 4), Shape(4, 6)},
      {Shape(2, 3, 4), Shape(3, 8)}, {Shape(2, 3, 4), Shape(-1, 12)}};
};

TEST_F(ReshapeBackwardTest, Reshape) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReshapeNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReshapeBackwardTest, ReshapeFast) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReshapeFastNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReshapeBackwardTest, Reshape2) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    Reshape2Node Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReshapeBackwardTest, Reshape2Fast) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    Reshape2FastNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

class ExpandDimBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, int>> SHAPE_AXIS_PAIRS = {
      {Shape(2), 0},        {Shape(2), 1},        {Shape(2), -1},
      {Shape(2), -2},       {Shape(2, 3), 0},     {Shape(2, 3), 1},
      {Shape(2, 3), 2},     {Shape(2, 3), -1},    {Shape(2, 3), -2},
      {Shape(2, 3), -3},    {Shape(2, 3, 4), 0},  {Shape(2, 3, 4), 1},
      {Shape(2, 3, 4), 2},  {Shape(2, 3, 4), 3},  {Shape(2, 3, 4), -1},
      {Shape(2, 3, 4), -2}, {Shape(2, 3, 4), -3}, {Shape(2, 3, 4), -4}};
};

TEST_F(ExpandDimBackwardTest, ExpandDim) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ExpandDimNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ExpandDimBackwardTest, ExpandDimFast) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ExpandDimFastNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

class SqueezeBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, int>> SHAPE_AXIS_PAIRS = {
      {Shape(1, 3), 0},    {Shape(2, 1), 1},     {Shape(2, 1), -1},
      {Shape(1, 3), -2},   {Shape(1, 3, 4), 0},  {Shape(2, 1, 4), 1},
      {Shape(2, 3, 1), 2}, {Shape(2, 3, 1), -1}, {Shape(2, 1, 4), -2},
      {Shape(1, 3, 4), -3}};
};

TEST_F(SqueezeBackwardTest, Squeeze) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    SqueezeNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(SqueezeBackwardTest, SqueezeFast) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    SqueezeFastNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
