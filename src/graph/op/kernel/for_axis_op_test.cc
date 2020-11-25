// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class ForAxisBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, int>> SHAPE_AXIS_PAIRS = {
      {Shape(1), -1},          {Shape(1), 0},          {Shape(2, 3), -1},
      {Shape(2, 3), 0},        {Shape(2, 3), 1},       {Shape(2, 3, 4), -1},
      {Shape(2, 3, 4), 0},     {Shape(2, 3, 4), 1},    {Shape(2, 3, 4), 2},
      {Shape(2, 3, 4, 5), -1}, {Shape(2, 3, 4, 5), 0}, {Shape(2, 3, 4, 5), 1},
      {Shape(2, 3, 4, 5), 2},  {Shape(2, 3, 4, 5), 3}};
};

TEST_F(ForAxisBackwardTest, Softmax) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode W("W", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    SoftmaxNode Y("Y", &X, entry.second);
    MulNode Z("Z", &Y, &W);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ForAxisBackwardTest, Softmax2) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode W("W", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    Softmax2Node Y("Y", &X, entry.second);
    MulNode Z("Z", &Y, &W);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ForAxisBackwardTest, LogSoftmax) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    LogSoftmaxNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ForAxisBackwardTest, Normalize2) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    Normalize2Node Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
