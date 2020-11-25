// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class ArgAxisForwardTest : public testing::Test, public DataType {};

TEST_F(ArgAxisForwardTest, ArgMax_Xshape2_axis0) {
  ConstantNode X("X", Shape(2), {0, 1});
  ArgMaxNode Y("Y", &X, 0);
  tsr_t expected_Y{1};
  CheckOpForward(&Y, 0, expected_Y);
}

TEST_F(ArgAxisForwardTest, ArgMax_Xshape23_axis1) {
  ConstantNode X("X", Shape(2, 3), {1, 3, 2, 5, 0, 4});
  ArgMaxNode Y("Y", &X, 1);
  tsr_t expected_Y{1, 0};
  CheckOpForward(&Y, 0, expected_Y);
}

TEST_F(ArgAxisForwardTest, ArgMax_Xshape234_axis2) {
  ConstantNode X("X", Shape(2, 3, 4), {7,  0,  23, 6,   //
                                       5,  1,  13, 10,  //
                                       21, 20, 14, 18,  //
                                       22, 9,  16, 2,   //
                                       11, 3,  15, 19,  //
                                       17, 12, 4,  8});
  ArgMaxNode Y("Y", &X, 2);
  tsr_t expected_Y{{2, 2, 0}, {0, 3, 0}};
  CheckOpForward(&Y, 0, expected_Y);
}

TEST_F(ArgAxisForwardTest, ArgMin_Xshape2_axis0) {
  ConstantNode X("X", Shape(2), {0, 1});
  ArgMinNode Y("Y", &X, 0);
  tsr_t expected_Y{0};
  CheckOpForward(&Y, 0, expected_Y);
}

TEST_F(ArgAxisForwardTest, ArgMin_Xshape23_axis1) {
  ConstantNode X("X", Shape(2, 3), {1, 3, 2, 5, 0, 4});
  ArgMinNode Y("Y", &X, 1);
  tsr_t expected_Y{0, 1};
  CheckOpForward(&Y, 0, expected_Y);
}

TEST_F(ArgAxisForwardTest, ArgMin_Xshape234_axis2) {
  ConstantNode X("X", Shape(2, 3, 4), {7,  0,  23, 6,   //
                                       5,  1,  13, 10,  //
                                       21, 20, 14, 18,  //
                                       22, 9,  16, 2,   //
                                       11, 3,  15, 19,  //
                                       17, 12, 4,  8});
  ArgMinNode Y("Y", &X, 2);
  tsr_t expected_Y{{1, 1, 2}, {3, 1, 2}};
  CheckOpForward(&Y, 0, expected_Y);
}

class ReduceAxisBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, int>> SHAPE_AXIS_PAIRS = {
      {Shape(1), -1},          {Shape(1), 0},          {Shape(2, 3), -1},
      {Shape(2, 3), 0},        {Shape(2, 3), 1},       {Shape(2, 3, 4), -1},
      {Shape(2, 3, 4), 0},     {Shape(2, 3, 4), 1},    {Shape(2, 3, 4), 0},
      {Shape(2, 3, 4, 5), -1}, {Shape(2, 3, 4, 5), 0}, {Shape(2, 3, 4, 5), 1},
      {Shape(2, 3, 4, 5), 2},  {Shape(2, 3, 4, 5), 3}};
};

TEST_F(ReduceAxisBackwardTest, ReduceSum_keep_dim0) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceSumNode Z("Z", &X, entry.second, 0);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceSum_keep_dim1) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceSumNode Z("Z", &X, entry.second, 1);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceSum_reduce_all) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceSumNode Z("Z", &X);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceMean_keep_dim0) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceMeanNode Z("Z", &X, entry.second, 0);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceMean_keep_dim1) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceMeanNode Z("Z", &X, entry.second, 1);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceMean_reduce_all) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceMeanNode Z("Z", &X);
    CheckOpBackward(&Z, 0);
  }
}

// ReduceMax and ReduceMin are not differentiable.

TEST_F(ReduceAxisBackwardTest, ReduceL1_keep_dim0) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
    ReduceL1Node Z("Z", &X, entry.second, 0);
    CheckOpBackward(&Z, 0);

    X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceL1_keep_dim1) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
    ReduceL1Node Z("Z", &X, entry.second, 1);
    CheckOpBackward(&Z, 0);

    X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceL1_reduce_all) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RAND, 1, 2);
    ReduceL1Node Z("Z", &X);
    CheckOpBackward(&Z, 0);

    X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, -2, -1);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceL2_keep_dim0) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceL2Node Z("Z", &X, entry.second, 0);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceL2_keep_dim1) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceL2Node Z("Z", &X, entry.second, 1);
    CheckOpBackward(&Z, 0);
  }
}

TEST_F(ReduceAxisBackwardTest, ReduceL2_reduce_all) {
  for (const auto& entry : SHAPE_AXIS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ReduceL2Node Z("Z", &X);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
