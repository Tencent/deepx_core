// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class MatmulBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, Shape>> SHAPE_PAIRS = {
      {Shape(2, 3), Shape(3, 4)},       {Shape(3), Shape(3, 4)},
      {Shape(2, 3), Shape(3)},          {Shape(3), Shape(3)},
      {Shape(2, 2, 3), Shape(2, 3, 4)}, {Shape(2, 3), Shape(2, 3, 4)},
      {Shape(3), Shape(2, 3, 4)},       {Shape(2, 2, 3), Shape(3, 4)},
      {Shape(2, 2, 3), Shape(3)},       {Shape(2, 3, 2, 3), Shape(2, 3, 3, 4)},
      {Shape(2, 3), Shape(2, 3, 3, 4)}, {Shape(3), Shape(2, 3, 3, 4)},
      {Shape(2, 3, 2, 3), Shape(3, 4)}, {Shape(2, 3, 2, 3), Shape(3)}};
};

TEST_F(MatmulBackwardTest, Matmul) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    MatmulNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);
  }
}

class Matmul2BackwardTest : public testing::Test {
 protected:
  const std::vector<std::tuple<Shape, Shape, int, int>> SHAPE_TRANS_TUPLES = {
      {std::make_tuple(Shape(2, 3), Shape(3, 4), 0, 0)},
      {std::make_tuple(Shape(2, 3), Shape(4, 3), 0, 1)},
      {std::make_tuple(Shape(3, 2), Shape(3, 4), 1, 0)},
      {std::make_tuple(Shape(3, 2), Shape(4, 3), 1, 1)},
      {std::make_tuple(Shape(3), Shape(3, 4), 0, 0)},
      {std::make_tuple(Shape(3), Shape(4, 3), 0, 1)},
      {std::make_tuple(Shape(2, 3), Shape(3), 0, 0)},
      {std::make_tuple(Shape(3, 2), Shape(3), 1, 0)},
      {std::make_tuple(Shape(3), Shape(3), 0, 0)},
      {std::make_tuple(Shape(2, 2, 3), Shape(2, 3, 4), 0, 0)},
      {std::make_tuple(Shape(2, 2, 3), Shape(2, 4, 3), 0, 1)},
      {std::make_tuple(Shape(2, 3, 2), Shape(2, 3, 4), 1, 0)},
      {std::make_tuple(Shape(2, 3, 2), Shape(2, 4, 3), 1, 1)},
      {std::make_tuple(Shape(2, 3), Shape(2, 3, 4), 0, 0)},
      {std::make_tuple(Shape(2, 3), Shape(2, 4, 3), 0, 1)},
      {std::make_tuple(Shape(3, 2), Shape(2, 3, 4), 1, 0)},
      {std::make_tuple(Shape(3, 2), Shape(2, 4, 3), 1, 1)},
      {std::make_tuple(Shape(3), Shape(2, 3, 4), 0, 0)},
      {std::make_tuple(Shape(3), Shape(2, 4, 3), 0, 1)},
      {std::make_tuple(Shape(2, 2, 3), Shape(3, 4), 0, 0)},
      {std::make_tuple(Shape(2, 2, 3), Shape(4, 3), 0, 1)},
      {std::make_tuple(Shape(2, 3, 2), Shape(3, 4), 1, 0)},
      {std::make_tuple(Shape(2, 3, 2), Shape(4, 3), 1, 1)},
      {std::make_tuple(Shape(2, 2, 3), Shape(3), 0, 0)},
      {std::make_tuple(Shape(2, 3, 2), Shape(3), 1, 0)},
      {std::make_tuple(Shape(2, 3, 2, 3), Shape(2, 3, 3, 4), 0, 0)},
      {std::make_tuple(Shape(2, 3, 2, 3), Shape(2, 3, 4, 3), 0, 1)},
      {std::make_tuple(Shape(2, 3, 3, 2), Shape(2, 3, 3, 4), 1, 0)},
      {std::make_tuple(Shape(2, 3, 3, 2), Shape(2, 3, 4, 3), 1, 1)},
      {std::make_tuple(Shape(2, 3), Shape(2, 3, 3, 4), 0, 0)},
      {std::make_tuple(Shape(2, 3), Shape(2, 3, 4, 3), 0, 1)},
      {std::make_tuple(Shape(3, 2), Shape(2, 3, 3, 4), 1, 0)},
      {std::make_tuple(Shape(3, 2), Shape(2, 3, 4, 3), 1, 1)},
      {std::make_tuple(Shape(3), Shape(2, 3, 3, 4), 0, 0)},
      {std::make_tuple(Shape(3), Shape(2, 3, 4, 3), 0, 1)},
      {std::make_tuple(Shape(2, 3, 2, 3), Shape(3, 4), 0, 0)},
      {std::make_tuple(Shape(2, 3, 2, 3), Shape(4, 3), 0, 1)},
      {std::make_tuple(Shape(2, 3, 3, 2), Shape(3, 4), 1, 0)},
      {std::make_tuple(Shape(2, 3, 3, 2), Shape(4, 3), 1, 1)},
      {std::make_tuple(Shape(2, 3, 2, 3), Shape(3), 0, 0)},
      {std::make_tuple(Shape(2, 3, 3, 2), Shape(3), 1, 0)}};
};

TEST_F(Matmul2BackwardTest, Matmul2) {
  for (const auto& entry : SHAPE_TRANS_TUPLES) {
    VariableNode X("X", std::get<0>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    VariableNode Y("Y", std::get<1>(entry), TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    Matmul2Node Z("Z", &X, &Y, std::get<2>(entry), std::get<3>(entry));
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
