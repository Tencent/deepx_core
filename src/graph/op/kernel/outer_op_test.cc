// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class OuterBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, Shape>> SHAPE_PAIRS = {
      {Shape(2), Shape(3)},
      {Shape(4), Shape(2, 3)},
      {Shape(2, 3), Shape(4)},
      {Shape(2, 3, 4), Shape(5)},
      {Shape(2, 3), Shape(4, 5)}};
};

TEST_F(OuterBackwardTest, Outer) {
  for (const auto& entry : SHAPE_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    VariableNode Y("Y", entry.second, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    OuterNode Z("Z", &X, &Y);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
