// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class LayerNormBackwardTest : public testing::Test {};

TEST_F(LayerNormBackwardTest, LayerNorm) {
  int batch = 30;
  int m = 5;
  VariableNode X("X", Shape(batch, m), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode gamma("gamma", Shape(m), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode beta("beta", Shape(m), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  LayerNormNode Z("Z", &X, &gamma, &beta);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
