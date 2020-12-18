// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class BatchNormBackwardTest : public testing::Test {};

TEST_F(BatchNormBackwardTest, BatchNorm) {
  int batch = 30;
  int m = 5;
  VariableNode X("X", Shape(batch, m), TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  VariableNode gamma("gamma", Shape(m), TENSOR_INITIALIZER_TYPE_ONES, 0, 0);
  VariableNode beta("beta", Shape(m), TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
  VariableNode mean("mean", Shape(m), TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
  mean.set_need_grad(0);
  VariableNode var("var", Shape(m), TENSOR_INITIALIZER_TYPE_ONES, 0, 0);
  var.set_need_grad(0);
  BatchNormNode Z("Z", &X, &gamma, &beta, &mean, &var, 0.9);
  CheckOpBackward(&Z, 0);
}

}  // namespace deepx_core
