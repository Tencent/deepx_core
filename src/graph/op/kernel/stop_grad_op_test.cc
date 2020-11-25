// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class StopGradForwardTest : public testing::Test, public DataType {};

TEST_F(StopGradForwardTest, StopGrad) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  StopGradNode Z("Z", &X);
  tsr_t expected_Z{{0, 1, 2},  //
                   {3, 4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

}  // namespace deepx_core
