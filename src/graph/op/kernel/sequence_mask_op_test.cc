// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class SequenceMaskForwardTest : public testing::Test, public DataType {};

TEST_F(SequenceMaskForwardTest, SequenceMask_max_size3) {
  ConstantNode X("X", Shape(5), {0, 1, 2, 3, 4});
  SequenceMaskNode Z("Z", &X, 3);
  tsr_t expected_Z{{0, 0, 0},  //
                   {1, 0, 0},  //
                   {1, 1, 0},  //
                   {1, 1, 1},  //
                   {1, 1, 1}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SequenceMaskForwardTest, SequenceMask_max_size4) {
  ConstantNode X("X", Shape(5), {0, 1, 2, 3, 4});
  SequenceMaskNode Z("Z", &X, 4);
  tsr_t expected_Z{{0, 0, 0, 0},  //
                   {1, 0, 0, 0},  //
                   {1, 1, 0, 0},  //
                   {1, 1, 1, 0},  //
                   {1, 1, 1, 1}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SequenceMaskForwardTest, SequenceMask_max_size5) {
  ConstantNode X("X", Shape(5), {0, 1, 2, 3, 4});
  SequenceMaskNode Z("Z", &X, 5);
  tsr_t expected_Z{{0, 0, 0, 0, 0},  //
                   {1, 0, 0, 0, 0},  //
                   {1, 1, 0, 0, 0},  //
                   {1, 1, 1, 0, 0},  //
                   {1, 1, 1, 1, 0}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(SequenceMaskForwardTest, SequenceMask_max_size6) {
  ConstantNode X("X", Shape(5), {0, 1, 2, 3, 4});
  SequenceMaskNode Z("Z", &X, 6);
  tsr_t expected_Z{{0, 0, 0, 0, 0, 0},  //
                   {1, 0, 0, 0, 0, 0},  //
                   {1, 1, 0, 0, 0, 0},  //
                   {1, 1, 1, 0, 0, 0},  //
                   {1, 1, 1, 1, 0, 0}};
  CheckOpForward(&Z, 0, expected_Z);
}

}  // namespace deepx_core
