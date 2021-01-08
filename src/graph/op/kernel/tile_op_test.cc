// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../op_test.h"

namespace deepx_core {

class TileForwardTest : public testing::Test, public DataType {};

TEST_F(TileForwardTest, Tile_Xshape2_reps2) {
  ConstantNode X("X", Shape(2), {0, 1});
  TileNode Z("Z", &X, 2);
  tsr_t expected_Z{0, 1, 0, 1};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(TileForwardTest, Tile_Xshape23_reps22) {
  ConstantNode X("X", Shape(2, 3), {0, 1, 2, 3, 4, 5});
  TileNode Z("Z", &X, {2, 2});
  tsr_t expected_Z{{0, 1, 2, 0, 1, 2},  //
                   {3, 4, 5, 3, 4, 5},  //
                   {0, 1, 2, 0, 1, 2},  //
                   {3, 4, 5, 3, 4, 5}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(TileForwardTest, Tile_Xshape234_reps222) {
  ConstantNode X("X", Shape(2, 3, 4), {0,  1,  2,  3,   //
                                       4,  5,  6,  7,   //
                                       8,  9,  10, 11,  //
                                       12, 13, 14, 15,  //
                                       16, 17, 18, 19,  //
                                       20, 21, 22, 23});
  TileNode Z("Z", &X, {2, 2, 2});
  tsr_t expected_Z{0,  1,  2,  3,  0,  1,  2,  3,   //
                   4,  5,  6,  7,  4,  5,  6,  7,   //
                   8,  9,  10, 11, 8,  9,  10, 11,  //
                   0,  1,  2,  3,  0,  1,  2,  3,   //
                   4,  5,  6,  7,  4,  5,  6,  7,   //
                   8,  9,  10, 11, 8,  9,  10, 11,  //
                   12, 13, 14, 15, 12, 13, 14, 15,  //
                   16, 17, 18, 19, 16, 17, 18, 19,  //
                   20, 21, 22, 23, 20, 21, 22, 23,  //
                   12, 13, 14, 15, 12, 13, 14, 15,  //
                   16, 17, 18, 19, 16, 17, 18, 19,  //
                   20, 21, 22, 23, 20, 21, 22, 23,  //
                   0,  1,  2,  3,  0,  1,  2,  3,   //
                   4,  5,  6,  7,  4,  5,  6,  7,   //
                   8,  9,  10, 11, 8,  9,  10, 11,  //
                   0,  1,  2,  3,  0,  1,  2,  3,   //
                   4,  5,  6,  7,  4,  5,  6,  7,   //
                   8,  9,  10, 11, 8,  9,  10, 11,  //
                   12, 13, 14, 15, 12, 13, 14, 15,  //
                   16, 17, 18, 19, 16, 17, 18, 19,  //
                   20, 21, 22, 23, 20, 21, 22, 23,  //
                   12, 13, 14, 15, 12, 13, 14, 15,  //
                   16, 17, 18, 19, 16, 17, 18, 19,  //
                   20, 21, 22, 23, 20, 21, 22, 23};
  expected_Z.reshape(4, 6, 8);
  CheckOpForward(&Z, 0, expected_Z);
}

class TileBackwardTest : public testing::Test {
 protected:
  const std::vector<std::pair<Shape, std::vector<int>>> SHAPE_REPS_PAIRS = {
      {Shape(2), {1}},
      {Shape(2), {2}},
      {Shape(2, 3), {1, 1}},
      {Shape(2, 3), {1, 2}},
      {Shape(2, 3), {2, 1}},
      {Shape(2, 3), {2, 2}},
      {Shape(2, 3), {2, 3}},
      {Shape(2, 3, 4), {1, 1, 1}},
      {Shape(2, 3, 4), {1, 1, 2}},
      {Shape(2, 3, 4), {1, 2, 1}},
      {Shape(2, 3, 4), {2, 1, 1}},
      {Shape(2, 3, 4), {2, 2, 2}},
      {Shape(2, 3, 4), {2, 3, 4}}};
};

TEST_F(TileBackwardTest, Tile) {
  for (const auto& entry : SHAPE_REPS_PAIRS) {
    VariableNode X("X", entry.first, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    TileNode Z("Z", &X, entry.second);
    CheckOpBackward(&Z, 0);
  }
}

}  // namespace deepx_core
