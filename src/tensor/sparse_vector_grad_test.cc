// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>
#include <utility>

namespace deepx_core {

class SparseVectorGradTest : public testing::Test, public DataTypeD {};

TEST_F(SparseVectorGradTest, Construct_std_il) {
  svg_t X{{1, 1}, {2, 2}};
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_EQ(X.get_scalar_no_init(0), 0);
  EXPECT_EQ(X.get_scalar_no_init(1), 1);
  EXPECT_EQ(X.get_scalar_no_init(2), 2);
}

TEST_F(SparseVectorGradTest, Move) {
  svg_t X{{1, 1}, {2, 2}};

  svg_t Y(std::move(X));
  EXPECT_EQ(Y.size(), 2u);

  svg_t Z;
  Z = std::move(Y);
  EXPECT_EQ(Z.size(), 2u);
}

TEST_F(SparseVectorGradTest, clear) {
  svg_t X{{1, 1}, {2, 2}};
  EXPECT_EQ(X.size(), 2u);
  X.clear();
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseVectorGradTest, zeros) {
  svg_t X{{1, 1}, {2, 2}};
  EXPECT_EQ(X.size(), 2u);
  X.zeros();
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseVectorGradTest, assign) {
  svg_t X{{1, 1}, {2, 2}};
  X.assign(0, 1);
  EXPECT_EQ(X.size(), 3u);
  X.assign(1, 2);
  EXPECT_EQ(X.size(), 3u);
  X.assign(2, 3);

  svg_t expected_X{{0, 1}, {1, 2}, {2, 3}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorGradTest, remove_if) {
  svg_t X{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
          {6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}};
  X.remove_if(
      [](const svg_t::value_type& entry) { return entry.first % 2 == 0; });

  svg_t expected_X{{1, 1}, {3, 3}, {5, 5}, {7, 7}, {9, 9}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorGradTest, remove_zeros) {
  svg_t X{{1, 1}, {2, 0}, {3, 3}, {4, 0}, {5, 5},
          {6, 0}, {7, 7}, {8, 0}, {9, 9}, {10, 0}};
  X.remove_zeros();

  svg_t expected_X{{1, 1}, {3, 3}, {5, 5}, {7, 7}, {9, 9}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorGradTest, find) {
  svg_t X{{{2, 2}, {3, 3}}};
  const svg_t& cX = X;
  EXPECT_EQ(X.find(0), X.end());
  EXPECT_EQ(X.find(0), cX.end());
  EXPECT_EQ(cX.find(1), X.end());
  EXPECT_EQ(cX.find(1), cX.end());
  ASSERT_NE(X.find(2), X.end());
  ASSERT_NE(X.find(2), X.cend());
  EXPECT_EQ(X.find(2)->first, 2u);
  EXPECT_EQ(X.find(2)->second, 2);
  ASSERT_NE(X.find(3), X.end());
  ASSERT_NE(X.find(3), X.cend());
  EXPECT_EQ(X.find(3)->first, 3u);
  EXPECT_EQ(X.find(3)->second, 3);
}

TEST_F(SparseVectorGradTest, iterator) {
  svg_t X{{{1, 11}, {2, 22}, {3, 33}}};
  float_t sum = 0;
  for (const auto& entry : X) {
    sum += entry.second;
  }
  EXPECT_EQ(sum, 66);

  const svg_t& cX = X;
  sum = 0;
  for (const auto& entry : cX) {
    sum += entry.second;
  }
  EXPECT_EQ(sum, 66);
}

TEST_F(SparseVectorGradTest, Compare) {
  svg_t X{{{1, 11}, {2, 22}, {3, 33}}};
  svg_t Y{{{1, 11}, {3, 33}, {2, 22}}};
  svg_t Z{{{1, 11}, {3, 33}}};
  EXPECT_TRUE(X == Y);
  EXPECT_FALSE(X != Y);
  EXPECT_FALSE(X == Z);
  EXPECT_TRUE(X != Z);
}

TEST_F(SparseVectorGradTest, WriteRead) {
  svg_t X{{{1, 11}, {2, 22}, {3, 33}}}, read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_X;
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
}

}  // namespace deepx_core
