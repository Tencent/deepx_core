// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/tensor/data_type.h>
#include <deepx_core/tensor/sparse_row_grad.h>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace deepx_core {

class SparseRowGradTest : public testing::Test, public DataTypeD {};

TEST_F(SparseRowGradTest, Construct_std_il_col1) {
  srg_t X{{1, 2}, {{1}, {2}}};
  EXPECT_EQ(X.col(), 1);
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_EQ(X.get_row_no_init(0)[0], 0);
  EXPECT_EQ(X.get_row_no_init(1)[0], 1);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
}

TEST_F(SparseRowGradTest, Construct_std_il_col2) {
  srg_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_EQ(X.get_row_no_init(0)[0], 0);
  EXPECT_EQ(X.get_row_no_init(0)[1], 0);
  EXPECT_EQ(X.get_row_no_init(1)[0], 1);
  EXPECT_EQ(X.get_row_no_init(1)[1], 11);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
  EXPECT_EQ(X.get_row_no_init(2)[1], 22);
}

TEST_F(SparseRowGradTest, Move) {
  srg_t X{{1, 2}, {{1, 11}, {2, 22}}};

  srg_t Y(std::move(X));
  EXPECT_EQ(Y.size(), 2u);

  srg_t Z;
  Z = std::move(Y);
  EXPECT_EQ(Z.size(), 2u);
}

TEST_F(SparseRowGradTest, clear) {
  srg_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  X.clear();
  EXPECT_EQ(X.col(), 0);
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseRowGradTest, assign) {
  srg_t X{{1, 2}, {{1, 11}, {2, 22}}};
  std::vector<float_t> row_value;
  row_value = {0, 0};
  X.assign(0, row_value.data());
  EXPECT_EQ(X.size(), 3u);
  row_value = {1, 1};
  X.assign(1, row_value.data());
  EXPECT_EQ(X.size(), 3u);
  row_value = {2, 2};
  X.assign(2, row_value.data());

  srg_t expected_X{{0, 1, 2}, {{0, 0}, {1, 1}, {2, 2}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowGradTest, remove_if) {
  srg_t X{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
          {{1, 11},
           {2, 22},
           {3, 33},
           {4, 44},
           {5, 55},
           {6, 66},
           {7, 77},
           {8, 88},
           {9, 99},
           {10, 1010}}};
  X.remove_if(
      [](const srg_t::value_type& entry) { return entry.first % 2 == 0; });

  srg_t expected_X{{1, 3, 5, 7, 9},
                   {{1, 11}, {3, 33}, {5, 55}, {7, 77}, {9, 99}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowGradTest, remove_zeros) {
  srg_t X{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
          {{0, 11},
           {0, 0},
           {3, 0},
           {0, 0},
           {5, 55},
           {0, 0},
           {7, 77},
           {0, 0},
           {9, 99},
           {0, 0}}};
  X.remove_zeros();

  srg_t expected_X{{1, 3, 5, 7, 9},
                   {{0, 11}, {3, 0}, {5, 55}, {7, 77}, {9, 99}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowGradTest, find) {
  srg_t X{{2, 3}, {{2}, {3}}};
  const srg_t& cX = X;
  EXPECT_EQ(X.find(0), X.end());
  EXPECT_EQ(X.find(0), cX.end());
  EXPECT_EQ(cX.find(1), X.end());
  EXPECT_EQ(cX.find(1), cX.end());
  ASSERT_NE(X.find(2), X.end());
  ASSERT_NE(X.find(2), X.cend());
  EXPECT_EQ(X.find(2)->first, 2u);
  EXPECT_EQ(X.find(2)->second[0], 2);
  ASSERT_NE(X.find(3), X.end());
  ASSERT_NE(X.find(3), X.cend());
  EXPECT_EQ(X.find(3)->first, 3u);
  EXPECT_EQ(X.find(3)->second[0], 3);
}

TEST_F(SparseRowGradTest, iterator) {
  srg_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}};
  float_t sum = 0;
  for (const auto& entry : X) {
    sum += entry.second[1];
  }
  EXPECT_EQ(sum, 66);

  const srg_t& cX = X;
  sum = 0;
  for (const auto& entry : cX) {
    sum += entry.second[1];
  }
  EXPECT_EQ(sum, 66);
}

TEST_F(SparseRowGradTest, Compare) {
  srg_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}};
  srg_t Y{{1, 3, 2}, {{1, 11}, {3, 33}, {2, 22}}};
  srg_t Z{{1, 3}, {{1, 11}, {3, 33}}};
  EXPECT_TRUE(X == Y);
  EXPECT_FALSE(X != Y);
  EXPECT_FALSE(X == Z);
  EXPECT_TRUE(X != Z);
}

TEST_F(SparseRowGradTest, WriteRead) {
  srg_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}}, read_X;

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
