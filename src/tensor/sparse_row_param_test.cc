// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>
#include <random>
#include <utility>
#include <vector>

namespace deepx_core {

class SparseRowParamTest : public testing::Test, public DataTypeD {
 protected:
  std::default_random_engine engine;
};

TEST_F(SparseRowParamTest, Construct_std_il_col1) {
  const srp_t X{{1, 2}, {{1}, {2}}};
  EXPECT_EQ(X.col(), 1);
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_FALSE(X.get_row_no_init(0));
  EXPECT_EQ(X.get_row_no_init(1)[0], 1);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
}

TEST_F(SparseRowParamTest, Construct_std_il_col2) {
  const srp_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_FALSE(X.get_row_no_init(0));
  EXPECT_EQ(X.get_row_no_init(1)[0], 1);
  EXPECT_EQ(X.get_row_no_init(1)[1], 11);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
  EXPECT_EQ(X.get_row_no_init(2)[1], 22);
}

TEST_F(SparseRowParamTest, Move) {
  srp_t X{{1, 2}, {{1, 11}, {2, 22}}};

  srp_t Y(std::move(X));
  EXPECT_EQ(Y.size(), 2u);

  srp_t Z;
  Z = std::move(Y);
  EXPECT_EQ(Z.size(), 2u);
}

TEST_F(SparseRowParamTest, clear) {
  srp_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  X.clear();
  EXPECT_EQ(X.col(), 0);
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseRowParamTest, zeros) {
  srp_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  X.zeros();
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseRowParamTest, upsert) {
  srp_t X{{1, 3}, {{1, 1}, {2, 2}}};
  srp_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.upsert(Y);

  srp_t expected_X{{1, 3, 4}, {{1, 1}, {3, 3}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, upsert_if) {
  srp_t X{{1, 3}, {{1, 1}, {2, 2}}};
  srp_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.upsert_if(
      Y, [](const srp_t::value_type& entry) { return entry.first % 2 == 0; });

  srp_t expected_X{{1, 3, 4}, {{1, 1}, {2, 2}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, merge_1) {
  srp_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srp_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge(Y);

  srp_t expected_X{{1, 2, 3, 4}, {{1, 1}, {2, 2}, {3, 3}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, merge_2) {
  srp_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srp_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge(std::move(Y));

  srp_t expected_X{{1, 2, 3, 4}, {{1, 1}, {2, 2}, {3, 3}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, merge_if_1) {
  srp_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srp_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge_if(
      Y, [](const srp_t::value_type& entry) { return entry.first % 2 == 0; });

  srp_t expected_X{{1, 2, 4}, {{1, 1}, {2, 2}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, merge_if_2) {
  srp_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srp_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge_if(std::move(Y), [](const srp_t::value_type& entry) {
    return entry.first % 2 == 0;
  });

  srp_t expected_X{{1, 2, 4}, {{1, 1}, {2, 2}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, assign) {
  srp_t X{{1, 2}, {{1, 11}, {2, 22}}};
  std::vector<float_t> row_value;
  row_value = {0, 0};
  X.assign(0, row_value.data());
  EXPECT_EQ(X.size(), 3u);
  row_value = {1, 1};
  X.assign(1, row_value.data());
  EXPECT_EQ(X.size(), 3u);
  row_value = {2, 2};
  X.assign(2, row_value.data());

  srp_t expected_X{{0, 1, 2}, {{0, 0}, {1, 1}, {2, 2}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, assign_view) {
  srp_t X;
  std::vector<float_t> row_value1(2);
  std::vector<float_t> row_value2(2);
  X.set_col(2);

  X.assign_view(1, row_value1.data());
  X.assign_view(2, row_value2.data());

  srp_t expected_X1{{1, 2}, {{0, 0}, {0, 0}}};
  EXPECT_EQ(X, expected_X1);

  row_value1 = {1, 1};
  row_value2 = {2, 2};

  srp_t expected_X2{{1, 2}, {{1, 1}, {2, 2}}};
  EXPECT_EQ(X, expected_X2);
}

TEST_F(SparseRowParamTest, remove_if) {
  srp_t X{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
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
      [](const srp_t::value_type& entry) { return entry.first % 2 == 0; });

  srp_t expected_X{{1, 3, 5, 7, 9},
                   {{1, 11}, {3, 33}, {5, 55}, {7, 77}, {9, 99}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, remove_zeros) {
  srp_t X{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
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

  srp_t expected_X{{1, 3, 5, 7, 9},
                   {{0, 11}, {3, 0}, {5, 55}, {7, 77}, {9, 99}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, get_row_TENSOR_INITIALIZER_TYPE_ZEROS) {
  srp_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);

  srp_t expected_X{{0, 1, 2}, {{0, 0}, {0, 0}, {2, 22}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, get_row_TENSOR_INITIALIZER_TYPE_ONES) {
  srp_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);

  srp_t expected_X{{0, 1, 2}, {{1, 1}, {0, 0}, {2, 22}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, get_row_TENSOR_INITIALIZER_TYPE_CONSTANT) {
  srp_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);

  srp_t expected_X{{0, 1, 2}, {{6, 6}, {0, 0}, {2, 22}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowParamTest, get_row_TENSOR_INITIALIZER_TYPE_RAND) {
  srp_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, 0.5, 1.0);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);
  EXPECT_EQ(X.size(), 3u);
  EXPECT_NE(X.get_row_no_init(0)[0], 0);
  EXPECT_NE(X.get_row_no_init(0)[1], 0);
  EXPECT_EQ(X.get_row_no_init(1)[0], 0);
  EXPECT_EQ(X.get_row_no_init(1)[1], 0);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
  EXPECT_EQ(X.get_row_no_init(2)[1], 22);
}

TEST_F(SparseRowParamTest, get_row_TENSOR_INITIALIZER_TYPE_RANDN) {
  srp_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0.5, 1.0);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);
  EXPECT_EQ(X.size(), 3u);
  EXPECT_NE(X.get_row_no_init(0)[0], 0);
  EXPECT_NE(X.get_row_no_init(0)[1], 0);
  EXPECT_EQ(X.get_row_no_init(1)[0], 0);
  EXPECT_EQ(X.get_row_no_init(1)[1], 0);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
  EXPECT_EQ(X.get_row_no_init(2)[1], 22);
}

TEST_F(SparseRowParamTest, find) {
  srp_t X{{2, 3}, {{2}, {3}}};
  const srp_t& cX = X;
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

TEST_F(SparseRowParamTest, iterator) {
  srp_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}};
  float_t sum = 0;
  for (const auto& entry : X) {
    sum += entry.second[1];
  }
  EXPECT_EQ(sum, 66);

  const srp_t& cX = X;
  sum = 0;
  for (const auto& entry : cX) {
    sum += entry.second[1];
  }
  EXPECT_EQ(sum, 66);
}

TEST_F(SparseRowParamTest, Compare) {
  srp_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}};
  srp_t Y{{1, 3, 2}, {{1, 11}, {3, 33}, {2, 22}}};
  srp_t Z{{1, 3}, {{1, 11}, {3, 33}}};
  EXPECT_TRUE(X == Y);
  EXPECT_FALSE(X != Y);
  EXPECT_FALSE(X == Z);
  EXPECT_TRUE(X != Z);
}

TEST_F(SparseRowParamTest, WriteRead) {
  srp_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}}, read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_X;
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
}

TEST_F(SparseRowParamTest, WriteReadView) {
  srp_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}}, read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_X);
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
}

}  // namespace deepx_core
