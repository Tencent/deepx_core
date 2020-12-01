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

class SparseRowMatrixTest : public testing::Test, public DataTypeD {
 protected:
  std::default_random_engine engine;
};

TEST_F(SparseRowMatrixTest, Construct_std_il_col1) {
  const srm_t X{{1, 2}, {{1}, {2}}};
  EXPECT_EQ(X.col(), 1);
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_FALSE(X.get_row_no_init(0));
  EXPECT_EQ(X.get_row_no_init(1)[0], 1);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
}

TEST_F(SparseRowMatrixTest, Construct_std_il_col2) {
  const srm_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_FALSE(X.get_row_no_init(0));
  EXPECT_EQ(X.get_row_no_init(1)[0], 1);
  EXPECT_EQ(X.get_row_no_init(1)[1], 11);
  EXPECT_EQ(X.get_row_no_init(2)[0], 2);
  EXPECT_EQ(X.get_row_no_init(2)[1], 22);
}

TEST_F(SparseRowMatrixTest, Move) {
  srm_t X{{1, 2}, {{1, 11}, {2, 22}}};

  srm_t Y(std::move(X));
  EXPECT_EQ(Y.size(), 2u);

  srm_t Z;
  Z = std::move(Y);
  EXPECT_EQ(Z.size(), 2u);
}

TEST_F(SparseRowMatrixTest, clear) {
  srm_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  X.clear();
  EXPECT_EQ(X.col(), 0);
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseRowMatrixTest, zeros) {
  srm_t X{{1, 2}, {{1, 11}, {2, 22}}};
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 2u);
  X.zeros();
  EXPECT_EQ(X.col(), 2);
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseRowMatrixTest, upsert) {
  srm_t X{{1, 3}, {{1, 1}, {2, 2}}};
  srm_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.upsert(Y);

  srm_t expected_X{{1, 3, 4}, {{1, 1}, {3, 3}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, upsert_if) {
  srm_t X{{1, 3}, {{1, 1}, {2, 2}}};
  srm_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.upsert_if(
      Y, [](const srm_t::value_type& entry) { return entry.first % 2 == 0; });

  srm_t expected_X{{1, 3, 4}, {{1, 1}, {2, 2}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, merge_1) {
  srm_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srm_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge(Y);

  srm_t expected_X{{1, 2, 3, 4}, {{1, 1}, {2, 2}, {3, 3}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, merge_2) {
  srm_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srm_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge(std::move(Y));

  srm_t expected_X{{1, 2, 3, 4}, {{1, 1}, {2, 2}, {3, 3}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, merge_if_1) {
  srm_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srm_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge_if(
      Y, [](const srm_t::value_type& entry) { return entry.first % 2 == 0; });

  srm_t expected_X{{1, 2, 4}, {{1, 1}, {2, 2}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, merge_if_2) {
  srm_t X{{1, 2}, {{1, 1}, {2, 2}}};
  srm_t Y{{3, 4}, {{3, 3}, {4, 4}}};
  X.merge_if(std::move(Y), [](const srm_t::value_type& entry) {
    return entry.first % 2 == 0;
  });

  srm_t expected_X{{1, 2, 4}, {{1, 1}, {2, 2}, {4, 4}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, assign) {
  srm_t X{{1, 2}, {{1, 11}, {2, 22}}};
  std::vector<float_t> row_value;
  row_value = {0, 0};
  X.assign(0, row_value.data());
  EXPECT_EQ(X.size(), 3u);
  row_value = {1, 1};
  X.assign(1, row_value.data());
  EXPECT_EQ(X.size(), 3u);
  row_value = {2, 2};
  X.assign(2, row_value.data());

  srm_t expected_X{{0, 1, 2}, {{0, 0}, {1, 1}, {2, 2}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, assign_view) {
  srm_t X;
  std::vector<float_t> row_value1(2);
  std::vector<float_t> row_value2(2);
  X.set_col(2);

  X.assign_view(1, row_value1.data());
  X.assign_view(2, row_value2.data());

  srm_t expected_X1{{1, 2}, {{0, 0}, {0, 0}}};
  EXPECT_EQ(X, expected_X1);

  row_value1 = {1, 1};
  row_value2 = {2, 2};

  srm_t expected_X2{{1, 2}, {{1, 1}, {2, 2}}};
  EXPECT_EQ(X, expected_X2);
}

TEST_F(SparseRowMatrixTest, remove_if) {
  srm_t X{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
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
      [](const srm_t::value_type& entry) { return entry.first % 2 == 0; });

  srm_t expected_X{{1, 3, 5, 7, 9},
                   {{1, 11}, {3, 33}, {5, 55}, {7, 77}, {9, 99}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, remove_zeros) {
  srm_t X{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
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

  srm_t expected_X{{1, 3, 5, 7, 9},
                   {{0, 11}, {3, 0}, {5, 55}, {7, 77}, {9, 99}}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, get_row_TENSOR_INITIALIZER_TYPE_ZEROS) {
  srm_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);

  srm_t expected_X{{0, 1, 2}, {{0, 0}, {0, 0}, {2, 22}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, get_row_TENSOR_INITIALIZER_TYPE_ONES) {
  srm_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);

  srm_t expected_X{{0, 1, 2}, {{1, 1}, {0, 0}, {2, 22}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, get_row_TENSOR_INITIALIZER_TYPE_CONSTANT) {
  srm_t X{{2}, {{2, 22}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  X.get_row(engine, 0);
  X.get_row_no_init(1);
  X.get_row(engine, 2);

  srm_t expected_X{{0, 1, 2}, {{6, 6}, {0, 0}, {2, 22}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, get_row_TENSOR_INITIALIZER_TYPE_RAND) {
  srm_t X{{2}, {{2, 22}}};
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

TEST_F(SparseRowMatrixTest, get_row_TENSOR_INITIALIZER_TYPE_RANDN) {
  srm_t X{{2}, {{2, 22}}};
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

TEST_F(SparseRowMatrixTest, get_scalar_TENSOR_INITIALIZER_TYPE_ZEROS) {
  srm_t X{{2}, {{2}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);

  srm_t expected_X{{0, 1, 2}, {{0}, {0}, {2}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, get_scalar_TENSOR_INITIALIZER_TYPE_ONES) {
  srm_t X{{2}, {{2}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);

  srm_t expected_X{{0, 1, 2}, {{1}, {0}, {2}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, get_scalar_TENSOR_INITIALIZER_TYPE_CONSTANT) {
  srm_t X{{2}, {{2}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);

  srm_t expected_X{{0, 1, 2}, {{6}, {0}, {2}}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseRowMatrixTest, get_scalar_TENSOR_INITIALIZER_TYPE_RAND) {
  srm_t X{{2}, {{2}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, 0.5, 1.0);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);
  EXPECT_EQ(X.size(), 3u);
  EXPECT_NE(X.get_scalar_no_init(0), 0);
  EXPECT_EQ(X.get_scalar_no_init(1), 0);
  EXPECT_EQ(X.get_scalar_no_init(2), 2);
}

TEST_F(SparseRowMatrixTest, get_scalar_TENSOR_INITIALIZER_TYPE_RANDN) {
  srm_t X{{2}, {{2}}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0.5, 1.0);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);
  EXPECT_EQ(X.size(), 3u);
  EXPECT_NE(X.get_scalar_no_init(0), 0);
  EXPECT_EQ(X.get_scalar_no_init(1), 0);
  EXPECT_EQ(X.get_scalar_no_init(2), 2);
}

TEST_F(SparseRowMatrixTest, find) {
  srm_t X{{2, 3}, {{2}, {3}}};
  const srm_t& cX = X;
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

TEST_F(SparseRowMatrixTest, iterator) {
  srm_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}};
  float_t sum = 0;
  for (const auto& entry : X) {
    sum += entry.second[1];
  }
  EXPECT_EQ(sum, 66);

  const srm_t& cX = X;
  sum = 0;
  for (const auto& entry : cX) {
    sum += entry.second[1];
  }
  EXPECT_EQ(sum, 66);
}

TEST_F(SparseRowMatrixTest, Compare) {
  srm_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}};
  srm_t Y{{1, 3, 2}, {{1, 11}, {3, 33}, {2, 22}}};
  srm_t Z{{1, 3}, {{1, 11}, {3, 33}}};
  EXPECT_TRUE(X == Y);
  EXPECT_FALSE(X != Y);
  EXPECT_FALSE(X == Z);
  EXPECT_TRUE(X != Z);
}

TEST_F(SparseRowMatrixTest, WriteRead) {
  srm_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}}, read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_X;
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
}

TEST_F(SparseRowMatrixTest, WriteReadView) {
  srm_t X{{1, 2, 3}, {{1, 11}, {2, 22}, {3, 33}}}, read_X;

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
