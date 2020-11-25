// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/tensor/data_type.h>
#include <deepx_core/tensor/sparse_vector_param.h>
#include <gtest/gtest.h>
#include <random>
#include <utility>

namespace deepx_core {

class SparseVectorParamTest : public testing::Test, public DataTypeD {
 protected:
  std::default_random_engine engine;
};

TEST_F(SparseVectorParamTest, Construct_std_il) {
  const svp_t X{{1, 1}, {2, 2}};
  EXPECT_EQ(X.size(), 2u);
  EXPECT_FALSE(X.empty());
  EXPECT_EQ(X.get_scalar_no_init(0), 0);
  EXPECT_EQ(X.get_scalar_no_init(1), 1);
  EXPECT_EQ(X.get_scalar_no_init(2), 2);
}

TEST_F(SparseVectorParamTest, Move) {
  svp_t X{{1, 1}, {2, 2}};

  svp_t Y(std::move(X));
  EXPECT_EQ(Y.size(), 2u);

  svp_t Z;
  Z = std::move(Y);
  EXPECT_EQ(Z.size(), 2u);
}

TEST_F(SparseVectorParamTest, clear) {
  svp_t X{{1, 1}, {2, 2}};
  EXPECT_EQ(X.size(), 2u);
  X.clear();
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseVectorParamTest, zeros) {
  svp_t X{{1, 1}, {2, 2}};
  EXPECT_EQ(X.size(), 2u);
  X.zeros();
  EXPECT_EQ(X.size(), 0u);
}

TEST_F(SparseVectorParamTest, upsert) {
  svp_t X{{1, 1}, {3, 2}};
  svp_t Y{{3, 3}, {4, 4}};
  X.upsert(Y);

  svp_t expected_X{{1, 1}, {3, 3}, {4, 4}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, upsert_if) {
  svp_t X{{1, 1}, {3, 2}};
  svp_t Y{{3, 3}, {4, 4}};
  X.upsert_if(
      Y, [](const svp_t::value_type& entry) { return entry.first % 2 == 0; });

  svp_t expected_X{{1, 1}, {3, 2}, {4, 4}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, merge_1) {
  svp_t X{{1, 1}, {2, 2}};
  svp_t Y{{3, 3}, {4, 4}};
  X.merge(Y);

  svp_t expected_X{{1, 1}, {2, 2}, {3, 3}, {4, 4}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, merge_2) {
  svp_t X{{1, 1}, {2, 2}};
  svp_t Y{{3, 3}, {4, 4}};
  X.merge(std::move(Y));

  svp_t expected_X{{1, 1}, {2, 2}, {3, 3}, {4, 4}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, merge_if_1) {
  svp_t X{{1, 1}, {2, 2}};
  svp_t Y{{3, 3}, {4, 4}};
  X.merge_if(
      Y, [](const svp_t::value_type& entry) { return entry.first % 2 == 0; });

  svp_t expected_X{{1, 1}, {2, 2}, {4, 4}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, merge_if_2) {
  svp_t X{{1, 1}, {2, 2}};
  svp_t Y{{3, 3}, {4, 4}};
  X.merge_if(std::move(Y), [](const svp_t::value_type& entry) {
    return entry.first % 2 == 0;
  });

  svp_t expected_X{{1, 1}, {2, 2}, {4, 4}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, assign) {
  svp_t X{{1, 1}, {2, 2}};
  X.assign(0, 1);
  EXPECT_EQ(X.size(), 3u);
  X.assign(1, 2);
  EXPECT_EQ(X.size(), 3u);
  X.assign(2, 3);

  svp_t expected_X{{0, 1}, {1, 2}, {2, 3}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, remove_if) {
  svp_t X{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
          {6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}};
  X.remove_if(
      [](const svp_t::value_type& entry) { return entry.first % 2 == 0; });

  svp_t expected_X{{1, 1}, {3, 3}, {5, 5}, {7, 7}, {9, 9}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, remove_zeros) {
  svp_t X{{1, 1}, {2, 0}, {3, 3}, {4, 0}, {5, 5},
          {6, 0}, {7, 7}, {8, 0}, {9, 9}, {10, 0}};
  X.remove_zeros();

  svp_t expected_X{{1, 1}, {3, 3}, {5, 5}, {7, 7}, {9, 9}};
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, get_scalar_TENSOR_INITIALIZER_TYPE_ZEROS) {
  svp_t X{{2, 2}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);

  svp_t expected_X{{0, 0}, {1, 0}, {2, 2}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ZEROS);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, get_scalar_TENSOR_INITIALIZER_TYPE_ONES) {
  svp_t X{{2, 2}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);

  svp_t expected_X{{0, 1}, {1, 0}, {2, 2}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_ONES);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, get_scalar_TENSOR_INITIALIZER_TYPE_CONSTANT) {
  svp_t X{{2, 2}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);

  svp_t expected_X{{0, 6}, {1, 0}, {2, 2}};
  expected_X.set_initializer(TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  EXPECT_EQ(X, expected_X);
}

TEST_F(SparseVectorParamTest, get_scalar_TENSOR_INITIALIZER_TYPE_RAND) {
  svp_t X{{2, 2}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_RAND, 0.5, 1.0);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);
  EXPECT_EQ(X.size(), 3u);
  EXPECT_NE(X.get_scalar_no_init(0), 0);
  EXPECT_EQ(X.get_scalar_no_init(1), 0);
  EXPECT_EQ(X.get_scalar_no_init(2), 2);
}

TEST_F(SparseVectorParamTest, get_scalar_TENSOR_INITIALIZER_TYPE_RANDN) {
  svp_t X{{2, 2}};
  X.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0.5, 1.0);
  X.get_scalar(engine, 0);
  X.get_scalar_no_init(1);
  X.get_scalar(engine, 2);
  EXPECT_EQ(X.size(), 3u);
  EXPECT_NE(X.get_scalar_no_init(0), 0);
  EXPECT_EQ(X.get_scalar_no_init(1), 0);
  EXPECT_EQ(X.get_scalar_no_init(2), 2);
}

TEST_F(SparseVectorParamTest, find) {
  svp_t X{{{2, 2}, {3, 3}}};
  const svp_t& cX = X;
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

TEST_F(SparseVectorParamTest, iterator) {
  svp_t X{{{1, 11}, {2, 22}, {3, 33}}};
  float_t sum = 0;
  for (const auto& entry : X) {
    sum += entry.second;
  }
  EXPECT_EQ(sum, 66);

  const svp_t& cX = X;
  sum = 0;
  for (const auto& entry : cX) {
    sum += entry.second;
  }
  EXPECT_EQ(sum, 66);
}

TEST_F(SparseVectorParamTest, Compare) {
  svp_t X{{{1, 11}, {2, 22}, {3, 33}}};
  svp_t Y{{{1, 11}, {3, 33}, {2, 22}}};
  svp_t Z{{{1, 11}, {3, 33}}};
  EXPECT_TRUE(X == Y);
  EXPECT_FALSE(X != Y);
  EXPECT_FALSE(X == Z);
  EXPECT_TRUE(X != Z);
}

TEST_F(SparseVectorParamTest, WriteRead) {
  svp_t X{{{1, 11}, {2, 22}, {3, 33}}}, read_X;

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
