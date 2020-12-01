// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_gtest.h>
#include <deepx_core/tensor/data_type.h>
#include <deepx_core/tensor/ll_tensor.h>

namespace deepx_core {

class LLSparseTensorTest : public testing::Test, public DataTypeD {};

TEST_F(LLSparseTensorTest, add_col1) {
  srm_t X{{0, 2}, {{1}, {2}}};
  tsr_t Y{1, 2, 3};
  Y.reshape(3, 1);

  ll_sparse_tensor_t::add(X, 1, &Y);
  tsr_t expected_Y1{2, 2, 5};
  expected_Y1.reshape(3, 1);
  EXPECT_TSR_NEAR(Y, expected_Y1);

  ll_sparse_tensor_t::add(X, 0, &Y);
  tsr_t expected_Y2{1, 0, 2};
  expected_Y2.reshape(3, 1);
  EXPECT_TSR_NEAR(Y, expected_Y2);
}

TEST_F(LLSparseTensorTest, add_col2) {
  srm_t X{{0, 1}, {{0, 1}, {0, 2}}};
  tsr_t Y{{1, 2}, {4, 5}};

  ll_sparse_tensor_t::add(X, 1, &Y);
  tsr_t expected_Y1{{1, 3}, {4, 7}};
  EXPECT_TSR_NEAR(Y, expected_Y1);

  ll_sparse_tensor_t::add(X, 0, &Y);
  tsr_t expected_Y2{{0, 1}, {0, 2}};
  EXPECT_TSR_NEAR(Y, expected_Y2);
}

TEST_F(LLSparseTensorTest, gesmm_mod_col1) {
  csr_t X{{0, 2, 5}, {6, 16, 777, 888, 999}, {1, 1, 1, 1, 1}};
  tsr_t Y{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  Y.reshape(10, 1);
  tsr_t Z(Shape(X.row(), Y.dim(1)));

  ll_sparse_tensor_t::gesmm_mod(X, Y, 0, &Z);
  // 6 + 6 = 12
  // 7 + 8 + 9 = 24
  tsr_t expected_Z1{12, 24};
  expected_Z1.reshape(2, 1);
  EXPECT_TSR_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gesmm_mod(X, Y, 1, &Z);
  tsr_t expected_Z2{24, 48};
  expected_Z2.reshape(2, 1);
  EXPECT_TSR_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, gesmm_mod_col2) {
  csr_t X{{0, 2, 5}, {6, 16, 777, 888, 999}, {1, 1, 1, 1, 1}};
  tsr_t Y{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
          {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}};
  tsr_t Z(Shape(X.row(), Y.dim(1)));

  ll_sparse_tensor_t::gesmm_mod(X, Y, 0, &Z);
  tsr_t expected_Z1{{12, 12}, {24, 24}};
  EXPECT_TSR_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gesmm_mod(X, Y, 1, &Z);
  tsr_t expected_Z2{{24, 24}, {48, 48}};
  EXPECT_TSR_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, gesmsm_col1) {
  csr_t X{{0, 1, 4}, {6666, 7777, 8, 9999}, {1, 1, 1, 1}};
  srm_t Y{{6666, 7777, 8888, 9999}, {{6}, {7}, {8}, {9}}};
  tsr_t Z(Shape(X.row(), 1));

  ll_sparse_tensor_t::gesmsm(X, Y, 0, &Z);
  // 6 = 6
  // 7 + 9 = 16
  tsr_t expected_Z1{6, 16};
  expected_Z1.reshape(2, 1);
  EXPECT_TSR_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gesmsm(X, Y, 1, &Z);
  tsr_t expected_Z2{12, 32};
  expected_Z2.reshape(2, 1);
  EXPECT_TSR_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, gesmsm_col2) {
  csr_t X{{0, 1, 4}, {6666, 7777, 8, 9999}, {1, 1, 1, 1}};
  srm_t Y{{6666, 7777, 8888, 9999}, {{6, 6}, {7, 7}, {8, 8}, {9, 9}}};
  tsr_t Z(Shape(X.row(), 2));

  ll_sparse_tensor_t::gesmsm(X, Y, 0, &Z);
  tsr_t expected_Z1{{6, 6}, {16, 16}};
  EXPECT_TSR_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gesmsm(X, Y, 1, &Z);
  tsr_t expected_Z2{{12, 12}, {32, 32}};
  EXPECT_TSR_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, gestmm_mod_col1) {
  csr_t X{{0, 1, 4, 6, 7},
          {1, 2, 3, 4, 15, 16, 17},
          {1.0, 0.2, 0.4, 0.4, 1.0, 1.0, 0.5}};
  tsr_t Y(Shape(X.row(), 1));
  Y.ones();
  srm_t Z;
  Z.set_col(Y.dim(1));

  ll_sparse_tensor_t::gestmm_mod(10, X, Y, 0, &Z);
  srm_t expected_Z1{{1, 2, 3, 4, 5, 6, 7},
                    {{1.0}, {0.2}, {0.4}, {0.4}, {1.0}, {1.0}, {0.5}}};
  EXPECT_SRM_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gestmm_mod(10, X, Y, 1, &Z);
  srm_t expected_Z2{{1, 2, 3, 4, 5, 6, 7},
                    {{2.0}, {0.4}, {0.8}, {0.8}, {2.0}, {2.0}, {1.0}}};
  EXPECT_SRM_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, gestmm_mod_col2) {
  csr_t X{{0, 1, 4, 6, 7},
          {1, 2, 3, 4, 15, 16, 17},
          {1.0, 0.2, 0.4, 0.4, 1.0, 1.0, 0.5}};
  tsr_t Y(Shape(X.row(), 2));
  Y.ones();
  srm_t Z;
  Z.set_col(Y.dim(1));

  ll_sparse_tensor_t::gestmm_mod(10, X, Y, 0, &Z);
  srm_t expected_Z1{{1, 2, 3, 4, 5, 6, 7},
                    {{1.0, 1.0},
                     {0.2, 0.2},
                     {0.4, 0.4},
                     {0.4, 0.4},
                     {1.0, 1.0},
                     {1.0, 1.0},
                     {0.5, 0.5}}};
  EXPECT_SRM_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gestmm_mod(10, X, Y, 1, &Z);
  srm_t expected_Z2{{1, 2, 3, 4, 5, 6, 7},
                    {{2.0, 2.0},
                     {0.4, 0.4},
                     {0.8, 0.8},
                     {0.8, 0.8},
                     {2.0, 2.0},
                     {2.0, 2.0},
                     {1.0, 1.0}}};
  EXPECT_SRM_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, gestmm_col1) {
  csr_t X{{0, 1, 4, 6, 7},
          {1, 2, 3, 4, 15, 16, 17},
          {1.0, 0.2, 0.4, 0.4, 1.0, 1.0, 0.5}};
  tsr_t Y(Shape(X.row(), 1));
  Y.ones();
  srm_t Z;
  Z.set_col(Y.dim(1));

  ll_sparse_tensor_t::gestmm(X, Y, 0, &Z);
  srm_t expected_Z1{{1, 2, 3, 4, 15, 16, 17},
                    {{1.0}, {0.2}, {0.4}, {0.4}, {1.0}, {1.0}, {0.5}}};
  EXPECT_SRM_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gestmm(X, Y, 1, &Z);
  srm_t expected_Z2{{1, 2, 3, 4, 15, 16, 17},
                    {{2.0}, {0.4}, {0.8}, {0.8}, {2.0}, {2.0}, {1.0}}};
  EXPECT_SRM_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, gestmm_col2) {
  csr_t X{{0, 1, 4, 6, 7},
          {1, 2, 3, 4, 15, 16, 17},
          {1.0, 0.2, 0.4, 0.4, 1.0, 1.0, 0.5}};
  tsr_t Y(Shape(X.row(), 2));
  Y.ones();
  srm_t Z;
  Z.set_col(Y.dim(1));

  ll_sparse_tensor_t::gestmm(X, Y, 0, &Z);
  srm_t expected_Z1{{1, 2, 3, 4, 15, 16, 17},
                    {{1.0, 1.0},
                     {0.2, 0.2},
                     {0.4, 0.4},
                     {0.4, 0.4},
                     {1.0, 1.0},
                     {1.0, 1.0},
                     {0.5, 0.5}}};
  EXPECT_SRM_NEAR(Z, expected_Z1);

  ll_sparse_tensor_t::gestmm(X, Y, 1, &Z);
  srm_t expected_Z2{{1, 2, 3, 4, 15, 16, 17},
                    {{2.0, 2.0},
                     {0.4, 0.4},
                     {0.8, 0.8},
                     {0.8, 0.8},
                     {2.0, 2.0},
                     {2.0, 2.0},
                     {1.0, 1.0}}};
  EXPECT_SRM_NEAR(Z, expected_Z2);
}

TEST_F(LLSparseTensorTest, add_to_tsr) {
  tsr_t X{{0, 1, 2}, {3, 4, 5}};
  tsr_t Z{{0, 1, 2}, {3, 4, 5}};
  ll_sparse_tensor_t::add_to(X, &Z);
  tsr_t expected_Z{{0, 2, 4},  //
                   {6, 8, 10}};
  EXPECT_TSR_NEAR(Z, expected_Z);
}

TEST_F(LLSparseTensorTest, add_to) {
  srm_t X{{0, 1}, {{1, 1}, {2, 2}}};
  srm_t Z{{0, 2}, {{3, 3}, {4, 4}}};
  ll_sparse_tensor_t::add_to(X, &Z);
  srm_t expected_Z{{0, 1, 2}, {{4, 4}, {2, 2}, {4, 4}}};
  EXPECT_SRM_NEAR(Z, expected_Z);
}

TEST_F(LLSparseTensorTest, scale_tsr) {
  tsr_t Z{{0, 1, 2}, {3, 4, 5}};
  ll_sparse_tensor_t::scale(2, &Z);
  tsr_t expected_Z{{0, 2, 4},  //
                   {6, 8, 10}};
  EXPECT_TSR_NEAR(Z, expected_Z);
}

TEST_F(LLSparseTensorTest, scale) {
  srm_t Z{{0, 1}, {{1, 1}, {2, 2}}};
  ll_sparse_tensor_t::scale(2, &Z);
  srm_t expected_Z{{0, 1}, {{2, 2}, {4, 4}}};
  EXPECT_SRM_NEAR(Z, expected_Z);
}

}  // namespace deepx_core
