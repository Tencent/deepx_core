// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/tensor/csr_matrix.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>

namespace deepx_core {

class CSRMatrixTest : public testing::Test, public DataTypeD {};

TEST_F(CSRMatrixTest, iterator) {
  int_t sum_j = 0;
  float_t sum_xij = 0;
  csr_t X{{0, 1, 4}, {6666, 7777, 8888, 9999}, {1, 2, 3, 4}};
  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      sum_j += CSR_COL(X);
      sum_xij += CSR_VALUE(X);
    }
  }
  EXPECT_EQ(sum_j, (int_t)33330);
  EXPECT_EQ(sum_xij, (float_t)10);
}

}  // namespace deepx_core
