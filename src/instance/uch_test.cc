// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/instance/uch.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace deepx_core {

class UCHInstanceReaderHelperTest : public testing::Test, public DataTypeD {
 protected:
  using reader_t = UCHInstanceReaderHelper<float_t, int_t>;
};

TEST_F(UCHInstanceReaderHelperTest, Parse) {
  std::vector<std::string> lines = {"1 0|1|2|3|4",      "2 10|11",
                                    "3 20|21|22",       "4 30|31|32|33",
                                    "5 40|41|42|43|44", "6 50|51|52|53|54|55"};
  csr_t X_user;
  csr_t X_cand;
  csr_t X_hist0, X_hist1, X_hist2;
  std::vector<csr_t*> X_hist = {&X_hist0, &X_hist1, &X_hist2};
  tsr_t X_hist_size;
  tsr_t Y(Shape(0, 1));
  tsr_t* W = nullptr;
  tsrs_t* uuid = nullptr;

  for (const std::string& line : lines) {
    ASSERT_TRUE(reader_t(line).Parse(&X_user, &X_cand, &X_hist, &X_hist_size,
                                     &Y, W, uuid));
  }

  csr_t expected_X_user{
      {0, 1, 2, 3, 4, 5, 6}, {0, 10, 20, 30, 40, 50}, {1, 1, 1, 1, 1, 1}};
  csr_t expected_X_cand{
      {0, 1, 2, 3, 4, 5, 6}, {1, 11, 21, 31, 41, 51}, {1, 1, 1, 1, 1, 1}};
  csr_t expected_X_hist0{
      {0, 1, 1, 2, 3, 4, 5}, {2, 22, 32, 42, 52}, {1, 1, 1, 1, 1}};
  csr_t expected_X_hist1{{0, 1, 1, 1, 2, 3, 4}, {3, 33, 43, 53}, {1, 1, 1, 1}};
  csr_t expected_X_hist2{{0, 1, 1, 1, 1, 2, 3}, {4, 44, 54}, {1, 1, 1}};
  tsr_t expected_X_hist_size{3, 0, 1, 2, 3, 3};
  tsr_t expected_Y{1, 2, 3, 4, 5, 6};
  expected_Y.reshape(-1, 1);
  EXPECT_EQ(X_user, expected_X_user);
  EXPECT_EQ(X_cand, expected_X_cand);
  EXPECT_EQ(X_hist0, expected_X_hist0);
  EXPECT_EQ(X_hist1, expected_X_hist1);
  EXPECT_EQ(X_hist2, expected_X_hist2);
  EXPECT_EQ(X_hist_size, expected_X_hist_size);
  EXPECT_EQ(Y, expected_Y);
}

TEST_F(UCHInstanceReaderHelperTest, Parse_Bad) {
  std::vector<std::string> lines = {"",
                                    " ",
                                    "a",
                                    "10001",
                                    "-10001",
                                    "1~",
                                    "1:a",
                                    "1:-1",
                                    "1:10001",
                                    "1:1 uuid:",
                                    "1:1 uuid=10000",
                                    "1:1 uuid2=10000",
                                    "1:1 uuid:10000 a:1",
                                    "1:1 uuid:10000 1~1",
                                    "1:1 uuid:10000 1:a",
                                    "1:1 uuid:10000 1:101",
                                    "1:1 uuid:10000 1:-101",
                                    "1:1 uuid:10000 1:1|a:1",
                                    "1:1 uuid:10000 1:1|1~1",
                                    "1:1 uuid:10000 1:1|1:a",
                                    "1:1 uuid:10000 1:1|1:101",
                                    "1:1 uuid:10000 1:1|1:-101"};
  csr_t X_user;
  csr_t X_cand;
  csr_t X_hist0, X_hist1, X_hist2;
  std::vector<csr_t*> X_hist = {&X_hist0, &X_hist1, &X_hist2};
  tsr_t X_hist_size;
  tsr_t Y(Shape(0, 1));
  tsr_t* W = nullptr;
  tsrs_t* uuid = nullptr;

  for (const std::string& line : lines) {
    EXPECT_FALSE(reader_t(line).Parse(&X_user, &X_cand, &X_hist, &X_hist_size,
                                      &Y, W, uuid));
  }
}

}  // namespace deepx_core
