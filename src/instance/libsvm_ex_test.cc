// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/instance/libsvm_ex.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace deepx_core {

class LibsvmExInstanceReaderHelperTest : public testing::Test,
                                         public DataTypeD {
 protected:
  using reader_t = LibsvmExInstanceReaderHelper<float_t, int_t>;
};

TEST_F(LibsvmExInstanceReaderHelperTest, Parse) {
  std::vector<std::string> lines = {"\t1\t0:1\t1:1|2\t3\t", "2:2 4:2 5:2|6 7",
                                    "3 uuid:10000 8:1 9:1|10 11",
                                    "4:4 uuid:10001 12:2 13:2|14 15"};
  csr_t X0, X1;
  std::vector<csr_t*> X = {&X0, &X1};
  tsr_t Y(Shape(0, 1));
  tsr_t W;
  tsrs_t uuid;

  for (const std::string& line : lines) {
    ASSERT_TRUE(reader_t(line).Parse(&X, &Y, &W, &uuid));
  }

  csr_t expected_X0{
      {0, 2, 4, 6, 8}, {0, 1, 4, 5, 8, 9, 12, 13}, {1, 1, 2, 2, 1, 1, 2, 2}};
  csr_t expected_X1{
      {0, 2, 4, 6, 8}, {2, 3, 6, 7, 10, 11, 14, 15}, {1, 1, 1, 1, 1, 1, 1, 1}};
  tsr_t expected_Y{1, 2, 3, 4};
  expected_Y.reshape(-1, 1);
  tsr_t expected_W{1, 2, 1, 4};
  expected_W.reshape(-1, 1);
  tsrs_t expected_uuid{"", "", "10000", "10001"};
  EXPECT_EQ(X0, expected_X0);
  EXPECT_EQ(X1, expected_X1);
  EXPECT_EQ(Y, expected_Y);
  EXPECT_EQ(W, expected_W);
  EXPECT_EQ(uuid, expected_uuid);
}

TEST_F(LibsvmExInstanceReaderHelperTest, Parse_Bad) {
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
  csr_t X0, X1;
  std::vector<csr_t*> X = {&X0, &X1};
  tsr_t Y(Shape(0, 1));
  tsr_t W;
  tsrs_t uuid;

  for (const std::string& line : lines) {
    EXPECT_FALSE(reader_t(line).Parse(&X, &Y, &W, &uuid));
  }
}

}  // namespace deepx_core
