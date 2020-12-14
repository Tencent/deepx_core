// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/instance/libsvm.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace deepx_core {

class LibsvmInstanceReaderHelperTest : public testing::Test, public DataTypeD {
 protected:
  using reader_t = LibsvmInstanceReaderHelper<float_t, int_t>;
};

TEST_F(LibsvmInstanceReaderHelperTest, Parse) {
  std::vector<std::string> lines = {"\t1\t0:1\t1:1\t2\t3\t", "2:2 4:2 5:2 6 7",
                                    "3 uuid:10000 8:1 9:1 10 11",
                                    "4:4 uuid:10001 12:2 13:2 14 15"};
  csr_t X;
  tsr_t Y(Shape(0, 1));
  tsr_t W;
  tsrs_t uuid;

  for (const std::string& line : lines) {
    ASSERT_TRUE(reader_t(line).Parse(&X, &Y, &W, &uuid));
  }

  csr_t expected_X{{0, 4, 8, 12, 16},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                   {1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1}};
  tsr_t expected_Y{1, 2, 3, 4};
  expected_Y.reshape(-1, 1);
  tsr_t expected_W{1, 2, 1, 4};
  expected_W.reshape(-1, 1);
  tsrs_t expected_uuid{"", "", "10000", "10001"};
  EXPECT_EQ(X, expected_X);
  EXPECT_EQ(Y, expected_Y);
  EXPECT_EQ(W, expected_W);
  EXPECT_EQ(uuid, expected_uuid);
}

TEST_F(LibsvmInstanceReaderHelperTest, Parse_label_size2) {
  std::vector<std::string> lines = {
      "\t1\t10\t0:1\t1:1\t2\t3\t", "2:2 20:20 4:2 5:2 6 7",
      "3 30 uuid:10000 8:1 9:1 10 11", "4:4 40:40 uuid:10001 12:2 13:2 14 15"};
  csr_t X;
  tsr_t Y(Shape(0, 2));
  tsr_t W;
  tsrs_t uuid;

  for (const std::string& line : lines) {
    ASSERT_TRUE(reader_t(line).Parse(&X, &Y, &W, &uuid));
  }

  csr_t expected_X{{0, 4, 8, 12, 16},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                   {1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1}};
  tsr_t expected_Y = {{1, 10},  //
                      {2, 20},  //
                      {3, 30},  //
                      {4, 40}};
  tsr_t expected_W = {{1, 1},   //
                      {2, 20},  //
                      {1, 1},   //
                      {4, 40}};
  tsrs_t expected_uuid{"", "", "10000", "10001"};
  EXPECT_EQ(X, expected_X);
  EXPECT_EQ(Y, expected_Y);
  EXPECT_EQ(W, expected_W);
  EXPECT_EQ(uuid, expected_uuid);
}

TEST_F(LibsvmInstanceReaderHelperTest, Parse_Bad) {
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
                                    "1:1 uuid:10000 1:-101"};
  csr_t X;
  tsr_t Y(Shape(0, 1));
  tsr_t W;
  tsrs_t uuid;

  for (const std::string& line : lines) {
    EXPECT_FALSE(reader_t(line).Parse(&X, &Y, &W, &uuid));
  }
}

}  // namespace deepx_core
