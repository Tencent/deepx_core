// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/compress.h>
#include <gtest/gtest.h>
#include <string>

namespace deepx_core {

class CompressTest : public testing::Test {};

TEST_F(CompressTest, Compress_Decompress) {
  std::string in = "a short string";
  std::string out;
  ASSERT_TRUE(Compress(in, &out));

  std::string expected_in;
  ASSERT_TRUE(Decompress(out, &expected_in));
  EXPECT_EQ(in, expected_in);
}

TEST_F(CompressTest, HighCompress_Decompress) {
  std::string in = "a long string ";
  for (int i = 0; i < 10; ++i) {
    in.append(in);
  }
  std::string out;
  ASSERT_TRUE(HighCompress(in, &out));

  std::string expected_in;
  ASSERT_TRUE(Decompress(out, &expected_in));
  EXPECT_EQ(in, expected_in);
}

}  // namespace deepx_core
