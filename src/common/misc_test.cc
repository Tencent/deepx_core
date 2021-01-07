// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/misc.h>
#include <deepx_core/common/stream.h>
#include <gtest/gtest.h>
#include <vector>

namespace deepx_core {

class MiscTest : public testing::Test {};

TEST_F(MiscTest, GetOutputPredictFile) {
  EXPECT_EQ(GetOutputPredictFile(StdinStdoutPath(), "any"), StdinStdoutPath());
  EXPECT_EQ(GetOutputPredictFile("dir", StdinStdoutPath()), "dir/stdin");
  EXPECT_EQ(GetOutputPredictFile("dir", "file"), "dir/file");
  EXPECT_EQ(GetOutputPredictFile("dir", "hdfs://file"), "dir/file");
  EXPECT_EQ(GetOutputPredictFile("dir", "file.gz"), "dir/file");
  EXPECT_EQ(GetOutputPredictFile("dir", "hdfs://file.gz"), "dir/file");
  EXPECT_EQ(GetOutputPredictFile("dir", "a/b/file"), "dir/a_b_file");
}

TEST_F(MiscTest, ParseDeepDims) {
  std::vector<int> deep_dims;
  ASSERT_TRUE(ParseDeepDims("64,32", &deep_dims, ""));
  EXPECT_EQ(deep_dims, std::vector<int>({64, 32}));

  ASSERT_TRUE(ParseDeepDims("64,32,1", &deep_dims, ""));
  EXPECT_EQ(deep_dims, std::vector<int>({64, 32, 1}));

  ASSERT_FALSE(ParseDeepDims("64,-32", &deep_dims, ""));
  ASSERT_FALSE(ParseDeepDims("a,32", &deep_dims, ""));
  ASSERT_FALSE(ParseDeepDims("64,b", &deep_dims, ""));
}

TEST_F(MiscTest, ParseDeepDimsAppendOne) {
  std::vector<int> deep_dims;
  ASSERT_TRUE(ParseDeepDimsAppendOne("64,32", &deep_dims, ""));
  EXPECT_EQ(deep_dims, std::vector<int>({64, 32, 1}));

  ASSERT_TRUE(ParseDeepDimsAppendOne("64,32,1", &deep_dims, ""));
  EXPECT_EQ(deep_dims, std::vector<int>({64, 32, 1}));

  ASSERT_FALSE(ParseDeepDimsAppendOne("64,-32", &deep_dims, ""));
  ASSERT_FALSE(ParseDeepDimsAppendOne("a,32", &deep_dims, ""));
  ASSERT_FALSE(ParseDeepDimsAppendOne("64,b", &deep_dims, ""));
}

}  // namespace deepx_core
