// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/str_util.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace deepx_core {

class StrUtilTest : public testing::Test {
 protected:
  using vi_t = std::vector<int>;
  using vs_t = std::vector<std::string>;
};

TEST_F(StrUtilTest, Trim) {
  std::string s;
  s = " 123 ";
  EXPECT_EQ(Trim(&s), "123");

  s = " 123";
  EXPECT_EQ(Trim(&s), "123");

  s = "123 ";
  EXPECT_EQ(Trim(&s), "123");

  s = "\r\n  \t123 \r\n\t\r\n";
  EXPECT_EQ(Trim(&s), "123");

  s = "\r\n  \t \r\n\t\r\n";
  EXPECT_TRUE(Trim(&s).empty());
}

TEST_F(StrUtilTest, BeginWith) {
  EXPECT_TRUE(BeginWith("123", "1"));
  EXPECT_TRUE(BeginWith("123", "12"));
  EXPECT_TRUE(BeginWith("123", "123"));
  EXPECT_FALSE(BeginWith("123", "312"));
  EXPECT_FALSE(BeginWith("123", "2"));
  EXPECT_FALSE(BeginWith("123", "23"));
  EXPECT_TRUE(BeginWith("123", ""));
  EXPECT_FALSE(BeginWith("", "123"));
}

TEST_F(StrUtilTest, EndWith) {
  EXPECT_TRUE(EndWith("123", "3"));
  EXPECT_TRUE(EndWith("123", "23"));
  EXPECT_TRUE(EndWith("123", "123"));
  EXPECT_FALSE(EndWith("123", "312"));
  EXPECT_FALSE(EndWith("123", "2"));
  EXPECT_FALSE(EndWith("123", "12"));
  EXPECT_TRUE(EndWith("123", ""));
  EXPECT_FALSE(EndWith("", "123"));
}

TEST_F(StrUtilTest, Split) {
  vs_t vs;
  Split("", ";", &vs);
  EXPECT_TRUE(vs.empty());

  Split("1", ";", &vs);
  EXPECT_EQ(vs, vs_t{"1"});

  Split("123", ";", &vs);
  EXPECT_EQ(vs, vs_t{"123"});

  Split("12;34;5;6;7;;8;9;10;;", ";", &vs);
  EXPECT_EQ(vs, vs_t({"12", "34", "5", "6", "7", "8", "9", "10"}));

  Split("12;34;5;6;7;;8;9;10;;", ";", &vs, false);
  EXPECT_EQ(vs, vs_t({"12", "34", "5", "6", "7", "", "8", "9", "10", "", ""}));
}

TEST_F(StrUtilTest, Split_int) {
  vi_t vi;
  ASSERT_TRUE(Split("", ";", &vi));
  EXPECT_TRUE(vi.empty());

  ASSERT_TRUE(Split("1", ";", &vi));
  EXPECT_EQ(vi, vi_t{1});

  ASSERT_TRUE(Split("123", ";", &vi));
  EXPECT_EQ(vi, vi_t{123});

  ASSERT_TRUE(Split("12;34;5;6;7;;8;9;10;;", ";", &vi));
  EXPECT_EQ(vi, vi_t({12, 34, 5, 6, 7, 8, 9, 10}));

  ASSERT_FALSE(Split("12;34;5;6;7;;8;9;10;;", ";", &vi, false));
}

TEST_F(StrUtilTest, Join) {
  EXPECT_EQ(Join({}, ","), "");
  EXPECT_EQ(Join({"1", "2"}, ","), "1,2");
  EXPECT_EQ(Join({"aa", "bb", "cc", "dd"}, ";"), "aa;bb;cc;dd");
  EXPECT_EQ(Join({"aa", "bb", "cc", "dd"}, ""), "aabbccdd");
}

}  // namespace deepx_core
