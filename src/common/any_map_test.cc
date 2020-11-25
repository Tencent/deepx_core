// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/any_map.h>
#include <gtest/gtest.h>
#include <string>

namespace deepx_core {

class AnyMapTest : public testing::Test {};

TEST_F(AnyMapTest, get_or_insert) {
  AnyMap map;
  map["0"] = true;
  map["1"] = 1;
  EXPECT_EQ(map.get_or_insert<bool>("0"), true);
  EXPECT_EQ(map.get_or_insert<int>("1"), 1);
  EXPECT_EQ(map.get_or_insert<bool>("2"), false);
  EXPECT_EQ(map.get_or_insert<int>("3"), 0);
  EXPECT_ANY_THROW(map.get_or_insert<int>("0"));
  EXPECT_ANY_THROW(map.get_or_insert<bool>("1"));
}

TEST_F(AnyMapTest, insert) {
  AnyMap map;
  EXPECT_EQ(map.insert<bool>("0"), false);
  EXPECT_EQ(map.insert<int>("1"), 0);
  EXPECT_EQ(map.insert<int>("0"), 0);
  EXPECT_EQ(map.insert<bool>("1"), false);
}

TEST_F(AnyMapTest, get) {
  AnyMap map;
  map["0"] = true;
  map["1"] = 1;
  EXPECT_EQ(map.get<bool>("0"), true);
  EXPECT_EQ(map.get<int>("1"), 1);
  EXPECT_ANY_THROW(map.get<int>("0"));
  EXPECT_ANY_THROW(map.get<bool>("1"));
  EXPECT_ANY_THROW(map.get<bool>("2"));
  EXPECT_ANY_THROW(map.get<int>("3"));
}

TEST_F(AnyMapTest, unsafe_get) {
  AnyMap map;
  map["0"] = true;
  map["1"] = 1;
  EXPECT_EQ(map.unsafe_get<bool>("0"), true);
  EXPECT_EQ(map.unsafe_get<int>("1"), 1);
  EXPECT_ANY_THROW(map.unsafe_get<bool>("2"));
  EXPECT_ANY_THROW(map.unsafe_get<int>("3"));
}

TEST_F(AnyMapTest, ParseConfig_AnyMap) {
  AnyMap map;
  ASSERT_TRUE(ParseConfig("a=b;c=d;e=f;", &map));
  EXPECT_EQ(map.size(), 3u);
  EXPECT_EQ(map.unsafe_get<std::string>("a"), "b");
  EXPECT_EQ(map.unsafe_get<std::string>("c"), "d");
  EXPECT_EQ(map.unsafe_get<std::string>("e"), "f");

  ASSERT_FALSE(ParseConfig("a=b;cd;e=f;", &map));
}

TEST_F(AnyMapTest, ParseConfig_StringMap) {
  StringMap map;
  ASSERT_TRUE(ParseConfig("a=b;c=d;e=f;", &map));
  EXPECT_EQ(map.size(), 3u);
  EXPECT_EQ(map.at("a"), "b");
  EXPECT_EQ(map.at("c"), "d");
  EXPECT_EQ(map.at("e"), "f");

  ASSERT_FALSE(ParseConfig("a=b;cd;e=f;", &map));
}

TEST_F(AnyMapTest, StringMapToAnyMap) {
  StringMap from;
  AnyMap to;
  ASSERT_TRUE(ParseConfig("a=b;c=d;e=f;", &from));
  StringMapToAnyMap(from, &to);
  EXPECT_EQ(to.size(), 3u);
  EXPECT_EQ(to.unsafe_get<std::string>("a"), "b");
  EXPECT_EQ(to.unsafe_get<std::string>("c"), "d");
  EXPECT_EQ(to.unsafe_get<std::string>("e"), "f");
}

TEST_F(AnyMapTest, AnyMapToStringMap) {
  AnyMap from;
  StringMap to;
  ASSERT_TRUE(ParseConfig("a=b;c=d;e=f;", &from));
  AnyMapToStringMap(from, &to);
  EXPECT_EQ(to.size(), 3u);
  EXPECT_EQ(to.at("a"), "b");
  EXPECT_EQ(to.at("c"), "d");
  EXPECT_EQ(to.at("e"), "f");
}

}  // namespace deepx_core
