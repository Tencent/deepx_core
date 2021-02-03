// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/lru_cache.h>
#include <gtest/gtest.h>

namespace deepx_core {

class LRUCacheTest : public testing::Test {};

TEST_F(LRUCacheTest, Evict) {
  LRUCache<int, int> cache;
  EXPECT_EQ(cache.size(), 0u);
  EXPECT_EQ(cache.capacity(), 0u);
  EXPECT_TRUE(cache.empty());

  cache.init(4);
  EXPECT_EQ(cache.size(), 0u);
  EXPECT_EQ(cache.capacity(), 4u);
  EXPECT_TRUE(cache.empty());

  cache.insert(0, 0);
  cache.insert(1, 1);
  cache.insert(2, 2);
  cache.insert(3, 3);
  EXPECT_EQ(cache.size(), 4u);
  // 0, 1, 2, 3

  {
    auto node = cache.get_or_insert(4);
    EXPECT_EQ(node->value(), 0);
    *node->mutable_value() = 4;
  }
  // 1, 2, 3, 4

  {
    auto node = cache.get_or_insert(1);
    EXPECT_EQ(node->value(), 1);
  }
  // 2, 3, 4, 1

  {
    auto node = cache.emplace(5, 5);
    EXPECT_EQ(node->value(), 5);
  }
  // 3, 4, 1, 5

  EXPECT_FALSE(cache.get(0));
  EXPECT_FALSE(cache.get(2));
}

TEST_F(LRUCacheTest, erase) {
  LRUCache<int, int> cache;

  cache.init(4);

  cache.insert(0, 0);
  cache.insert(1, 1);
  cache.insert(2, 2);
  cache.insert(3, 3);
  EXPECT_EQ(cache.size(), 4u);

  cache.erase(0);
  EXPECT_EQ(cache.size(), 3u);

  cache.erase(1);
  EXPECT_EQ(cache.size(), 2u);

  cache.erase(2);
  EXPECT_EQ(cache.size(), 1u);

  cache.erase(3);
  EXPECT_EQ(cache.size(), 0u);
}

}  // namespace deepx_core
