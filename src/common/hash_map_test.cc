// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/hash_map.h>
#include <deepx_core/common/hash_map_io.h>
#include <deepx_core/common/stream.h>
#include <gtest/gtest.h>
#include <utility>

namespace deepx_core {

class HashMapTest : public testing::Test {
 protected:
  using hash_map_t = HashMap<int, int>;
  const int N = 10000;
};

TEST_F(HashMapTest, Construct_ii) {
  hash_map_t hash_map1{{0, 0}, {1, 1}};
  hash_map_t hash_map2(hash_map1.begin(), hash_map1.end(), 0);
  EXPECT_EQ(hash_map1, hash_map2);
}

TEST_F(HashMapTest, Copy) {
  hash_map_t hash_map1{{0, 0}, {1, 1}};
  hash_map_t hash_map2(hash_map1);
  hash_map_t hash_map3;
  hash_map3 = hash_map2;
  EXPECT_EQ(hash_map1, hash_map2);
  EXPECT_EQ(hash_map1, hash_map3);
}

TEST_F(HashMapTest, Move) {
  hash_map_t hash_map1{{0, 0}, {1, 1}};

  hash_map_t hash_map2(std::move(hash_map1));
  EXPECT_EQ(hash_map2, hash_map_t({{0, 0}, {1, 1}}));

  hash_map_t hash_map3;
  hash_map3 = std::move(hash_map2);
  EXPECT_EQ(hash_map3, hash_map_t({{0, 0}, {1, 1}}));
}

TEST_F(HashMapTest, Construct_std_il) {
  hash_map_t hash_map1{{0, 0}, {1, 1}};
  EXPECT_EQ(hash_map1.size(), 2u);
  EXPECT_FALSE(hash_map1.empty());
  EXPECT_EQ(hash_map1[0], 0);
  EXPECT_EQ(hash_map1[1], 1);

  hash_map_t hash_map2;
  hash_map2 = {{2, 2}, {3, 3}};
  EXPECT_EQ(hash_map2.size(), 2u);
  EXPECT_FALSE(hash_map2.empty());
  EXPECT_EQ(hash_map2[2], 2);
  EXPECT_EQ(hash_map2[3], 3);
}

TEST_F(HashMapTest, iterator) {
  hash_map_t hash_map{{0, 0}, {1, 1}, {2, 2}, {3, 3}};
  int sum = 0;
  for (const auto& entry : hash_map) {
    sum += entry.second;
  }
  EXPECT_EQ(sum, 6);

  const hash_map_t& chash_map = hash_map;
  sum = 0;
  for (const auto& entry : chash_map) {
    sum += entry.second;
  }
  EXPECT_EQ(sum, 6);
}

TEST_F(HashMapTest, Compare) {
  hash_map_t hash_map1{{0, 0}, {1, 1}, {2, 2}};
  hash_map_t hash_map2{{1, 1}, {2, 2}, {0, 0}};
  hash_map_t hash_map3{{0, 0}, {1, 1}};
  EXPECT_TRUE(hash_map1 == hash_map2);
  EXPECT_FALSE(hash_map1 != hash_map2);
  EXPECT_FALSE(hash_map1 == hash_map3);
  EXPECT_TRUE(hash_map1 != hash_map3);
}

TEST_F(HashMapTest, Subscript) {
  hash_map_t hash_map;
  hash_map[0] = 0;
  EXPECT_EQ(hash_map[0], 0);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}}));
  hash_map[1] = 1;
  EXPECT_EQ(hash_map[1], 1);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}, {1, 1}}));
  hash_map[2];
  EXPECT_EQ(hash_map[2], 0);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}, {1, 1}, {2, 0}}));
  hash_map[3] = 3;
  EXPECT_EQ(hash_map[3], 3);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}, {1, 1}, {2, 0}, {3, 3}}));
}

TEST_F(HashMapTest, at) {
  hash_map_t hash_map{{0, 0}};
  EXPECT_EQ(hash_map.at(0), 0);
  hash_map.at(0) = 1;
  EXPECT_EQ(hash_map, hash_map_t({{0, 1}}));
  EXPECT_ANY_THROW(hash_map.at(1));

  const hash_map_t& chash_map = hash_map;
  EXPECT_EQ(chash_map.at(0), 1);
  EXPECT_ANY_THROW(chash_map.at(1));
}

TEST_F(HashMapTest, find) {
  hash_map_t hash_map{{0, 0}};
  const hash_map_t& chash_map = hash_map;
  EXPECT_NE(hash_map.find(0), hash_map.end());
  EXPECT_NE(hash_map.find(0), chash_map.end());
  EXPECT_EQ(hash_map.find(0)->second, 0);
  EXPECT_EQ(hash_map.find(1), hash_map.end());
  EXPECT_EQ(hash_map.find(1), chash_map.end());
  EXPECT_NE(chash_map.find(0), hash_map.end());
  EXPECT_NE(chash_map.find(0), chash_map.end());
  EXPECT_EQ(chash_map.find(0)->second, 0);
  EXPECT_EQ(chash_map.find(1), hash_map.end());
  EXPECT_EQ(chash_map.find(1), chash_map.end());
}

TEST_F(HashMapTest, count) {
  hash_map_t hash_map{{0, 0}};
  EXPECT_EQ(hash_map.count(0), 1u);
  EXPECT_EQ(hash_map.count(1), 0u);
}

TEST_F(HashMapTest, insert) {
  hash_map_t hash_map;
  auto ii = hash_map.insert(std::make_pair(0, 0));
  EXPECT_TRUE(ii.second);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}}));
  ii = hash_map.insert(std::make_pair(0, 0));
  EXPECT_FALSE(ii.second);
  ii = hash_map.insert(std::make_pair(1, 1));
  EXPECT_TRUE(ii.second);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}, {1, 1}}));
  ii.first->second = 2;
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}, {1, 2}}));
}

TEST_F(HashMapTest, insert_ii) {
  hash_map_t hash_map1{{0, 0}, {1, 1}};
  hash_map_t hash_map2;
  hash_map2.insert(hash_map1.begin(), hash_map1.end());
  EXPECT_EQ(hash_map1, hash_map2);
}

TEST_F(HashMapTest, emplace) {
  hash_map_t hash_map;
  auto ii = hash_map.emplace(std::make_pair(0, 0));
  EXPECT_TRUE(ii.second);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}}));
  ii = hash_map.emplace(0, 0);
  EXPECT_FALSE(ii.second);
  ii = hash_map.emplace(1, 1);
  EXPECT_TRUE(ii.second);
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}, {1, 1}}));
  ii.first->second = 2;
  EXPECT_EQ(hash_map, hash_map_t({{0, 0}, {1, 2}}));
}

TEST_F(HashMapTest, erase_1) {
  hash_map_t hash_map;
  for (int i = 0; i < N; ++i) {
    hash_map.emplace(i, i);
  }
  EXPECT_EQ(hash_map.size(), (size_t)N);

  for (int i = 0; i < N; ++i) {
    auto it = hash_map.find(i);
    EXPECT_NE(it, hash_map.end());
    it = hash_map.erase(it);
    if (i != N - 1) {
      EXPECT_NE(it, hash_map.end());
    } else {
      EXPECT_EQ(it, hash_map.end());
    }
    EXPECT_EQ(hash_map.size(), (size_t)(N - 1 - i));
  }
  EXPECT_TRUE(hash_map.empty());
}

TEST_F(HashMapTest, erase_2) {
  hash_map_t hash_map;
  for (int i = 0; i < N; ++i) {
    hash_map.emplace(i, i);
  }
  EXPECT_EQ(hash_map.size(), (size_t)N);

  int n = 0;
  for (auto it = hash_map.begin(); it != hash_map.end();) {
    it = hash_map.erase(it);
    ++n;
  }
  EXPECT_EQ(n, N);
  EXPECT_TRUE(hash_map.empty());
}

TEST_F(HashMapTest, clear) {
  hash_map_t hash_map{{0, 0}};
  EXPECT_FALSE(hash_map.empty());
  hash_map.clear();
  EXPECT_TRUE(hash_map.empty());
}

TEST_F(HashMapTest, rehash) {
  hash_map_t hash_map;
  hash_map.rehash(N);
  size_t bucket_size = hash_map.bucket_size();
  for (int i = 0; i < N; ++i) {
    hash_map.emplace(i, i);
    EXPECT_EQ(bucket_size, hash_map.bucket_size());
  }
}

TEST_F(HashMapTest, swap) {
  hash_map_t hash_map1{{0, 0}, {1, 1}};
  hash_map_t hash_map2{{2, 2}, {3, 3}};
  hash_map1.swap(hash_map2);
  EXPECT_EQ(hash_map1, hash_map_t({{2, 2}, {3, 3}}));
  EXPECT_EQ(hash_map2, hash_map_t({{0, 0}, {1, 1}}));
}

TEST_F(HashMapTest, WriteRead) {
  hash_map_t hash_map{{0, 0}, {1, 1}}, read_hash_map;

  OutputStringStream os;
  InputStringStream is;

  os << hash_map;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_hash_map;
  ASSERT_TRUE(is);

  EXPECT_EQ(hash_map, read_hash_map);
}

TEST_F(HashMapTest, WriteReadView) {
  hash_map_t hash_map{{0, 0}, {1, 1}}, read_hash_map;

  OutputStringStream os;
  InputStringStream is;

  os << hash_map;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_hash_map);
  ASSERT_TRUE(is);

  EXPECT_EQ(hash_map, read_hash_map);
}

}  // namespace deepx_core
