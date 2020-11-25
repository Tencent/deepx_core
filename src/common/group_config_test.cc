// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/group_config.h>
#include <gtest/gtest.h>
#include <vector>

namespace deepx_core {

class GroupConfigTest : public testing::Test {};

TEST_F(GroupConfigTest, LoadGroupConfig_1) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(LoadGroupConfig("testdata/common/group_config/1.txt", &items,
                              &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);

  std::vector<GroupConfigItem3> lr_items = GetLRGroupConfig(items);
  EXPECT_EQ(lr_items.size(), 3u);
  EXPECT_EQ(lr_items[0].group_id, 1);
  EXPECT_EQ(lr_items[1].group_id, 2);
  EXPECT_EQ(lr_items[2].group_id, 3);
  EXPECT_EQ(lr_items[0].embedding_row, 10);
  EXPECT_EQ(lr_items[1].embedding_row, 10);
  EXPECT_EQ(lr_items[2].embedding_row, 10);
  EXPECT_EQ(lr_items[0].embedding_col, 1);
  EXPECT_EQ(lr_items[1].embedding_col, 1);
  EXPECT_EQ(lr_items[2].embedding_col, 1);
  EXPECT_TRUE(IsFMGroupConfig(lr_items));
  EXPECT_EQ(GetTotalEmbeddingCol(lr_items), 3);
}

TEST_F(GroupConfigTest, LoadGroupConfig_2) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(LoadGroupConfig("testdata/common/group_config/2.txt", &items,
                              &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 10);
  EXPECT_EQ(items[1].group_id, 20);
  EXPECT_EQ(items[2].group_id, 30);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 31);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);

  std::vector<GroupConfigItem3> lr_items = GetLRGroupConfig(items);
  EXPECT_EQ(lr_items.size(), 3u);
  EXPECT_EQ(lr_items[0].group_id, 10);
  EXPECT_EQ(lr_items[1].group_id, 20);
  EXPECT_EQ(lr_items[2].group_id, 30);
  EXPECT_EQ(lr_items[0].embedding_row, 10);
  EXPECT_EQ(lr_items[1].embedding_row, 10);
  EXPECT_EQ(lr_items[2].embedding_row, 10);
  EXPECT_EQ(lr_items[0].embedding_col, 1);
  EXPECT_EQ(lr_items[1].embedding_col, 1);
  EXPECT_EQ(lr_items[2].embedding_col, 1);
  EXPECT_TRUE(IsFMGroupConfig(lr_items));
  EXPECT_EQ(GetTotalEmbeddingCol(lr_items), 3);
}

TEST_F(GroupConfigTest, LoadGroupConfig_3) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(LoadGroupConfig("testdata/common/group_config/3.txt", &items,
                              &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 4);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 8);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_FALSE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);

  std::vector<GroupConfigItem3> lr_items = GetLRGroupConfig(items);
  EXPECT_EQ(lr_items.size(), 3u);
  EXPECT_EQ(lr_items[0].group_id, 1);
  EXPECT_EQ(lr_items[1].group_id, 2);
  EXPECT_EQ(lr_items[2].group_id, 3);
  EXPECT_EQ(lr_items[0].embedding_row, 10);
  EXPECT_EQ(lr_items[1].embedding_row, 10);
  EXPECT_EQ(lr_items[2].embedding_row, 10);
  EXPECT_EQ(lr_items[0].embedding_col, 1);
  EXPECT_EQ(lr_items[1].embedding_col, 1);
  EXPECT_EQ(lr_items[2].embedding_col, 1);
  EXPECT_TRUE(IsFMGroupConfig(lr_items));
  EXPECT_EQ(GetTotalEmbeddingCol(lr_items), 3);
}

TEST_F(GroupConfigTest, ParseGroupConfig_1) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(ParseGroupConfig("1:10:6,2:10:6,3:10:6", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);

  ASSERT_TRUE(ParseGroupConfig("1:6,2:6,3:6", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 1);
  EXPECT_EQ(items[1].embedding_row, 1);
  EXPECT_EQ(items[2].embedding_row, 1);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);
}

TEST_F(GroupConfigTest, ParseGroupConfig_2) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(
      ParseGroupConfig("10:10:6,20:10:6,30:10:6", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 10);
  EXPECT_EQ(items[1].group_id, 20);
  EXPECT_EQ(items[2].group_id, 30);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 31);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);

  ASSERT_TRUE(ParseGroupConfig("10:6,20:6,30:6", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 10);
  EXPECT_EQ(items[1].group_id, 20);
  EXPECT_EQ(items[2].group_id, 30);
  EXPECT_EQ(items[0].embedding_row, 1);
  EXPECT_EQ(items[1].embedding_row, 1);
  EXPECT_EQ(items[2].embedding_row, 1);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 31);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);
}

TEST_F(GroupConfigTest, ParseGroupConfig_3) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(ParseGroupConfig("1:10:4,2:10:6,3:10:8", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 4);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 8);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_FALSE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);

  ASSERT_TRUE(ParseGroupConfig("1:4,2:6,3:8", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 1);
  EXPECT_EQ(items[1].embedding_row, 1);
  EXPECT_EQ(items[2].embedding_row, 1);
  EXPECT_EQ(items[0].embedding_col, 4);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 8);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_FALSE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);
}

TEST_F(GroupConfigTest, GuessGroupConfig_1) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(GuessGroupConfig("testdata/common/group_config/1.txt", &items,
                               &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);
}

TEST_F(GroupConfigTest, GuessGroupConfig_2) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(GuessGroupConfig("1:10:6,2:10:6,3:10:6", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 10);
  EXPECT_EQ(items[1].embedding_row, 10);
  EXPECT_EQ(items[2].embedding_row, 10);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);
}

TEST_F(GroupConfigTest, GuessGroupConfig_3) {
  std::vector<GroupConfigItem3> items;
  int max_group_id;

  ASSERT_TRUE(GuessGroupConfig("1:6,2:6,3:6", &items, &max_group_id));
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items[0].group_id, 1);
  EXPECT_EQ(items[1].group_id, 2);
  EXPECT_EQ(items[2].group_id, 3);
  EXPECT_EQ(items[0].embedding_row, 1);
  EXPECT_EQ(items[1].embedding_row, 1);
  EXPECT_EQ(items[2].embedding_row, 1);
  EXPECT_EQ(items[0].embedding_col, 6);
  EXPECT_EQ(items[1].embedding_col, 6);
  EXPECT_EQ(items[2].embedding_col, 6);
  EXPECT_EQ(max_group_id, 4);
  EXPECT_TRUE(IsFMGroupConfig(items));
  EXPECT_EQ(GetTotalEmbeddingCol(items), 18);
}

}  // namespace deepx_core
