// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <string>
#include <vector>

namespace deepx_core {

constexpr int MAX_GROUP_ID = (1 << 24) - 1;

/************************************************************************/
/* GroupConfigItem */
/************************************************************************/
struct GroupConfigItem {
  // group id
  int group_id = 0;
  // row of embedding lookup matrix
  int embedding_row = 0;
  // col of embedding lookup matrix
  int embedding_col = 0;
};

// backward compatibility
using GroupConfigItem3 = GroupConfigItem;

/************************************************************************/
/* GroupConfigItem functions */
/************************************************************************/
// Load 'items' from 'file'.
// 'max_group_id' can be nullptr.
bool LoadGroupConfig(const std::string& file,
                     std::vector<GroupConfigItem>* items, int* max_group_id);
bool LoadGroupConfig(const std::string& file,
                     std::vector<GroupConfigItem>* items, int* max_group_id,
                     const char* gflag);
// Parse 'items' from string 'info'.
// 'max_group_id' can be nullptr.
bool ParseGroupConfig(const std::string& info,
                      std::vector<GroupConfigItem>* items, int* max_group_id);
bool ParseGroupConfig(const std::string& info,
                      std::vector<GroupConfigItem>* items, int* max_group_id,
                      const char* gflag);
// Try to load 'items' from file or parse 'items' from string.
// 'max_group_id' can be nullptr.
bool GuessGroupConfig(const std::string& file_or_info,
                      std::vector<GroupConfigItem>* items, int* max_group_id);
bool GuessGroupConfig(const std::string& file_or_info,
                      std::vector<GroupConfigItem>* items, int* max_group_id,
                      const char* gflag);

// Get the corresponding LR/linear group config of 'items'.
std::vector<GroupConfigItem> GetLRGroupConfig(
    const std::vector<GroupConfigItem>& items);

// Return if 'items' can be a FM group config.
//
// 'IsFMGroupConfig' is silent.
// 'CheckFMGroupConfig' is verbose.
bool IsFMGroupConfig(const std::vector<GroupConfigItem>& items);
bool CheckFMGroupConfig(const std::vector<GroupConfigItem>& items);

// Get the total col of all embedding lookup matrices.
int GetTotalEmbeddingCol(const std::vector<GroupConfigItem>& items);

}  // namespace deepx_core
