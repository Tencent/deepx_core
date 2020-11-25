// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* GroupConfigItem3 */
/************************************************************************/
struct GroupConfigItem3 {
  // group id
  uint16_t group_id = 0;
  // row of embedding lookup matrix
  int embedding_row = 0;
  // col of embedding lookup matrix
  int embedding_col = 0;
};

OutputStream& operator<<(OutputStream& os, const GroupConfigItem3& item);
InputStream& operator>>(InputStream& is, GroupConfigItem3& item);
std::istream& operator>>(std::istream& is, GroupConfigItem3& item);

/************************************************************************/
/* GroupConfigItem3 functions */
/************************************************************************/
// Load group config from 'file'.
bool LoadGroupConfig(const std::string& file,
                     std::vector<GroupConfigItem3>* items, int* max_group_id);
bool LoadGroupConfig(const std::string& file,
                     std::vector<GroupConfigItem3>* items, int* max_group_id,
                     const char* gflag);
// Parse group config from string 'info'.
bool ParseGroupConfig(const std::string& info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id);
bool ParseGroupConfig(const std::string& info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id,
                      const char* gflag);
// Try to load from file or parse from string.
bool GuessGroupConfig(const std::string& file_or_info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id);
bool GuessGroupConfig(const std::string& file_or_info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id,
                      const char* gflag);

// Get the corresponding LR/linear group config of 'items'.
std::vector<GroupConfigItem3> GetLRGroupConfig(
    const std::vector<GroupConfigItem3>& items);

// Return if 'items' can be a FM group config.
//
// 'IsFMGroupConfig' is silent.
// 'CheckFMGroupConfig' is verbose.
bool IsFMGroupConfig(const std::vector<GroupConfigItem3>& items);
bool CheckFMGroupConfig(const std::vector<GroupConfigItem3>& items);

// Get the total col of all embedding lookup matrices.
int GetTotalEmbeddingCol(const std::vector<GroupConfigItem3>& items);

}  // namespace deepx_core
