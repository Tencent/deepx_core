// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/group_config.h>
#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>
#include <limits>  // std::numeric_limits
#include <sstream>
#include <unordered_set>

namespace deepx_core {

/************************************************************************/
/* GroupConfigItem3 */
/************************************************************************/
OutputStream& operator<<(OutputStream& os, const GroupConfigItem3& item) {
  os << item.group_id << item.embedding_row << item.embedding_col;
  return os;
}

InputStream& operator>>(InputStream& is, GroupConfigItem3& item) {
  is >> item.group_id >> item.embedding_row >> item.embedding_col;
  return is;
}

std::istream& operator>>(std::istream& is, GroupConfigItem3& item) {
  is >> item.group_id >> item.embedding_row >> item.embedding_col;
  return is;
}

/************************************************************************/
/* GroupConfigItem3 functions */
/************************************************************************/
bool LoadGroupConfig(const std::string& file,
                     std::vector<GroupConfigItem3>* items, int* max_group_id) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  std::string line;
  std::istringstream iss;
  std::unordered_set<uint16_t> dedup;

  items->clear();
  if (max_group_id) {
    *max_group_id = 0;
  }

  while (GetLine(is, line)) {
    if (line.find('#') != std::string::npos) {
      continue;
    }

    if (line.find("//") != std::string::npos) {
      continue;
    }

    GroupConfigItem3 item;
    iss.clear();
    iss.str(line);
    if (!(iss >> item)) {
      DXERROR("Invalid line: %s.", line.c_str());
      return false;
    }

    if (dedup.count(item.group_id) > 0) {
      DXERROR("Duplicate group id: %d.", (int)item.group_id);
      return false;
    }

    if (item.embedding_row <= 0) {
      DXERROR("Invalid embedding row: %d.", item.embedding_row);
      return false;
    }

    if (item.embedding_col <= 0) {
      DXERROR("Invalid embedding col: %d.", item.embedding_col);
      return false;
    }

    if ((uint64_t)item.embedding_row * (uint64_t)item.embedding_col >
        (uint64_t)std::numeric_limits<int>::max()) {
      DXERROR("Too large embedding row and embedding col: %d %d.",
              item.embedding_row, item.embedding_col);
      return false;
    }

    items->emplace_back(item);
    dedup.emplace(item.group_id);
    if (max_group_id) {
      if (*max_group_id < (int)item.group_id) {
        *max_group_id = (int)item.group_id;
      }
    }
  }

  if (max_group_id) {
    ++(*max_group_id);
  }
  DXINFO("Loaded %zu groups.", items->size());
  return !items->empty();
}

bool LoadGroupConfig(const std::string& file,
                     std::vector<GroupConfigItem3>* items, int* max_group_id,
                     const char* gflag) {
  if (file.empty()) {
    DXERROR("Please specify %s.", gflag);
    return false;
  }
  return LoadGroupConfig(file, items, max_group_id);
}

bool ParseGroupConfig(const std::string& info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id) {
  std::unordered_set<uint16_t> dedup;
  std::vector<std::string> str_items;
  std::vector<int> int_items;

  items->clear();
  if (max_group_id) {
    *max_group_id = 0;
  }

  Split(info, ",", &str_items);
  for (const std::string& str_item : str_items) {
    GroupConfigItem3 item;
    if (!Split(str_item, ":", &int_items)) {
      DXERROR("Invalid info: %s.", info.c_str());
      return false;
    }

    if (int_items[0] < 0 ||
        int_items[0] > (int)std::numeric_limits<uint16_t>::max()) {
      DXERROR("Invalid group id: %d.", int_items[0]);
      return false;
    }

    if (int_items.size() == 2) {
      item.group_id = (uint16_t)int_items[0];
      item.embedding_row = 1;
      item.embedding_col = int_items[1];
    } else if (int_items.size() == 3) {
      item.group_id = (uint16_t)int_items[0];
      item.embedding_row = int_items[1];
      item.embedding_col = int_items[2];
    } else {
      DXERROR("Invalid info: %s.", info.c_str());
      return false;
    }

    if (dedup.count(item.group_id) > 0) {
      DXERROR("Duplicate group id: %d.", (int)item.group_id);
      return false;
    }

    if (item.embedding_row <= 0) {
      DXERROR("Invalid embedding row: %d.", item.embedding_row);
      return false;
    }

    if (item.embedding_col <= 0) {
      DXERROR("Invalid embedding col: %d.", item.embedding_col);
      return false;
    }

    if ((uint64_t)item.embedding_row * (uint64_t)item.embedding_col >
        (uint64_t)std::numeric_limits<int>::max()) {
      DXERROR("Too large embedding row and embedding col: %d %d.",
              item.embedding_row, item.embedding_col);
      return false;
    }

    items->emplace_back(item);
    dedup.emplace(item.group_id);
    if (max_group_id) {
      if (*max_group_id < (int)item.group_id) {
        *max_group_id = (int)item.group_id;
      }
    }
  }

  if (max_group_id) {
    ++(*max_group_id);
  }
  DXINFO("Loaded %zu groups.", items->size());
  return !items->empty();
}

bool ParseGroupConfig(const std::string& info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id,
                      const char* gflag) {
  if (info.empty()) {
    DXERROR("Please specify %s.", gflag);
    return false;
  }
  return ParseGroupConfig(info, items, max_group_id);
}

bool GuessGroupConfig(const std::string& file_or_info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id) {
  AutoFileSystem fs;
  if (fs.Open(file_or_info) && fs.IsFile(file_or_info)) {
    return LoadGroupConfig(file_or_info, items, max_group_id);
  } else {
    return ParseGroupConfig(file_or_info, items, max_group_id);
  }
}

bool GuessGroupConfig(const std::string& file_or_info,
                      std::vector<GroupConfigItem3>* items, int* max_group_id,
                      const char* gflag) {
  if (file_or_info.empty()) {
    DXERROR("Please specify %s.", gflag);
    return false;
  }
  return GuessGroupConfig(file_or_info, items, max_group_id);
}

std::vector<GroupConfigItem3> GetLRGroupConfig(
    const std::vector<GroupConfigItem3>& items) {
  std::vector<GroupConfigItem3> lr_items(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    const GroupConfigItem3& item = items[i];
    GroupConfigItem3& lr_item = lr_items[i];
    lr_item.group_id = item.group_id;
    lr_item.embedding_row = item.embedding_row;
    lr_item.embedding_col = 1;
  }
  return lr_items;
}

bool IsFMGroupConfig(const std::vector<GroupConfigItem3>& items) {
  if (items.empty()) {
    return false;
  }

  int k = items.front().embedding_col;
  for (const GroupConfigItem3& item : items) {
    if (k != item.embedding_col) {
      return false;
    }
  }
  return true;
}

bool CheckFMGroupConfig(const std::vector<GroupConfigItem3>& items) {
  if (items.empty()) {
    DXERROR("items is empty.");
    return false;
  }

  int k = items.front().embedding_col;
  for (const GroupConfigItem3& item : items) {
    if (k != item.embedding_col) {
      DXERROR("Inconsistent embedding col: %d vs %d.", k, item.embedding_col);
      return false;
    }
  }
  return true;
}

int GetTotalEmbeddingCol(const std::vector<GroupConfigItem3>& items) {
  int total = 0;
  for (const GroupConfigItem3& item : items) {
    total += item.embedding_col;
  }
  return total;
}

}  // namespace deepx_core
