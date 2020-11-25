// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/common/str_util.h>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* AnyMap */
/************************************************************************/
bool AnyMap::ParseConfig(const std::string& config) {
  return deepx_core::ParseConfig(config, this);
}

/************************************************************************/
/* AnyMap & StringMap functions */
/************************************************************************/
bool ParseConfig(const std::string& config, AnyMap* map) {
  map->clear();
  std::vector<std::string> tmp1, tmp2;
  Split(config, ";", &tmp1, true);
  for (const std::string& tmp : tmp1) {
    Split(tmp, "=", &tmp2, true);
    if (tmp2.size() != 2) {
      return false;
    }
    (*map)[tmp2[0]] = tmp2[1];
  }
  return true;
}

bool ParseConfig(const std::string& config, StringMap* map) {
  map->clear();
  std::vector<std::string> tmp1, tmp2;
  Split(config, ";", &tmp1, true);
  for (const std::string& tmp : tmp1) {
    Split(tmp, "=", &tmp2, true);
    if (tmp2.size() != 2) {
      return false;
    }
    (*map)[tmp2[0]] = tmp2[1];
  }
  return true;
}

void StringMapToAnyMap(const StringMap& from, AnyMap* to) {
  to->clear();
  for (const auto& entry : from) {
    const std::string& k = entry.first;
    const std::string& v = entry.second;
    (*to)[k] = v;
  }
}

void AnyMapToStringMap(const AnyMap& from, StringMap* to) {
  to->clear();
  for (const auto& entry : from) {
    const std::string& k = entry.first;
    const auto& v = entry.second.to_ref<std::string>();
    (*to)[k] = v;
  }
}

}  // namespace deepx_core
