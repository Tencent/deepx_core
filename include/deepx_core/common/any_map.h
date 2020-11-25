// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/any.h>
#include <string>
#include <unordered_map>

namespace deepx_core {

/************************************************************************/
/* AnyMap */
/************************************************************************/
class AnyMap : public std::unordered_map<std::string, Any> {
 public:
  template <typename T>
  T& get_or_insert(const std::string& k) {
    auto it = find(k);
    if (it != end()) {
      return it->second.to_ref<T>();
    }
    Any& a = (*this)[k];
    a.emplace(T());
    return a.unsafe_to_ref<T>();
  }

  template <typename T>
  T& insert(const std::string& k) {
    Any& a = (*this)[k];
    a.emplace(T());
    return a.unsafe_to_ref<T>();
  }

  template <typename T>
  T& get(const std::string& k) {
    return at(k).to_ref<T>();
  }

  template <typename T>
  const T& get(const std::string& k) const {
    return at(k).to_ref<T>();
  }

  template <typename T>
  T& unsafe_get(const std::string& k) {
    return at(k).unsafe_to_ref<T>();
  }

  template <typename T>
  const T& unsafe_get(const std::string& k) const {
    return at(k).unsafe_to_ref<T>();
  }

  // If 'config' is like "a=b;c=d;...",
  // then 'this' will be like {"a":"b", "c":"d", ...}.
  bool ParseConfig(const std::string& config);
};

/************************************************************************/
/* AnyMap & StringMap functions */
/************************************************************************/
using StringMap = std::unordered_map<std::string, std::string>;

// If 'config' is like "a=b;c=d;...",
// then 'map' will be like {"a":"b", "c":"d", ...}.
bool ParseConfig(const std::string& config, AnyMap* map);
bool ParseConfig(const std::string& config, StringMap* map);

void StringMapToAnyMap(const StringMap& from, AnyMap* to);
void AnyMapToStringMap(const AnyMap& from, StringMap* to);

}  // namespace deepx_core
