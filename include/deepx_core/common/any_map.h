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
  // Get or insert value by 'key'.
  //
  // Return a reference to the associated value of T.
  // Throw std::bad_cast if the associated value is not of T.
  template <typename T>
  T& get_or_insert(const std::string& key) {
    auto it = find(key);
    if (it != end()) {
      return it->second.to_ref<T>();
    }
    Any& any = (*this)[key];
    any.emplace(T());
    return any.unsafe_to_ref<T>();
  }

  // Insert 'key', existing element will be replaced.
  //
  // Return a reference to the associated value of T.
  // NOTE: the semantic is different from STL's insert.
  template <typename T>
  T& insert(const std::string& key) {
    Any& any = (*this)[key];
    any.emplace(T());
    return any.unsafe_to_ref<T>();
  }

  // Get value by 'key'.
  //
  // Return a reference to the associated value of T.
  // Throw std::out_of_range if 'key' does not exist.
  // Throw std::bad_cast if the associated value is not of T.
  template <typename T>
  T& get(const std::string& key) {
    return at(key).to_ref<T>();
  }

  template <typename T>
  const T& get(const std::string& key) const {
    return at(key).to_ref<T>();
  }

  // Get value by 'key'.
  //
  // Return a reference to the associated value of T.
  // Throw std::out_of_range if 'key' does not exist.
  // The behavior is undefined if the associated value is not of T.
  template <typename T>
  T& unsafe_get(const std::string& key) {
    return at(key).unsafe_to_ref<T>();
  }

  template <typename T>
  const T& unsafe_get(const std::string& key) const {
    return at(key).unsafe_to_ref<T>();
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

// Convert from StringMap to AnyMap.
void StringMapToAnyMap(const StringMap& from, AnyMap* to);

// Convert from AnyMap to StringMap.
void AnyMapToStringMap(const AnyMap& from, StringMap* to);

}  // namespace deepx_core
