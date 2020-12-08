// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/hash_map.h>
#include <deepx_core/common/stream.h>
#include <cstdint>
#include <utility>

namespace deepx_core {

template <typename Key, typename Value, class KeyHash, class KeyEqual>
OutputStream& operator<<(OutputStream& os,
                         const HashMap<Key, Value, KeyHash, KeyEqual>& m) {
  int version = 0x0a0c72e7;            // magic number version
  uint64_t size = (uint64_t)m.size();  // NOLINT
  os << version;
  os << size;

  auto first = m.begin();
  auto last = m.end();
  for (; first != last; ++first) {
    os << first->first << first->second;
    if (!os) {
      break;
    }
  }
  return os;
}

template <typename Key, typename Value, class KeyHash, class KeyEqual>
InputStream& operator>>(InputStream& is,
                        HashMap<Key, Value, KeyHash, KeyEqual>& m) {
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (size_t)size_u64;
  } else {
    // backward compatibility
    int size_i = 0;
    is >> size_i;
    size = (size_t)size_i;
  }
  if (!is) {
    return is;
  }

  m.clear();
  if (size > 0) {
    Key key;
    Value value;
    m.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      is >> key >> value;
      if (!is) {
        return is;
      }
      m.emplace(std::move(key), std::move(value));
    }
  }
  return is;
}

template <typename Key, typename Value, class KeyHash, class KeyEqual>
InputStringStream& ReadView(
    InputStringStream& is,                        // NOLINT
    HashMap<Key, Value, KeyHash, KeyEqual>& m) {  // NOLINT
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (size_t)size_u64;
  } else {
    // backward compatibility
    int size_i = 0;
    is >> size_i;
    size = (size_t)size_i;
  }
  if (!is) {
    return is;
  }

  m.clear();
  if (size > 0) {
    Key key;
    Value value;
    m.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      ReadView(is, key);
      ReadView(is, value);
      if (!is) {
        return is;
      }
      m.emplace(std::move(key), std::move(value));
    }
  }
  return is;
}

}  // namespace deepx_core
