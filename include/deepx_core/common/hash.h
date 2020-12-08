// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <cstdint>
#include <string>

namespace deepx_core {

// MurmurHash.
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash2.cpp
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp

inline uint64_t MurmurHash3Mix(uint64_t k) noexcept {
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

uint64_t MurmurHash2(const void* data, size_t size, uint64_t seed) noexcept;

inline uint64_t MurmurHash2(const std::string& k) noexcept {
  return MurmurHash2(k.data(), k.size(), UINT64_C(0xc70f6907));
}

template <typename T>
struct MurmurHash;

#define DEFINE_MURMUR_HASH_T_IS_INTEGRAL(T)       \
  template <>                                     \
  struct MurmurHash<T> {                          \
    size_t operator()(T k) const noexcept {       \
      return (size_t)MurmurHash3Mix((uint64_t)k); \
    }                                             \
  };

DEFINE_MURMUR_HASH_T_IS_INTEGRAL(int8_t)
DEFINE_MURMUR_HASH_T_IS_INTEGRAL(uint8_t)
DEFINE_MURMUR_HASH_T_IS_INTEGRAL(int16_t)
DEFINE_MURMUR_HASH_T_IS_INTEGRAL(uint16_t)
DEFINE_MURMUR_HASH_T_IS_INTEGRAL(int32_t)
DEFINE_MURMUR_HASH_T_IS_INTEGRAL(uint32_t)
DEFINE_MURMUR_HASH_T_IS_INTEGRAL(int64_t)
DEFINE_MURMUR_HASH_T_IS_INTEGRAL(uint64_t)

#undef DEFINE_MURMUR_HASH_T_IS_INTEGRAL

template <>
struct MurmurHash<std::string> {
  size_t operator()(const std::string& k) const noexcept {
    return (size_t)MurmurHash2(k);
  }
};

}  // namespace deepx_core
