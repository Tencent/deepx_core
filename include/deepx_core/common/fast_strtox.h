// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <cstdint>

namespace deepx_core {

double fast_strtod(const char* s, char** end) noexcept;
uint8_t fast_strtou8(const char* s, char** end) noexcept;
int8_t fast_strtoi8(const char* s, char** end) noexcept;
uint16_t fast_strtou16(const char* s, char** end) noexcept;
int16_t fast_strtoi16(const char* s, char** end) noexcept;
uint32_t fast_strtou32(const char* s, char** end) noexcept;
int32_t fast_strtoi32(const char* s, char** end) noexcept;
uint64_t fast_strtou64(const char* s, char** end) noexcept;
int64_t fast_strtoi64(const char* s, char** end) noexcept;

template <typename Int>
Int fast_strtoi(const char* s, char** end) noexcept;

template <>
inline uint8_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtou8(s, end);
}

template <>
inline int8_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtoi8(s, end);
}

template <>
inline uint16_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtou16(s, end);
}

template <>
inline int16_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtoi16(s, end);
}

template <>
inline uint32_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtou32(s, end);
}

template <>
inline int32_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtoi32(s, end);
}

template <>
inline uint64_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtou64(s, end);
}

template <>
inline int64_t fast_strtoi(const char* s, char** end) noexcept {
  return fast_strtoi64(s, end);
}

}  // namespace deepx_core
