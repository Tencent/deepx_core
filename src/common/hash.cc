// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/hash.h>

namespace deepx_core {

uint64_t MurmurHash2(const void* data, size_t size, uint64_t seed) noexcept {
  static constexpr uint64_t M = UINT64_C(0xc6a4a7935bd1e995);
  static constexpr uint64_t R = 47;

  uint64_t h = seed ^ (size * M);
  const uint8_t* data1 = (const uint8_t*)data;  // NOLINT
  const uint8_t* end = data1 + (size & ~7);
  uint64_t k;
  while (data1 != end) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    k = *(const uint64_t*)data1;
#else
    k = (uint64_t)data1[0];
    k |= (uint64_t)data1[1] << 8;
    k |= (uint64_t)data1[2] << 16;
    k |= (uint64_t)data1[3] << 24;
    k |= (uint64_t)data1[4] << 32;
    k |= (uint64_t)data1[5] << 40;
    k |= (uint64_t)data1[6] << 48;
    k |= (uint64_t)data1[7] << 56;
#endif
    data1 += sizeof(uint64_t);
    k *= M;
    k ^= k >> R;
    k *= M;
    h ^= k;
    h *= M;
  }

  switch (size & 7) {
    case 7:
      h ^= (uint64_t)data1[6] << 48;
      // fallthrough
    case 6:
      h ^= (uint64_t)data1[5] << 40;
      // fallthrough
    case 5:
      h ^= (uint64_t)data1[4] << 32;
      // fallthrough
    case 4:
      h ^= (uint64_t)data1[3] << 24;
      // fallthrough
    case 3:
      h ^= (uint64_t)data1[2] << 16;
      // fallthrough
    case 2:
      h ^= (uint64_t)data1[1] << 8;
      // fallthrough
    case 1:
      h ^= (uint64_t)data1[0];
      h *= M;
      break;
  }

  h ^= h >> R;
  h *= M;
  h ^= h >> R;
  return h;
}

}  // namespace deepx_core
