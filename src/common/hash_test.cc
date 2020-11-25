// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/hash.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace deepx_core {

class HashTest : public testing::Test {};

TEST_F(HashTest, MurmurHash3Mix) {
  const std::vector<std::pair<uint64_t, uint64_t>> KEY_HASH_PAIRS = {
      {UINT64_C(0), UINT64_C(0)},
      {UINT64_C(123), UINT64_C(0x7fcb4990c961f6a8)},
      {UINT64_C(1234567890), UINT64_C(0x5d32d678ccf274ad)}};
  for (const auto& entry : KEY_HASH_PAIRS) {
    EXPECT_EQ(MurmurHash3Mix(entry.first), entry.second);
  }
}

TEST_F(HashTest, MurmurHash2_std_string) {
  const std::vector<std::pair<std::string, uint64_t>> KEY_HASH_PAIRS = {
      {"", UINT64_C(0x553e93901e462a6e)},
      {"123", UINT64_C(0x8c039e9e9dfbadf3)},
      {"1234567890", UINT64_C(0x21404b2de0e52ba5)},
      {"abc", UINT64_C(0x32d82bf8ed3dba39)},
      {"abcdefghijklmn", UINT64_C(0x329f682efbe70b4)},
      {"qwertyuiop", UINT64_C(0x13c1ce1fa0f044c9)}};
  for (const auto& entry : KEY_HASH_PAIRS) {
    EXPECT_EQ(MurmurHash2(entry.first), entry.second);
  }
}

}  // namespace deepx_core
