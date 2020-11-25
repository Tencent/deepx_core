/** Copyright 2019 the deepx authors.
 * Author: Yafei Zhang (kimmyzhang@tencent.com)
 */

#include "hash_feature.h"
#include "xxhash.h"

#define SEED 0
#define GROUP_SHIFT 48
#define GROUP_MASK UINT64_C(0xffff000000000000)
#define FEATURE_MASK UINT64_C(0x0000ffffffffffff)

uint16_t HashFeature_GetGroupId(uint64_t feature_id) {
  return (feature_id & GROUP_MASK) >> GROUP_SHIFT;
}

uint64_t HashFeature_GetSubFeatureId(uint64_t feature_id) {
  return feature_id & FEATURE_MASK;
}

uint64_t HashFeature_MakeFeatureId(uint16_t group_id, uint64_t sub_feature_id) {
  return ((uint64_t)group_id << GROUP_SHIFT) | (sub_feature_id & FEATURE_MASK);
}

uint64_t HashFeature_S(uint16_t group_id, const char* feature, int len) {
  uint64_t hash = XXH64(feature, len, SEED);
  return ((uint64_t)group_id << GROUP_SHIFT) | (hash & FEATURE_MASK);
}

uint64_t HashFeature_64(uint16_t group_id, uint64_t a) {
  uint64_t hash = XXH64(&a, sizeof(a), SEED);
  return ((uint64_t)group_id << GROUP_SHIFT) | (hash & FEATURE_MASK);
}

uint64_t HashFeature_642(uint16_t group_id, uint64_t a, uint64_t b) {
  uint64_t hash, d[2];
  d[0] = a;
  d[1] = b;
  hash = XXH64(&d, sizeof(d), SEED);
  return ((uint64_t)group_id << GROUP_SHIFT) | (hash & FEATURE_MASK);
}

uint64_t HashFeature_643(uint16_t group_id, uint64_t a, uint64_t b,
                         uint64_t c) {
  uint64_t hash, d[3];
  d[0] = a;
  d[1] = b;
  d[2] = c;
  hash = XXH64(&d, sizeof(d), SEED);
  return ((uint64_t)group_id << GROUP_SHIFT) | (hash & FEATURE_MASK);
}
