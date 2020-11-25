/** Copyright 2019 the deepx authors.
 * Author: Yafei Zhang (kimmyzhang@tencent.com)
 *
 * hash feature functions
 */

#ifndef HASH_FEATURE_H_
#define HASH_FEATURE_H_

#include <stdint.h>

#if defined __cplusplus
extern "C" {
#endif

uint16_t HashFeature_GetGroupId(uint64_t feature_id);
uint64_t HashFeature_GetSubFeatureId(uint64_t feature_id);
uint64_t HashFeature_MakeFeatureId(uint16_t group_id, uint64_t sub_feature_id);
uint64_t HashFeature_S(uint16_t group_id, const char* feature, int len);
uint64_t HashFeature_64(uint16_t group_id, uint64_t a);
uint64_t HashFeature_642(uint16_t group_id, uint64_t a, uint64_t b);
uint64_t HashFeature_643(uint16_t group_id, uint64_t a, uint64_t b, uint64_t c);

#if defined __cplusplus
}
#endif

#endif /* HASH_FEATURE_H_ */
