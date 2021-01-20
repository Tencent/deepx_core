// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/freq_store.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>

namespace deepx_core {

class FreqStoreTest : public testing::Test, public DataType {
 protected:
  const freq_t FREQ_THRESHOLD = 5;
  const freq_t LO_FREQ = 1;
  const freq_t HI_FREQ = 9;
};

TEST_F(FreqStoreTest, GetIdFreqMap) {
  Instance inst;
  inst.insert<csr_t>("X1") = csr_t{{0, 1, 2, 3, 4}, {1, 2, 3, 4}, {1, 1, 1, 1}};
  inst.insert<tsri_t>("X2") = tsri_t{{2, 3}, {4, 5}};

  id_freq_map_t id_freq_map;
  FreqStore::GetIdFreqMap(inst, &id_freq_map);

  id_freq_map_t expected_id_freq_map{{1, 1}, {2, 2}, {3, 2}, {4, 2}, {5, 1}};
  EXPECT_EQ(id_freq_map, expected_id_freq_map);
}

TEST_F(FreqStoreTest, Filter) {
  TensorMap param;

  FreqStore freq_store;
  freq_store.set_freq_filter_threshold(FREQ_THRESHOLD);
  freq_store.Init(&param);

  {
    PullRequest pull_request;
    pull_request.srm_map["W"] = {1, 2, 3, 4};
    pull_request.id_freq_map = {
        {1, LO_FREQ}, {2, LO_FREQ}, {3, HI_FREQ}, {4, HI_FREQ}};

    freq_store.Filter(&pull_request);
    id_set_t expected_id_set{3, 4};
    EXPECT_EQ(pull_request.srm_map["W"], expected_id_set);

    // 1: LO_FREQ
    // 2: LO_FREQ
    // 3: HI_FREQ
    // 4: HI_FREQ
  }

  {
    TensorMap grad;
    grad.insert<srm_t>("W") =
        srm_t{{1, 2, 3, 4}, {{1, 1}, {2, 2}, {3, 3}, {4, 4}}};

    freq_store.Filter(&grad);
    srm_t expected_gW{{3, 4}, {{3, 3}, {4, 4}}};
    EXPECT_EQ(grad.get<srm_t>("W"), expected_gW);
  }
}

}  // namespace deepx_core
