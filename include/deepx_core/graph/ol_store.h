// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#pragma once
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <string>
#include <unordered_map>

namespace deepx_core {

/************************************************************************/
/* OLStore */
/************************************************************************/
class OLStore : public DataType {
 private:
  struct State {
    freq_t update = 0;
  };
  using srm_state_t = std::unordered_map<int_t, State>;
  using srm_state_map_t = std::unordered_map<std::string, srm_state_t>;

  freq_t update_threshold_ = 0;
  float_t distance_threshold_ = 0;

  const Graph* graph_ = nullptr;
  const TensorMap* param_ = nullptr;
  TensorMap prev_param_;
  srm_state_map_t srm_state_map_;

 public:
  void set_update_threshold(freq_t update_threshold) noexcept {
    update_threshold_ = update_threshold;
  }
  freq_t update_threshold() const noexcept { return update_threshold_; }
  void set_distance_threshold(float_t distance_threshold) noexcept {
    distance_threshold_ = distance_threshold;
  }
  float_t distance_threshold() const noexcept { return distance_threshold_; }
  const Graph& graph() const noexcept { return *graph_; }
  const TensorMap& param() const noexcept { return *param_; }

 public:
  void Init(const Graph* graph, const TensorMap* param) noexcept;
  bool InitParam();
  void Update(TensorMap* param);
  // for unit test
  id_set_t Collect();

 private:
  bool Collect(const State& state, int n, const float_t* embedding,
               const float_t* prev_embedding) const;

 public:
  bool SaveFeatureKVModel(const std::string& file,
                          int feature_kv_protocol_version);
};

}  // namespace deepx_core
