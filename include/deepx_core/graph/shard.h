// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#pragma once
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* Shard */
/************************************************************************/
class Shard : public DataType {
 private:
  int shard_id_;
  int shard_size_;
  tsr_shard_func_t tsr_shard_func_;
  srm_shard_func_t srm_shard_func_;

 public:
  void set_shard_id(int shard_id) noexcept { shard_id_ = shard_id; }
  int shard_id() const noexcept { return shard_id_; }
  void set_shard_size(int shard_size) noexcept { shard_size_ = shard_size; }
  int shard_size() const noexcept { return shard_size_; }
  void set_tsr_shard_func(const tsr_shard_func_t& tsr_shard_func) {
    tsr_shard_func_ = tsr_shard_func;
  }
  const tsr_shard_func_t& tsr_shard_func() const noexcept {
    return tsr_shard_func_;
  }
  void set_srm_shard_func(const srm_shard_func_t& srm_shard_func) {
    srm_shard_func_ = srm_shard_func;
  }
  const srm_shard_func_t& srm_shard_func() const noexcept {
    return srm_shard_func_;
  }

 private:
  static int DefaultTSRShardFunc(const std::string& name,
                                 int shard_size) noexcept;
  static int DefaultSRMShardFunc(int_t id, int shard_size) noexcept;

 public:
  Shard();
  Shard(int shard_id, int shard_size,
        const tsr_shard_func_t& tsr_shard_func = nullptr,
        const srm_shard_func_t& srm_shard_func = nullptr);

 public:
  bool HasTSR(const std::string& name) const noexcept {
    return tsr_shard_func_(name, shard_size_) == shard_id_;
  }
  bool HasSRM(int_t id) const noexcept {
    return srm_shard_func_(id, shard_size_) == shard_id_;
  }

 public:
  void SplitPullRequest(const PullRequest& full_pull_request,
                        std::vector<PullRequest>* pull_requests,
                        std::vector<id_set_t*>* aux) const;
  void SplitGrad(const TensorMap& param, TensorMap* full_grad,
                 std::vector<std::unique_ptr<TensorMap>>* grads,
                 std::vector<srm_t*>* aux) const;
  void SplitParam(const TensorMap& full_param,
                  std::vector<std::unique_ptr<TensorMap>>* params,
                  std::vector<srm_t*>* aux) const;
};

}  // namespace deepx_core
