// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* Model */
/************************************************************************/
class Model : public DataType {
 public:
  using tsr_partitioner_t =
      std::function<int(const std::string& name, int shard_size)>;
  using srm_partitioner_t =
      std::function<int(int_t feature_id, int shard_size)>;

 private:
  static int DefaultTSRPartitioner(const std::string& name,
                                   int shard_size) noexcept;
  static int DefaultSRMPartitioner(int_t feature_id, int shard_size) noexcept;

 private:
  tsr_partitioner_t tsr_partitioner_;
  srm_partitioner_t srm_partitioner_;
  const Graph* graph_ = nullptr;
  TensorMap param_;
  int use_lock_ = 0;
  AnyMap param_lock_;

 public:
  void set_tsr_partitioner(const tsr_partitioner_t& tsr_partitioner) {
    tsr_partitioner_ = tsr_partitioner;
  }
  const tsr_partitioner_t& tsr_partitioner() const noexcept {
    return tsr_partitioner_;
  }
  void set_srm_partitioner(const srm_partitioner_t& srm_partitioner) {
    srm_partitioner_ = srm_partitioner;
  }
  const srm_partitioner_t& srm_partitioner() const noexcept {
    return srm_partitioner_;
  }
  const Graph& graph() const noexcept { return *graph_; }
  TensorMap* mutable_param() noexcept { return &param_; }
  const TensorMap& param() const noexcept { return param_; }
  AnyMap* mutable_param_lock() noexcept { return &param_lock_; }

 public:
  Model();
  void Init(const Graph* graph) noexcept;
  bool InitParamPlaceholder();
  bool InitParam(std::default_random_engine& engine);  // NOLINT
  void InitLock();
  bool Write(OutputStream& os) const;  // NOLINT
  bool Read(InputStream& is);          // NOLINT
  bool Save(const std::string& file) const;
  bool Load(const std::string& file);
  bool SaveText(const std::string& file) const;
  void Merge(Model* other, int other_shard_id, int other_shard_size);
  void Warmup(Model* other);

 public:
  bool HasSRM() const noexcept;
  void RemoveZerosSRM();
  void ForEachSRM(const std::function<void(const std::string&, srm_t*)>& func);
  void SplitPullRequest(const PullRequest& full_pull_request,
                        std::vector<PullRequest>* pull_requests,
                        std::vector<id_set_t*>* aux) const;
  // thread safe after 'InitLock'
  void Pull(std::default_random_engine& engine,  // NOLINT
            const PullRequest& pull_request, TensorMap* remote_param);
  void SetParam(std::vector<std::unique_ptr<TensorMap>>* remote_params);
  void SplitGrad(const TensorMap& param, TensorMap* full_grad,
                 std::vector<std::unique_ptr<TensorMap>>* grads,
                 std::vector<srm_t*>* aux) const;
  void SplitParam(const TensorMap& full_param,
                  std::vector<std::unique_ptr<TensorMap>>* params,
                  std::vector<srm_t*>* aux) const;
  // thread safe after 'InitLock'
  void Update(TensorMap* param);

 private:
  template <class ReduceTSR, class ReduceSRM>
  void Reduce(TensorMap* param, ReduceTSR&& reduce_tsr,
              ReduceSRM&& reduce_srm) {
    for (auto& entry : *param) {
      const std::string& name = entry.first;
      auto it = param_.find(name);
      if (it == param_.end()) {
        continue;
      }

      Any& local_Wany = it->second;
      Any& remote_Wany = entry.second;
      if (local_Wany.is<tsr_t>() && remote_Wany.is<tsr_t>()) {
        auto& local_W = local_Wany.unsafe_to_ref<tsr_t>();
        auto& remote_W = remote_Wany.unsafe_to_ref<tsr_t>();
        if (local_W.same_shape(remote_W)) {
          reduce_tsr(name, local_W, remote_W);
        }
      } else if (local_Wany.is<srm_t>() && remote_Wany.is<srm_t>()) {
        auto& local_W = local_Wany.unsafe_to_ref<srm_t>();
        auto& remote_W = remote_Wany.unsafe_to_ref<srm_t>();
        if (local_W.col() == remote_W.col()) {
          reduce_srm(name, local_W, remote_W);
        }
      }
    }
  }

  template <class ReduceTSR, class ReduceSRM>
  void Reduce(Model* other, ReduceTSR&& reduce_tsr, ReduceSRM&& reduce_srm) {
    Reduce(&other->param_, reduce_tsr, reduce_srm);
  }
};

}  // namespace deepx_core
