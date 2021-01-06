// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/shard.h>
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
 private:
  const Graph* graph_ = nullptr;
  TensorMap param_;
  int use_lock_ = 0;
  AnyMap param_lock_;

 public:
  const Graph& graph() const noexcept { return *graph_; }
  TensorMap* mutable_param() noexcept { return &param_; }
  const TensorMap& param() const noexcept { return param_; }
  AnyMap* mutable_param_lock() noexcept { return &param_lock_; }

 public:
  void Init(const Graph* graph) noexcept;
  bool InitParamPlaceholder();
  bool InitParam(std::default_random_engine& engine,  // NOLINT
                 const Shard* shard = nullptr);
  void InitLock();
  bool Write(OutputStream& os) const;  // NOLINT
  bool Read(InputStream& is);          // NOLINT
  bool Save(const std::string& file) const;
  bool Load(const std::string& file);
  bool SaveText(const std::string& file) const;
  void Merge(Model* other, const Shard* shard = nullptr);
  void Warmup(Model* other);

 public:
  bool HasSRM() const noexcept;
  void RemoveZerosSRM();
  void ForEachSRM(const std::function<void(const std::string&, srm_t*)>& func);
  // thread safe after 'InitLock'
  void Pull(std::default_random_engine& engine,  // NOLINT
            const PullRequest& pull_request, TensorMap* remote_param);
  void SetParam(std::vector<std::unique_ptr<TensorMap>>* remote_params);
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
