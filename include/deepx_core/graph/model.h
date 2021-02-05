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
                 const Shard* shard = nullptr, int shard_id = 0);
  void InitLock();
  // backward compatibility
  bool WriteLegacy(OutputStream& os) const;  // NOLINT
  bool Write(OutputStream& os) const;        // NOLINT
  // backward compatibility
  bool ReadLegacy(InputStream& is);  // NOLINT
  bool Read(InputStream& is);        // NOLINT
  // backward compatibility
  bool SaveLegacy(const std::string& file) const;
  bool Save(const std::string& file) const;
  // backward compatibility
  bool LoadLegacy(const std::string& file);
  bool Load(const std::string& file);
  bool SaveText(const std::string& file) const;
  bool SaveFeatureKV(const std::string& file,
                     int feature_kv_protocol_version) const;
  void Merge(Model* other, const Shard* shard = nullptr, int shard_id = 0);

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
  using tsr_reduce_func_t =
      std::function<void(const std::string&, tsr_t&, tsr_t&)>;
  using srm_reduce_func_t =
      std::function<void(const std::string&, srm_t&, srm_t&)>;
  void Reduce(TensorMap* param, const tsr_reduce_func_t& tsr_reduce_func,
              const srm_reduce_func_t& srm_reduce_func,
              const Shard* shard = nullptr, int shard_id = 0);
  void Reduce(Model* other, const tsr_reduce_func_t& tsr_reduce_func,
              const srm_reduce_func_t& srm_reduce_func,
              const Shard* shard = nullptr, int shard_id = 0);
};

}  // namespace deepx_core
