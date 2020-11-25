// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace deepx_core {

/************************************************************************/
/* TSStore */
/************************************************************************/
class TSStore : public DataType {
 public:
  using ts_t = uint32_t;
  using id_ts_map_t = std::unordered_map<int_t, ts_t>;
  using id_set_t = std::unordered_set<int_t>;

 private:
  ts_t now_ = 0;
  ts_t expire_threshold_ = 0;
  const Graph* graph_ = nullptr;
  id_ts_map_t id_ts_map_;
  int use_lock_ = 0;
  std::unique_ptr<std::mutex> id_ts_map_lock_;

 public:
  void set_now(ts_t now) noexcept { now_ = now; }
  ts_t now() const noexcept { return now_; }
  void set_expire_threshold(ts_t expire_threshold) noexcept {
    expire_threshold_ = expire_threshold;
  }
  ts_t expire_threshold() const noexcept { return expire_threshold_; }
  const Graph& graph() const noexcept { return *graph_; }

 public:
  void Init(const Graph* graph) noexcept;
  bool InitParam(const TensorMap& param);
  void InitLock();
  bool Write(OutputStream& os) const;  // NOLINT
  bool Read(InputStream& is);          // NOLINT
  bool Save(const std::string& file) const;
  bool Load(const std::string& file);
  void Warmup(TSStore* other);

 public:
  // thread safe after 'InitLock'
  void Update(TensorMap* grad);
  id_set_t Expire();

 private:
  void DumpMeta() const noexcept;
};

}  // namespace deepx_core
