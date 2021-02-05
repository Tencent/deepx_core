// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/shard.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <mutex>
#include <string>

namespace deepx_core {

/************************************************************************/
/* TSStore */
/************************************************************************/
class TSStore : public DataType {
 private:
  ts_t now_ = 0;
  ts_t expire_threshold_ = 0;
  const TensorMap* param_ = nullptr;
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
  const TensorMap& param() const noexcept { return *param_; }

 public:
  void Init(const TensorMap* param) noexcept;
  bool InitParam();
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
  void Merge(TSStore* other, const Shard* shard = nullptr, int shard_id = 0);

 public:
  // thread safe after 'InitLock'
  void Update(TensorMap* grad);
  id_set_t Expire();
};

}  // namespace deepx_core
