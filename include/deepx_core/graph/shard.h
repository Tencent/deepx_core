// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/tensor/data_type.h>
#include <string>

namespace deepx_core {

class Shard;

/************************************************************************/
/* Shard functions */
/************************************************************************/
bool SaveShard(const std::string& dir, const Shard& shard);
// backward compatibility
bool LoadShardLegacy(const std::string& dir, Shard* shard);
bool LoadShard(const std::string& dir, Shard* shard);

/************************************************************************/
/* Shard */
/************************************************************************/
class Shard : public DataType {
 private:
  // 0, non-shard mode
  // 1, shard mode
  int shard_mode_ = 0;
  int shard_size_ = 0;
  std::string shard_func_name_;
  tsr_shard_func_t tsr_shard_func_;
  srm_shard_func_t srm_shard_func_;

 public:
  int shard_mode() const noexcept { return shard_mode_; }
  int shard_size() const noexcept { return shard_size_; }
  const std::string& shard_func_name() const noexcept {
    return shard_func_name_;
  }
  const tsr_shard_func_t& tsr_shard_func() const noexcept {
    return tsr_shard_func_;
  }
  const srm_shard_func_t& srm_shard_func() const noexcept {
    return srm_shard_func_;
  }

 public:
  static void RegisterShardFunc(const std::string& shard_func_name,
                                const tsr_shard_func_t& tsr_shard_func,
                                const srm_shard_func_t& srm_shard_func);

 private:
  void _Init(int shard_mode, int shard_size,
             const std::string& shard_func_name);

 public:
  void InitNonShard();
  void InitShard(int shard_size, const std::string& shard_func_name);
  bool Write(OutputStream& os) const;  // NOLINT
  bool Read(InputStream& is);          // NOLINT
  bool Save(const std::string& file) const;
  bool Load(const std::string& file);

 public:
  int GetTSRShardId(const std::string& name) const noexcept {
    return tsr_shard_func_(name, shard_size_);
  }
  int GetSRMShardId(int_t id) const noexcept {
    return srm_shard_func_(id, shard_size_);
  }
  bool HasTSR(int shard_id, const std::string& name) const noexcept {
    return tsr_shard_func_(name, shard_size_) == shard_id;
  }
  bool HasSRM(int shard_id, int_t id) const noexcept {
    return srm_shard_func_(id, shard_size_) == shard_id;
  }
};

}  // namespace deepx_core
