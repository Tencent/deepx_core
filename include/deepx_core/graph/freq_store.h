// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Shuting Guo (tinkle@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/read_write_lock.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/shard.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <functional>
#include <memory>
#include <string>

namespace deepx_core {

/************************************************************************/
/* FreqStore */
/************************************************************************/
class FreqStore : public DataType {
 private:
  freq_t freq_filter_threshold_ = 0;
  id_freq_map_t id_freq_map_;
  const TensorMap* param_ = nullptr;
  int use_lock_ = 0;
  std::unique_ptr<ReadWriteLock> id_freq_map_lock_;

 public:
  void set_freq_filter_threshold(freq_t freq_filter_threshold) noexcept {
    freq_filter_threshold_ = freq_filter_threshold;
  }
  freq_t freq_filter_threshold() const noexcept {
    return freq_filter_threshold_;
  }
  const TensorMap& param() const noexcept { return *param_; }

 public:
  void Init(const TensorMap* param) noexcept;
  bool InitParam();
  void InitLock();
  bool Write(OutputStream& os) const;  // NOLINT
  bool Read(InputStream& is);          // NOLINT
  bool Save(const std::string& file) const;
  bool Load(const std::string& file);
  void Merge(FreqStore* other, const Shard* shard = nullptr, int shard_id = 0);
  void RemoveIf(
      const std::function<bool(const id_freq_map_t::value_type&)>& func);

 public:
  static void GetIdFreqMap(const Instance& inst, id_freq_map_t* id_freq_map);

 public:
  // thread safe after 'InitLock'
  void Filter(PullRequest* pull_request);
  // thread safe after 'InitLock'
  void Filter(TensorMap* grad) const;

 private:
  void Filter_NoLock(PullRequest* pull_request);
  void Filter_Lock(PullRequest* pull_request);
  void Filter_NoLock(TensorMap* grad) const;
  void Filter_Lock(TensorMap* grad) const;
  bool Filter_NoLock(int_t id) const noexcept;
  bool Filter_Lock(int_t id) const noexcept;
};

}  // namespace deepx_core
