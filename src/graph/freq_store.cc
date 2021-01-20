// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/freq_store.h>
#include <limits>  // std::numeric_limits

namespace deepx_core {

void FreqStore::Init(const TensorMap* param) noexcept { param_ = param; }

bool FreqStore::InitParam() {
  DXINFO("Initializing FreqStore...");
  id_freq_map_.clear();
  for (const auto& entry : *param_) {
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      for (const auto& _entry : W) {
        id_freq_map_[_entry.first] = std::numeric_limits<freq_t>::max();
      }
    }
  }
  DXINFO("FreqStore has %zu entries.", id_freq_map_.size());
  return true;
}

void FreqStore::InitLock() {
  use_lock_ = 1;
  id_freq_map_lock_.reset(new ReadWriteLock);
}

bool FreqStore::Write(OutputStream& os) const {
  int version = 0;
  os << version;
  os << id_freq_map_;
  if (!os) {
    DXERROR("Failed to write FreqStore.");
    return false;
  }
  return true;
}

bool FreqStore::Read(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read FreqStore.");
    return false;
  }

  if (version > 0) {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }

  is >> id_freq_map_;
  if (!is) {
    DXERROR("Failed to read FreqStore.");
    return false;
  }
  return true;
}

bool FreqStore::Save(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving FreqStore to %s...", file.c_str());
  if (!Write(os)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool FreqStore::Load(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading FreqStore from %s...", file.c_str());
  if (!Read(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

void FreqStore::Merge(FreqStore* other, const Shard* shard, int shard_id) {
  DXINFO("Merging FreqStore...");
  size_t prev_size = id_freq_map_.size();
  id_freq_map_.reserve(id_freq_map_.size() + other->id_freq_map_.size());
  for (auto& entry : other->id_freq_map_) {
    if (shard == nullptr || shard->HasSRM(shard_id, entry.first)) {
      id_freq_map_.emplace(entry);
    }
  }
  DXINFO("FreqStore has merged %zu entries.", id_freq_map_.size() - prev_size);
}

void FreqStore::RemoveIf(
    const std::function<bool(const id_freq_map_t::value_type&)>& func) {
  DXINFO("Removing from FreqStore...");
  size_t prev_size = id_freq_map_.size();
  auto first = id_freq_map_.begin();
  auto last = id_freq_map_.end();
  for (; first != last;) {
    if (func(*first)) {
      first = id_freq_map_.erase(first);
    } else {
      ++first;
    }
  }
  DXINFO("FreqStore has %zu entries removed, %zu entries remained.",
         prev_size - id_freq_map_.size(), id_freq_map_.size());
}

void FreqStore::GetIdFreqMap(const Instance& inst, id_freq_map_t* id_freq_map) {
  id_freq_map->clear();
  for (const auto& entry : inst) {
    const Any& Xany = entry.second;
    if (Xany.is<csr_t>()) {
      auto& X = Xany.unsafe_to_ref<csr_t>();
      for (size_t i = 0; i < X.col_size(); ++i) {
        int_t id = X.col(i);
        // no overflow check
        ++(*id_freq_map)[id];
      }
    } else if (Xany.is<tsri_t>()) {
      auto& X = Xany.unsafe_to_ref<tsri_t>();
      for (int_t id : X) {
        // no overflow check
        ++(*id_freq_map)[id];
      }
    }
  }
}

void FreqStore::Filter(PullRequest* pull_request) {
  if (freq_filter_threshold_ == 0) {
    return;
  }

  if (use_lock_) {
    Filter_Lock(pull_request);
  } else {
    Filter_NoLock(pull_request);
  }
}

void FreqStore::Filter(TensorMap* grad) const {
  if (freq_filter_threshold_ == 0) {
    return;
  }

  if (use_lock_) {
    Filter_Lock(grad);
  } else {
    Filter_NoLock(grad);
  }
}

void FreqStore::Filter_NoLock(PullRequest* pull_request) {
  for (const auto& entry : pull_request->id_freq_map) {
    freq_t& freq = id_freq_map_[entry.first];
    if (freq > std::numeric_limits<freq_t>::max() - entry.second) {
      freq = std::numeric_limits<freq_t>::max();
    } else {
      freq += entry.second;
    }
  }

  for (auto& entry : pull_request->srm_map) {
    id_set_t& id_set = entry.second;
    auto first = id_set.begin();
    auto last = id_set.end();
    for (; first != last;) {
      if (Filter_NoLock(*first)) {
        first = id_set.erase(first);
      } else {
        ++first;
      }
    }
  }
}

void FreqStore::Filter_Lock(PullRequest* pull_request) {
  for (const auto& entry : pull_request->id_freq_map) {
    WriteLockGuard guard(id_freq_map_lock_.get());
    freq_t& freq = id_freq_map_[entry.first];
    if (freq > std::numeric_limits<freq_t>::max() - entry.second) {
      freq = std::numeric_limits<freq_t>::max();
    } else {
      freq += entry.second;
    }
  }

  for (auto& entry : pull_request->srm_map) {
    id_set_t& id_set = entry.second;
    auto first = id_set.begin();
    auto last = id_set.end();
    for (; first != last;) {
      if (Filter_Lock(*first)) {
        first = id_set.erase(first);
      } else {
        ++first;
      }
    }
  }
}

void FreqStore::Filter_NoLock(TensorMap* grad) const {
  for (auto& entry : *grad) {
    Any& Gany = entry.second;
    if (Gany.is<srm_t>()) {
      auto& G = Gany.unsafe_to_ref<srm_t>();
      G.remove_if([this](const srm_t::value_type& entry) {
        return Filter_NoLock(entry.first);
      });
    }
  }
}

void FreqStore::Filter_Lock(TensorMap* grad) const {
  for (auto& entry : *grad) {
    Any& Gany = entry.second;
    if (Gany.is<srm_t>()) {
      auto& G = Gany.unsafe_to_ref<srm_t>();
      G.remove_if([this](const srm_t::value_type& entry) {
        return Filter_Lock(entry.first);
      });
    }
  }
}

bool FreqStore::Filter_NoLock(int_t id) const noexcept {
  auto it = id_freq_map_.find(id);
  if (it != id_freq_map_.end()) {
    return it->second < freq_filter_threshold_;
  }
  return true;
}

bool FreqStore::Filter_Lock(int_t id) const noexcept {
  ReadLockGuard guard(id_freq_map_lock_.get());
  auto it = id_freq_map_.find(id);
  if (it != id_freq_map_.end()) {
    return it->second < freq_filter_threshold_;
  }
  return true;
}

}  // namespace deepx_core
