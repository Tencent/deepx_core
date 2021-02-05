// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/ts_store.h>
#include <cstdint>
#include <unordered_map>

namespace deepx_core {

void TSStore::Init(const TensorMap* param) noexcept { param_ = param; }

bool TSStore::InitParam() {
  DXINFO("Initializing TSStore...");
  id_ts_map_.clear();
  for (const auto& entry : *param_) {
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      for (const auto& _entry : W) {
        id_ts_map_[_entry.first] = now_;
      }
    }
  }
  DXINFO("TSStore has %zu entries.", id_ts_map_.size());
  return true;
}

void TSStore::InitLock() {
  use_lock_ = 1;
  id_ts_map_lock_.reset(new std::mutex);
}

bool TSStore::WriteLegacy(OutputStream& os) const {
  int version = 2;
  os << version;
  os << id_ts_map_;
  if (!os) {
    DXERROR("Failed to write TSStore.");
    return false;
  }
  return true;
}

bool TSStore::Write(OutputStream& os) const {
  int version = 0;
  os << version;
  os << id_ts_map_;
  if (!os) {
    DXERROR("Failed to write TSStore.");
    return false;
  }
  return true;
}

bool TSStore::ReadLegacy(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read TSStore.");
    return false;
  }

  if (version == 0) {
    std::unordered_map<uint16_t, id_ts_map_t> node_2_id_ts_map;
    is >> node_2_id_ts_map;
    if (!is) {
      DXERROR("Failed to read TSStore.");
      return false;
    }
    id_ts_map_.clear();
    for (const auto& entry : node_2_id_ts_map) {
      const id_ts_map_t& id_ts_map = entry.second;
      for (const auto& _entry : id_ts_map) {
        ts_t& ts = id_ts_map_[_entry.first];
        if (ts < _entry.second) {
          ts = _entry.second;
        }
      }
    }
  } else if (version == 1) {
    std::unordered_map<std::string, id_ts_map_t> name_2_id_ts_map;
    is >> name_2_id_ts_map;
    if (!is) {
      DXERROR("Failed to read TSStore.");
      return false;
    }
    id_ts_map_.clear();
    for (const auto& entry : name_2_id_ts_map) {
      const id_ts_map_t& id_ts_map = entry.second;
      for (const auto& _entry : id_ts_map) {
        ts_t& ts = id_ts_map_[_entry.first];
        if (ts < _entry.second) {
          ts = _entry.second;
        }
      }
    }
  } else if (version == 2) {
    is >> id_ts_map_;
    if (!is) {
      DXERROR("Failed to read TSStore.");
      return false;
    }
  } else {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }
  return true;
}

bool TSStore::Read(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read TSStore.");
    return false;
  }

  if (version > 0) {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }

  is >> id_ts_map_;
  if (!is) {
    DXERROR("Failed to read TSStore.");
    return false;
  }
  return true;
}

bool TSStore::SaveLegacy(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving TSStore to %s...", file.c_str());
  if (!WriteLegacy(os)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool TSStore::Save(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving TSStore to %s...", file.c_str());
  if (!Write(os)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool TSStore::LoadLegacy(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading TSStore from %s...", file.c_str());
  if (!ReadLegacy(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool TSStore::Load(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading TSStore from %s...", file.c_str());
  if (!Read(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

void TSStore::Merge(TSStore* other, const Shard* shard, int shard_id) {
  DXINFO("Merging TSStore...");
  size_t prev_size = id_ts_map_.size();
  id_ts_map_.reserve(id_ts_map_.size() + other->id_ts_map_.size());
  for (const auto& entry : other->id_ts_map_) {
    if (shard == nullptr || shard->HasSRM(shard_id, entry.first)) {
      id_ts_map_.emplace(entry);
    }
  }
  DXINFO("TSStore has merged %zu entries.", id_ts_map_.size() - prev_size);
}

void TSStore::Update(TensorMap* grad) {
  for (const auto& entry : *grad) {
    const std::string& name = entry.first;
    auto it = param_->find(name);
    if (it == param_->end()) {
      continue;
    }

    const Any& Wany = it->second;
    const Any& Gany = entry.second;
    if (Wany.is<srm_t>() && Gany.is<srm_t>()) {
      const auto& G = Gany.unsafe_to_ref<srm_t>();
      if (use_lock_) {
        std::unique_lock<std::mutex> guard(*id_ts_map_lock_);
        for (const auto& _entry : G) {
          id_ts_map_[_entry.first] = now_;
        }
      } else {
        for (const auto& _entry : G) {
          id_ts_map_[_entry.first] = now_;
        }
      }
    }
  }
}

auto TSStore::Expire() -> id_set_t {
  id_set_t expired;
  if (expire_threshold_ > 0) {
    auto first = id_ts_map_.begin();
    auto last = id_ts_map_.end();
    for (; first != last;) {
      if (now_ > expire_threshold_ + first->second) {
        expired.emplace(first->first);
        first = id_ts_map_.erase(first);
      } else {
        ++first;
      }
    }
    DXINFO("TSStore has %zu entries expired, %zu entries remained.",
           expired.size(), id_ts_map_.size());
  }
  return expired;
}

}  // namespace deepx_core
