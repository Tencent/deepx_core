// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/ts_store.h>

namespace deepx_core {

void TSStore::Init(const Graph* graph) noexcept { graph_ = graph; }

bool TSStore::InitParam(const TensorMap& param) {
  DXINFO("Initializing TSStore...");
  id_ts_map_.clear();
  for (const auto& entry : param) {
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

void TSStore::Merge(TSStore* other) {
  DXINFO("Merging TSStore...");
  size_t merged = 0;
  id_ts_map_.reserve(id_ts_map_.size() + other->id_ts_map_.size());
  for (auto& entry : other->id_ts_map_) {
    ts_t& ts = id_ts_map_[entry.first];
    if (ts < entry.second) {
      ts = entry.second;
      ++merged;
    }
  }
  DXINFO("TSStore has merged %zu entries.", merged);
}

void TSStore::MergeIf(
    TSStore* other,
    const std::function<bool(const id_ts_map_t::value_type&)>& func) {
  DXINFO("Merging TSStore...");
  size_t merged = 0;
  id_ts_map_.reserve(id_ts_map_.size() + other->id_ts_map_.size());
  for (auto& entry : other->id_ts_map_) {
    if (func(entry)) {
      ts_t& ts = id_ts_map_[entry.first];
      if (ts < entry.second) {
        ts = entry.second;
        ++merged;
      }
    }
  }
  DXINFO("TSStore has merged %zu entries.", merged);
}

void TSStore::Warmup(TSStore* other) {
  DXINFO("Warming up TSStore...");
  id_ts_map_.reserve(id_ts_map_.size() + other->id_ts_map_.size());
  for (auto& entry : other->id_ts_map_) {
    id_ts_map_.emplace(entry);
  }
  DXINFO("Done.");
}

void TSStore::Update(TensorMap* grad) {
  for (const auto& entry : *grad) {
    const std::string& name = entry.first;
    const GraphNode* node = graph_->find_node(name);
    DXASSERT(node);
    DXASSERT(node->node_type() == GRAPH_NODE_TYPE_PARAM);
    const Any& Gany = entry.second;
    if (Gany.is<srm_t>() && node->tensor_type() == TENSOR_TYPE_SRM) {
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
