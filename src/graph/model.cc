// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/common/read_write_lock.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/model.h>
#include <fstream>
#include <utility>

namespace deepx_core {

void Model::Init(const Graph* graph) noexcept { graph_ = graph; }

bool Model::InitParamPlaceholder() {
  // keep silent
  for (const auto& entry : graph_->name_2_node()) {
    const GraphNode* node = entry.second;
    if (node->node_type() != GRAPH_NODE_TYPE_PARAM) {
      continue;
    }

    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        auto& W = param_.insert<tsr_t>(node->name());
        W.resize(node->shape());
        // no rand_init
      } break;
      case TENSOR_TYPE_SRM: {
        auto& W = param_.insert<srm_t>(node->name());
        W.set_col(node->shape()[1]);
        W.set_initializer(node->initializer_type(),
                          (float_t)node->initializer_param1(),
                          (float_t)node->initializer_param2());
      } break;
    }
  }
  return true;
}

bool Model::InitParam(std::default_random_engine& engine, const Shard* shard) {
  DXINFO("Initializing param...");
  for (const auto& entry : graph_->name_2_node()) {
    const GraphNode* node = entry.second;
    if (node->node_type() != GRAPH_NODE_TYPE_PARAM) {
      continue;
    }

    auto it = param_.find(node->name());
    if (it != param_.end()) {
      continue;
    }

    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        if (!shard || shard->HasTSR(node->name())) {
          DXINFO("Initializing TSR %s...", node->name().c_str());
          auto& W = param_.insert<tsr_t>(node->name());
          W.resize(node->shape());
          W.rand_init(engine, node->initializer_type(),
                      (float_t)node->initializer_param1(),
                      (float_t)node->initializer_param2());
        }
      } break;
      case TENSOR_TYPE_SRM: {
        DXINFO("Initializing SRM %s...", node->name().c_str());
        auto& W = param_.insert<srm_t>(node->name());
        W.set_col(node->shape()[1]);
        W.set_initializer(node->initializer_type(),
                          (float_t)node->initializer_param1(),
                          (float_t)node->initializer_param2());
        int row = node->shape()[0];
        row = (row < 10000) ? 10000 : row;            // magic number
        row = (row > 1000000000) ? 1000000000 : row;  // magic number
        W.reserve(row);
      } break;
    }
  }
  DXINFO("Done.");

  DXINFO("Post-checking param...");
  for (const auto& entry : graph_->name_2_node()) {
    const GraphNode* node = entry.second;
    if (node->node_type() != GRAPH_NODE_TYPE_PARAM) {
      continue;
    }

    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        if (!shard || shard->HasTSR(node->name())) {
          auto it = param_.find(node->name());
          if (it == param_.end()) {
            DXERROR("%s is missing.", node->name().c_str());
            return false;
          }

          const Any& Wany = it->second;
          if (!Wany.is<tsr_t>()) {
            DXERROR("TSR %s has inconsistent type.", node->name().c_str());
            return false;
          }

          auto& W = Wany.unsafe_to_ref<tsr_t>();
          if (W.shape() != node->shape()) {
            DXERROR("TSR %s has inconsistent shape: %s vs %s.",
                    node->name().c_str(), to_string(node->shape()).c_str(),
                    to_string(W.shape()).c_str());
            return false;
          }
        }
      } break;
      case TENSOR_TYPE_SRM: {
        auto it = param_.find(node->name());
        if (it == param_.end()) {
          DXERROR("%s is missing.", node->name().c_str());
          return false;
        }

        const Any& Wany = it->second;
        if (!Wany.is<srm_t>()) {
          DXERROR("SRM %s has inconsistent type.", node->name().c_str());
          return false;
        }

        auto& W = Wany.unsafe_to_ref<srm_t>();
        if (W.col() != node->shape()[1]) {
          DXERROR("SRM %s has inconsistent col: %d vs %d.",
                  node->name().c_str(), W.col(), node->shape()[1]);
          return false;
        }
      } break;
    }
  }
  DXINFO("Done.");
  return true;
}

void Model::InitLock() {
  use_lock_ = 1;
  param_lock_.clear();
  for (const auto& entry : param_) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      std::shared_ptr<ReadWriteLock> lock(new ReadWriteLock);
      param_lock_[name].emplace(std::move(lock));
    }
  }
}

bool Model::Write(OutputStream& os) const {
  int version = 0;
  os << version;
  os << param_;
  if (!os) {
    DXERROR("Failed to write model.");
    return false;
  }
  return true;
}

bool Model::Read(InputStream& is) {
  int version;
  is >> version;
  if (!is) {
    DXERROR("Failed to read model.");
    return false;
  }

  if (version > 0) {
    DXERROR("Couldn't handle a higher version: %d.", version);
    is.set_bad();
    return false;
  }

  is >> param_;
  if (!is) {
    DXERROR("Failed to read model.");
    return false;
  }
  return true;
}

bool Model::Save(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving model to %s...", file.c_str());
  if (!Write(os)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool Model::Load(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading model from %s...", file.c_str());
  if (!Read(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool Model::SaveText(const std::string& file) const {
  // NOTE: only local file system is supported.
  std::ofstream os;
  os.open(file);
  if (!os.is_open()) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving model to %s...", file.c_str());
  os << param_;
  if (!os) {
    DXERROR("Failed to write text model.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

void Model::Merge(Model* other, const Shard* shard) {
  DXINFO("Merging model...");
  auto merge_tsr = [](const std::string& name, tsr_t& local_W,
                      tsr_t& remote_W) {
    DXINFO("Merging TSR %s...", name.c_str());
    local_W = std::move(remote_W);
  };
  auto merge_srm = [shard](const std::string& name, srm_t& local_W,
                           srm_t& remote_W) {
    DXINFO("Merging SRM %s...", name.c_str());
    local_W.merge_if(std::move(remote_W),
                     [shard](const srm_t::value_type& entry) {
                       return !shard || shard->HasSRM(entry.first);
                     });
  };

  for (auto& entry : other->param_) {
    const std::string& name = entry.first;
    Any& remote_Wany = entry.second;

    if (remote_Wany.is<tsr_t>()) {
      auto& remote_W = remote_Wany.unsafe_to_ref<tsr_t>();
      auto it = param_.find(name);
      if (it != param_.end()) {
        Any& local_Wany = it->second;
        if (!local_Wany.is<tsr_t>()) {
          continue;
        }
        auto& local_W = local_Wany.unsafe_to_ref<tsr_t>();
        merge_tsr(name, local_W, remote_W);
      } else if (!shard || shard->HasTSR(name)) {
        auto& local_W = param_.insert<tsr_t>(name);
        merge_tsr(name, local_W, remote_W);
      }
    } else if (remote_Wany.is<srm_t>()) {
      auto& remote_W = remote_Wany.unsafe_to_ref<srm_t>();
      auto it = param_.find(name);
      if (it != param_.end()) {
        Any& local_Wany = it->second;
        if (!local_Wany.is<srm_t>()) {
          continue;
        }
        auto& local_W = local_Wany.unsafe_to_ref<srm_t>();
        merge_srm(name, local_W, remote_W);
      } else {
        auto& local_W = param_.insert<srm_t>(name);
        local_W.set_col(remote_W.col());
        merge_srm(name, local_W, remote_W);
      }
    }
  }
  DXINFO("Done.");
}

void Model::Warmup(Model* other) {
  DXINFO("Warming up model...");
  auto reduce_tsr = [](const std::string& name, tsr_t& local_W,
                       tsr_t& remote_W) {
    DXINFO("Warming up TSR %s...", name.c_str());
    local_W = std::move(remote_W);
  };
  auto reduce_srm = [](const std::string& name, srm_t& local_W,
                       srm_t& remote_W) {
    DXINFO("Warming up SRM %s...", name.c_str());
    local_W.merge(std::move(remote_W));
  };
  Reduce(other, reduce_tsr, reduce_srm);
  DXINFO("Done.");
}

bool Model::HasSRM() const noexcept {
  for (const auto& entry : param_) {
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      return true;
    }
  }
  return false;
}

void Model::RemoveZerosSRM() {
  DXINFO("Removing zeros...");
  for (auto& entry : param_) {
    const std::string& name = entry.first;
    Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      size_t prev_size = W.size();
      W.remove_zeros();
      DXINFO("SRM %s has %zu entries removed, %zu entries remained.",
             name.c_str(), prev_size - W.size(), W.size());
    }
  }
  DXINFO("Done.");
}

void Model::ForEachSRM(
    const std::function<void(const std::string&, srm_t*)>& func) {
  for (auto& entry : param_) {
    const std::string& name = entry.first;
    Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      func(name, &W);
    }
  }
}

void Model::Pull(std::default_random_engine& engine,
                 const PullRequest& pull_request, TensorMap* remote_param) {
  remote_param->ClearValue();

  for (const std::string& name : pull_request.tsr_set) {
    const auto& local_W = param_.get<tsr_t>(name);
    auto& remote_W = remote_param->get_or_insert<tsr_t>(name);
    // view, zero-copy
    remote_W = local_W.get_view();
  }

  for (const auto& entry : pull_request.srm_map) {
    const std::string& name = entry.first;
    const id_set_t& feature_id_set = entry.second;
    auto& local_W = param_.get<srm_t>(name);
    const auto& const_local_W = (const srm_t&)local_W;
    auto& remote_W = remote_param->get_or_insert<srm_t>(name);
    remote_W.set_col(local_W.col());
    remote_W.reserve(feature_id_set.size());
    if (pull_request.is_train) {
      // get random values for missing keys
      if (use_lock_) {
        auto& lock =
            param_lock_.unsafe_get<std::shared_ptr<ReadWriteLock>>(name);
        for (int_t feature_id : feature_id_set) {
          const float_t* feature_embedding =
              local_W.get_row(engine, feature_id, lock.get());
          // view, zero-copy
          remote_W.assign_view(feature_id, feature_embedding);
        }
      } else {
        for (int_t feature_id : feature_id_set) {
          const float_t* feature_embedding =
              local_W.get_row(engine, feature_id);
          // view, zero-copy
          remote_W.assign_view(feature_id, feature_embedding);
        }
      }
    } else {
      // get nothing for missing keys
      for (int_t feature_id : feature_id_set) {
        const float_t* feature_embedding =
            const_local_W.get_row_no_init(feature_id);
        if (feature_embedding) {
          // view, zero-copy
          remote_W.assign_view(feature_id, feature_embedding);
        }
      }
    }
  }

  remote_param->RemoveEmptyValue();
}

void Model::SetParam(std::vector<std::unique_ptr<TensorMap>>* remote_params) {
  param_.ClearSRMValue();

  for (auto& remote_param : *remote_params) {
    for (auto& entry : *remote_param) {
      const std::string& name = entry.first;
      Any& Wany = entry.second;
      if (Wany.is<tsr_t>()) {
        auto& local_W = param_.get<tsr_t>(name);
        auto& remote_W = Wany.unsafe_to_ref<tsr_t>();
        // copy, not view
        local_W.set_data(remote_W);
      } else if (Wany.is<srm_t>()) {
        auto& local_W = param_.get<srm_t>(name);
        auto& remote_W = Wany.unsafe_to_ref<srm_t>();
        local_W.merge(std::move(remote_W));
      }
    }
  }
}

void Model::Update(TensorMap* param) {
  auto reduce_tsr = [](const std::string& /*name*/, tsr_t& local_W,
                       tsr_t& remote_W) { local_W.set_data(remote_W); };
  auto reduce_srm = [this](const std::string& name, srm_t& local_W,
                           srm_t& remote_W) {
    if (use_lock_) {
      auto& lock = param_lock_.unsafe_get<std::shared_ptr<ReadWriteLock>>(name);
      local_W.upsert(remote_W, lock.get());
    } else {
      local_W.upsert(remote_W);
    }
  };
  Reduce(param, reduce_tsr, reduce_srm);
}

}  // namespace deepx_core
