// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/common/hash.h>
#include <deepx_core/common/read_write_lock.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/model.h>
#include <cstdint>
#include <fstream>
#include <utility>

namespace deepx_core {

int Model::DefaultTSRPartitioner(const std::string& name,
                                 int shard_size) noexcept {
  return (int)((uint32_t)MurmurHash2(name) % shard_size);
}

int Model::DefaultSRMPartitioner(int_t feature_id, int shard_size) noexcept {
  return (int)((uint32_t)feature_id % shard_size);
}

Model::Model()
    : tsr_partitioner_(DefaultTSRPartitioner),
      srm_partitioner_(DefaultSRMPartitioner) {}

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

bool Model::InitParam(std::default_random_engine& engine) {
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
        DXINFO("Initializing TSR %s...", node->name().c_str());
        auto& W = param_.insert<tsr_t>(node->name());
        W.resize(node->shape());
        W.rand_init(engine, node->initializer_type(),
                    (float_t)node->initializer_param1(),
                    (float_t)node->initializer_param2());
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

    auto it = param_.find(node->name());
    if (it == param_.end()) {
      DXERROR("%s is missing.", node->name().c_str());
      return false;
    }

    const Any& Wany = it->second;
    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR: {
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
      } break;
      case TENSOR_TYPE_SRM: {
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

void Model::Merge(Model* other, int other_shard_id, int other_shard_size) {
  DXINFO("Merging model %d/%d...", other_shard_id, other_shard_size);
  auto reduce_tsr = [other, other_shard_id, other_shard_size](
                        const std::string& name, tsr_t& local_W,
                        tsr_t& remote_W) {
    if (other->tsr_partitioner_(name, other_shard_size) == other_shard_id) {
      DXINFO("Merging TSR %s...", name.c_str());
      local_W = std::move(remote_W);
    }
  };
  auto reduce_srm = [](const std::string& name, srm_t& local_W,
                       srm_t& remote_W) {
    DXINFO("Merging SRM %s...", name.c_str());
    local_W.merge(std::move(remote_W));
  };
  Reduce(other, reduce_tsr, reduce_srm);
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

void Model::SplitPullRequest(const PullRequest& full_pull_request,
                             std::vector<PullRequest>* pull_requests,
                             std::vector<PullRequest::id_set_t*>* aux) const {
  int shard_size = (int)pull_requests->size();
  for (PullRequest& pull_request : *pull_requests) {
    pull_request.clear();
    pull_request.is_train = full_pull_request.is_train;
  }

  for (const std::string& name : full_pull_request.tsr_set) {
    int shard_id = tsr_partitioner_(name, shard_size);
    (*pull_requests)[shard_id].tsr_set.emplace(name);
  }

  size_t srm_feature_size = full_pull_request.srm_map.size() / shard_size;
  for (const auto& entry : full_pull_request.srm_map) {
    const std::string& name = entry.first;
    const PullRequest::id_set_t& feature_id_set = entry.second;
    for (int i = 0; i < shard_size; ++i) {
      (*aux)[i] = &(*pull_requests)[i].srm_map[name];
      (*aux)[i]->reserve(srm_feature_size);
    }
    for (int_t feature_id : feature_id_set) {
      int shard_id = srm_partitioner_(feature_id, shard_size);
      (*aux)[shard_id]->emplace(feature_id);
    }
  }

  for (const auto& entry : full_pull_request.id_freq_map) {
    int_t feature_id = entry.first;
    PullRequest::freq_t freq = entry.second;
    int shard_id = srm_partitioner_(feature_id, shard_size);
    (*pull_requests)[shard_id].id_freq_map.emplace(feature_id, freq);
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
    const PullRequest::id_set_t& feature_id_set = entry.second;
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

void Model::SplitGrad(const TensorMap& param, TensorMap* full_grad,
                      std::vector<std::unique_ptr<TensorMap>>* grads,
                      std::vector<srm_t*>* aux) const {
  int shard_size = (int)grads->size();
  for (auto& grad : *grads) {
    grad->ClearValue();
  }

  for (auto& entry : *full_grad) {
    const std::string& name = entry.first;
    auto it = param.find(name);
    if (it == param.end()) {
      continue;
    }

    const Any& Wany = it->second;
    Any& Gany = entry.second;
    if (Wany.is<tsr_t>()) {
      int shard_id = tsr_partitioner_(name, shard_size);
      if (Gany.is<tsr_t>()) {
        auto& G = Gany.unsafe_to_ref<tsr_t>();
        // view, zero-copy
        (*grads)[shard_id]->get_or_insert<tsr_t>(name) = G.get_view();
      } else if (Gany.is<srm_t>()) {
        auto& G = Gany.unsafe_to_ref<srm_t>();
        int col = G.col();
        (*grads)[shard_id]->get_or_insert<srm_t>(name) = std::move(G);
        G.clear();
        G.set_col(col);
      }
    } else if (Wany.is<srm_t>()) {
      if (Gany.is<srm_t>()) {
        auto& G = Gany.unsafe_to_ref<srm_t>();
        size_t srm_feature_size = G.size() / shard_size;
        for (int i = 0; i < shard_size; ++i) {
          (*aux)[i] = &(*grads)[i]->get_or_insert<srm_t>(name);
          (*aux)[i]->set_col(G.col());
          (*aux)[i]->reserve(srm_feature_size);
        }
        for (const auto& _entry : G) {
          int_t feature_id = _entry.first;
          const float_t* feature_embedding = _entry.second;
          int shard_id = srm_partitioner_(feature_id, shard_size);
          // view, zero-copy
          (*aux)[shard_id]->assign_view(feature_id, feature_embedding);
        }
      }
    }
  }
}

void Model::SplitParam(const TensorMap& full_param,
                       std::vector<std::unique_ptr<TensorMap>>* params,
                       std::vector<srm_t*>* aux) const {
  int shard_size = (int)params->size();
  for (auto& param : *params) {
    param->ClearValue();
  }

  for (const auto& entry : full_param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      int shard_id = tsr_partitioner_(name, shard_size);
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      // view, zero-copy
      (*params)[shard_id]->get_or_insert<tsr_t>(name) = W.get_view();
    } else if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      size_t srm_feature_size = W.size() / shard_size;
      for (int i = 0; i < shard_size; ++i) {
        (*aux)[i] = &(*params)[i]->get_or_insert<srm_t>(name);
        (*aux)[i]->set_col(W.col());
        (*aux)[i]->reserve(srm_feature_size);
      }
      for (const auto& _entry : W) {
        int_t feature_id = _entry.first;
        const float_t* feature_embedding = _entry.second;
        int shard_id = srm_partitioner_(feature_id, shard_size);
        // view, zero-copy
        (*aux)[shard_id]->assign_view(feature_id, feature_embedding);
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
