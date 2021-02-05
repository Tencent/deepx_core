// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/common/read_write_lock.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/feature_kv_util.h>
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
        // no reserve
      } break;
    }
  }
  return true;
}

bool Model::InitParam(std::default_random_engine& engine, const Shard* shard,
                      int shard_id) {
  DXINFO("Initializing model...");
  for (const auto& entry : graph_->name_2_node()) {
    const GraphNode* node = entry.second;
    if (node->node_type() != GRAPH_NODE_TYPE_PARAM) {
      continue;
    }

    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        if (shard == nullptr || shard->HasTSR(shard_id, node->name())) {
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

bool Model::WriteLegacy(OutputStream& os) const {
  int version = 0;
  os << version;
  os << DATA_TYPE_TOKEN;
  os << param_;
  if (!os) {
    DXERROR("Failed to write model.");
    return false;
  }
  return true;
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

bool Model::ReadLegacy(InputStream& is) {
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

  int data_type_token;
  is >> data_type_token;
  if (!is) {
    DXERROR("Failed to read model.");
    return false;
  }

  if (data_type_token != DATA_TYPE_TOKEN) {
    DXERROR("Inconsistent data type token: %d vs %d.", data_type_token,
            DATA_TYPE_TOKEN);
    return false;
  }

  is >> param_;
  if (!is) {
    DXERROR("Failed to read model.");
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

bool Model::SaveLegacy(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving model to %s...", file.c_str());
  if (!WriteLegacy(os)) {
    return false;
  }
  DXINFO("Done.");
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

bool Model::LoadLegacy(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading model from %s...", file.c_str());
  if (!ReadLegacy(is)) {
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
    DXERROR("Failed to write model.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool Model::SaveFeatureKV(const std::string& file,
                          int feature_kv_protocol_version) const {
  return FeatureKVUtil::SaveModel(file, *graph_, param_,
                                  feature_kv_protocol_version);
}

void Model::Merge(Model* other, const Shard* shard, int shard_id) {
  DXINFO("Merging model...");
  auto tsr_reduce_func = [](const std::string& name, tsr_t& local_W,
                            tsr_t& remote_W) {
    DXINFO("Merging TSR %s...", name.c_str());
    local_W = std::move(remote_W);
  };
  auto srm_reduce_func = [shard, shard_id](const std::string& name,
                                           srm_t& local_W, srm_t& remote_W) {
    DXINFO("Merging SRM %s...", name.c_str());
    local_W.merge_if(
        std::move(remote_W), [shard, shard_id](const srm_t::value_type& entry) {
          return shard == nullptr || shard->HasSRM(shard_id, entry.first);
        });
  };
  Reduce(other, tsr_reduce_func, srm_reduce_func, shard, shard_id);
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
  DXINFO("Removing zeros from model...");
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
    const id_set_t& id_set = entry.second;
    auto& local_W = param_.get<srm_t>(name);
    auto& remote_W = remote_param->get_or_insert<srm_t>(name);
    remote_W.set_col(local_W.col());
    remote_W.reserve(id_set.size());
    if (pull_request.is_train) {
      // get random values for missing keys
      if (use_lock_) {
        auto& lock =
            param_lock_.unsafe_get<std::shared_ptr<ReadWriteLock>>(name);
        for (int_t id : id_set) {
          const float_t* embedding = local_W.get_row(engine, id, lock.get());
          // view, zero-copy
          remote_W.assign_view(id, embedding);
        }
      } else {
        for (int_t id : id_set) {
          const float_t* embedding = local_W.get_row(engine, id);
          // view, zero-copy
          remote_W.assign_view(id, embedding);
        }
      }
    } else {
      // get nothing for missing keys
      for (int_t id : id_set) {
        const float_t* embedding = ((const srm_t&)local_W).get_row_no_init(id);
        if (embedding) {
          // view, zero-copy
          remote_W.assign_view(id, embedding);
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
  auto tsr_reduce_func = [](const std::string& /*name*/, tsr_t& local_W,
                            tsr_t& remote_W) {
    // copy, not view
    local_W.set_data(remote_W);
  };
  auto srm_reduce_func = [this](const std::string& name, srm_t& local_W,
                                srm_t& remote_W) {
    if (use_lock_) {
      auto& lock = param_lock_.unsafe_get<std::shared_ptr<ReadWriteLock>>(name);
      // copy, not view
      local_W.upsert(remote_W, lock.get());
    } else {
      // copy, not view
      local_W.upsert(remote_W);
    }
  };
  Reduce(param, tsr_reduce_func, srm_reduce_func);
}

void Model::Reduce(TensorMap* param, const tsr_reduce_func_t& tsr_reduce_func,
                   const srm_reduce_func_t& srm_reduce_func, const Shard* shard,
                   int shard_id) {
  for (auto& entry : *param) {
    const std::string& name = entry.first;
    auto it = param_.find(name);
    if (it == param_.end()) {
      continue;
    }

    Any& local_Wany = it->second;
    Any& remote_Wany = entry.second;
    if (local_Wany.is<tsr_t>() && remote_Wany.is<tsr_t>() &&
        (shard == nullptr || shard->HasTSR(shard_id, name))) {
      auto& local_W = local_Wany.unsafe_to_ref<tsr_t>();
      auto& remote_W = remote_Wany.unsafe_to_ref<tsr_t>();
      if (local_W.same_shape(remote_W)) {
        tsr_reduce_func(name, local_W, remote_W);
      }
    } else if (local_Wany.is<srm_t>() && remote_Wany.is<srm_t>()) {
      auto& local_W = local_Wany.unsafe_to_ref<srm_t>();
      auto& remote_W = remote_Wany.unsafe_to_ref<srm_t>();
      if (local_W.col() == remote_W.col()) {
        srm_reduce_func(name, local_W, remote_W);
      }
    }
  }
}

void Model::Reduce(Model* other, const tsr_reduce_func_t& tsr_reduce_func,
                   const srm_reduce_func_t& srm_reduce_func, const Shard* shard,
                   int shard_id) {
  Reduce(&other->param_, tsr_reduce_func, srm_reduce_func, shard, shard_id);
}

}  // namespace deepx_core
