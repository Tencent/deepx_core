// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/common/hash.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/shard.h>
#include <cstdint>
#include <utility>

namespace deepx_core {

int Shard::DefaultTSRShardFunc(const std::string& name,
                               int shard_size) noexcept {
  return (int)((uint32_t)MurmurHash2(name) % (uint32_t)shard_size);
}

int Shard::DefaultSRMShardFunc(int_t id, int shard_size) noexcept {
  return (int)((uint32_t)id % (uint32_t)shard_size);
}

Shard::Shard()
    : shard_id_(0),
      shard_size_(1),
      tsr_shard_func_(&DefaultTSRShardFunc),
      srm_shard_func_(&DefaultSRMShardFunc) {}

Shard::Shard(int shard_id, int shard_size,
             const tsr_shard_func_t& tsr_shard_func,
             const srm_shard_func_t& srm_shard_func)
    : shard_id_(shard_id),
      shard_size_(shard_size),
      tsr_shard_func_(tsr_shard_func ? tsr_shard_func : &DefaultTSRShardFunc),
      srm_shard_func_(srm_shard_func ? srm_shard_func : &DefaultSRMShardFunc) {
  DXCHECK(shard_id >= 0);
  DXCHECK(shard_id < shard_size);
}

void Shard::SplitPullRequest(const PullRequest& full_pull_request,
                             std::vector<PullRequest>* pull_requests,
                             std::vector<id_set_t*>* aux) const {
  DXASSERT((int)pull_requests->size() == shard_size_);
  DXASSERT((int)aux->size() == shard_size_);
  for (PullRequest& pull_request : *pull_requests) {
    pull_request.clear();
    pull_request.is_train = full_pull_request.is_train;
  }

  for (const std::string& name : full_pull_request.tsr_set) {
    int shard_id = tsr_shard_func_(name, shard_size_);
    (*pull_requests)[shard_id].tsr_set.emplace(name);
  }

  for (const auto& entry : full_pull_request.srm_map) {
    const std::string& name = entry.first;
    const id_set_t& id_set = entry.second;
    size_t srm_id_size = id_set.size() / shard_size_;
    for (int i = 0; i < shard_size_; ++i) {
      (*aux)[i] = &(*pull_requests)[i].srm_map[name];
      (*aux)[i]->reserve(srm_id_size);
    }
    for (int_t id : id_set) {
      int shard_id = srm_shard_func_(id, shard_size_);
      (*aux)[shard_id]->emplace(id);
    }
  }

  for (const auto& entry : full_pull_request.id_freq_map) {
    int_t id = entry.first;
    freq_t freq = entry.second;
    int shard_id = srm_shard_func_(id, shard_size_);
    (*pull_requests)[shard_id].id_freq_map.emplace(id, freq);
  }
}

void Shard::SplitGrad(const TensorMap& param, TensorMap* full_grad,
                      std::vector<std::unique_ptr<TensorMap>>* grads,
                      std::vector<srm_t*>* aux) const {
  DXASSERT((int)grads->size() == shard_size_);
  DXASSERT((int)aux->size() == shard_size_);
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
      int shard_id = tsr_shard_func_(name, shard_size_);
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
        size_t srm_id_size = G.size() / shard_size_;
        for (int i = 0; i < shard_size_; ++i) {
          (*aux)[i] = &(*grads)[i]->get_or_insert<srm_t>(name);
          (*aux)[i]->set_col(G.col());
          (*aux)[i]->reserve(srm_id_size);
        }
        for (const auto& _entry : G) {
          int_t id = _entry.first;
          const float_t* embedding = _entry.second;
          int shard_id = srm_shard_func_(id, shard_size_);
          // view, zero-copy
          (*aux)[shard_id]->assign_view(id, embedding);
        }
      }
    }
  }
}

void Shard::SplitParam(const TensorMap& full_param,
                       std::vector<std::unique_ptr<TensorMap>>* params,
                       std::vector<srm_t*>* aux) const {
  DXASSERT((int)params->size() == shard_size_);
  DXASSERT((int)aux->size() == shard_size_);
  for (auto& param : *params) {
    param->ClearValue();
  }

  for (const auto& entry : full_param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      int shard_id = tsr_shard_func_(name, shard_size_);
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      // view, zero-copy
      (*params)[shard_id]->get_or_insert<tsr_t>(name) = W.get_view();
    } else if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      size_t srm_id_size = W.size() / shard_size_;
      for (int i = 0; i < shard_size_; ++i) {
        (*aux)[i] = &(*params)[i]->get_or_insert<srm_t>(name);
        (*aux)[i]->set_col(W.col());
        (*aux)[i]->reserve(srm_id_size);
      }
      for (const auto& _entry : W) {
        int_t id = _entry.first;
        const float_t* embedding = _entry.second;
        int shard_id = srm_shard_func_(id, shard_size_);
        // view, zero-copy
        (*aux)[shard_id]->assign_view(id, embedding);
      }
    }
  }
}

}  // namespace deepx_core
