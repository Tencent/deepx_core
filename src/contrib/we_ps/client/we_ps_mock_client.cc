// Copyright 2021 the deepx authors
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/contrib/we_ps/client/we_ps_client_impl.h>

namespace deepx_core {

/************************************************************************/
/* WePSMockClient */
/************************************************************************/
class WePSMockClient : public WePSClient {
 private:
  TensorMap tsr_param_;
  TensorMap srm_param_;
  std::string serialized_graph_;

 public:
  DEFINE_WE_PS_CLIENT_LIKE(WePSMockClient);
  bool InitConfig(const AnyMap& /*config*/) override { return true; }
  bool InitConfig(const StringMap& /*config*/) override { return true; }

 public:
  bool SetTSR(const std::string& name, const tsr_t& W) override;
  bool GetTSR(const std::string& name, tsr_t* W) override;
  bool UpdateTSR(const std::string& name, const tsr_t& delta_W,
                 tsr_t* new_W) override;

  bool SetTSR(const TensorMap& param) override;
  bool GetTSR(TensorMap* param) override;
  bool UpdateTSR(const TensorMap& delta_param, TensorMap* new_param) override;

  bool SetSRM(const std::string& name, const srm_t& W) override;
  bool GetSRM(const std::string& name, const id_set_t& id_set,
              srm_t* W) override;
  bool UpdateSRM(const std::string& name, const srm_t& delta_W) override;

  bool SetSRM(const TensorMap& param) override;
  bool GetSRM(const std::unordered_map<std::string, id_set_t>& id_set_map,
              TensorMap* param) override;
  bool UpdateSRM(const TensorMap& delta_param) override;

  bool SetGraph(const Graph& graph) override;
  bool GetGraph(Graph* graph, int* exist) override;
};

bool WePSMockClient::SetTSR(const std::string& name, const tsr_t& remote_W) {
  tsr_param_.get_or_insert<tsr_t>(name) = remote_W;
  return true;
}

bool WePSMockClient::GetTSR(const std::string& name, tsr_t* remote_W) {
  auto it = tsr_param_.find(name);
  if (it == tsr_param_.end()) {
    return false;
  }

  const Any& Wany = it->second;
  const auto& W = Wany.unsafe_to_ref<tsr_t>();
  remote_W->set_data(W);
  return true;
}

bool WePSMockClient::UpdateTSR(const std::string& name, const tsr_t& delta_W,
                               tsr_t* new_W) {
  auto it = tsr_param_.find(name);
  if (it == tsr_param_.end()) {
    return false;
  }

  Any& Wany = it->second;
  auto& W = Wany.unsafe_to_ref<tsr_t>();
  ll_sparse_tensor_t::add_to(delta_W, &W);
  new_W->set_data(W);
  return true;
}

bool WePSMockClient::SetTSR(const TensorMap& remote_param) {
  for (const auto& entry : remote_param) {
    const std::string& name = entry.first;
    const Any& remote_Wany = entry.second;
    if (remote_Wany.is<tsr_t>()) {
      const auto& remote_W = remote_Wany.unsafe_to_ref<tsr_t>();
      if (!SetTSR(name, remote_W)) {
        return false;
      }
    }
  }
  return true;
}

bool WePSMockClient::GetTSR(TensorMap* remote_param) {
  for (auto& entry : *remote_param) {
    const std::string& name = entry.first;
    Any& remote_Wany = entry.second;
    if (remote_Wany.is<tsr_t>()) {
      auto& remote_W = remote_Wany.unsafe_to_ref<tsr_t>();
      if (!GetTSR(name, &remote_W)) {
        return false;
      }
    }
  }
  return true;
}

bool WePSMockClient::UpdateTSR(const TensorMap& delta_param,
                               TensorMap* new_param) {
  for (const auto& entry : delta_param) {
    const std::string& name = entry.first;
    const Any& delta_Wany = entry.second;
    if (delta_Wany.is<tsr_t>()) {
      const auto& delta_W = delta_Wany.unsafe_to_ref<tsr_t>();
      auto& new_W = new_param->get<tsr_t>(name);
      if (!UpdateTSR(name, delta_W, &new_W)) {
        return false;
      }
    }
  }
  return true;
}

bool WePSMockClient::SetSRM(const std::string& name, const srm_t& remote_W) {
  auto it = srm_param_.find(name);
  if (it == srm_param_.end()) {
    srm_param_.get_or_insert<srm_t>(name) = remote_W;
  } else {
    Any& Wany = it->second;
    auto& W = Wany.unsafe_to_ref<srm_t>();
    W.upsert(remote_W);
  }
  return true;
}

bool WePSMockClient::GetSRM(const std::string& name, const id_set_t& id_set,
                            srm_t* remote_W) {
  auto it = srm_param_.find(name);
  if (it == srm_param_.end()) {
    return true;
  }

  const Any& Wany = it->second;
  const auto& W = Wany.unsafe_to_ref<srm_t>();
  remote_W->zeros();
  DXASSERT(remote_W->col() == W.col());
  for (int_t id : id_set) {
    const float_t* embedding = W.get_row_no_init(id);
    if (embedding) {
      remote_W->assign(id, embedding);
    }
  }
  return true;
}

bool WePSMockClient::UpdateSRM(const std::string& name, const srm_t& delta_W) {
  auto it = srm_param_.find(name);
  if (it == srm_param_.end()) {
    return true;
  }

  Any& Wany = it->second;
  auto& W = Wany.unsafe_to_ref<srm_t>();
  DXASSERT(delta_W.col() == W.col());
  for (const auto& entry : delta_W) {
    int_t id = entry.first;
    const float_t* delta_embedding = entry.second;
    auto _it = W.find(id);
    if (_it == W.end()) {
      return false;
    }
    float_t* embedding = _it->second;
    ll_math_t::add(W.col(), embedding, delta_embedding, embedding);
  }
  return true;
}

bool WePSMockClient::SetSRM(const TensorMap& remote_param) {
  for (const auto& entry : remote_param) {
    const std::string& name = entry.first;
    const Any& remote_Wany = entry.second;
    if (remote_Wany.is<srm_t>()) {
      const auto& remote_W = remote_Wany.unsafe_to_ref<srm_t>();
      if (!SetSRM(name, remote_W)) {
        return false;
      }
    }
  }
  return true;
}

bool WePSMockClient::GetSRM(
    const std::unordered_map<std::string, id_set_t>& id_set_map,
    TensorMap* remote_param) {
  for (const auto& entry : id_set_map) {
    const std::string& name = entry.first;
    const id_set_t& id_set = entry.second;
    auto& remote_W = remote_param->get<srm_t>(name);
    if (!GetSRM(name, id_set, &remote_W)) {
      return false;
    }
  }
  return true;
}

bool WePSMockClient::UpdateSRM(const TensorMap& delta_param) {
  for (const auto& entry : delta_param) {
    const std::string& name = entry.first;
    const Any& delta_Wany = entry.second;
    if (delta_Wany.is<srm_t>()) {
      const auto& delta_W = delta_Wany.unsafe_to_ref<srm_t>();
      if (!UpdateSRM(name, delta_W)) {
        return false;
      }
    }
  }
  return true;
}

bool WePSMockClient::SetGraph(const Graph& graph) {
  OutputStringStream os;
  os.SetView(&serialized_graph_);
  return graph.Write(os);
}

bool WePSMockClient::GetGraph(Graph* graph, int* exist) {
  if (!serialized_graph_.empty()) {
    *exist = 1;
    return graph->ParseFromString(serialized_graph_);
  } else {
    *exist = 0;
    return true;
  }
}

WE_PS_CLIENT_REGISTER(WePSMockClient, "WePSMockClient");
WE_PS_CLIENT_REGISTER(WePSMockClient, "mock");

}  // namespace deepx_core
