// Copyright 2021 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include <deepx_core/common/any.h>
#include <deepx_core/contrib/we_ps/graph/we_ps_util.h>
#include <deepx_core/dx_log.h>
#include <cstring>

namespace deepx_core {

/************************************************************************/
/* WePSUtil */
/************************************************************************/
std::string WePSUtil::GetGraphKey() { return "graph"; }

bool WePSUtil::GetGraph(const std::string& value, Graph* graph) {
  return graph->ParseFromString(value);
}

const std::string& WePSUtil::GetDenseParamKey(
    const std::string& name) noexcept {
  return name;
}

void WePSUtil::GetDenseParamKeys(const TensorMap& param,
                                 std::vector<std::string>* keys) {
  keys->clear();
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      keys->emplace_back(GetDenseParamKey(name));
    }
  }
}

void WePSUtil::GetSparseParamKey(const std::string& name, int_t id,
                                 std::string* key) {
  key->resize(name.size() + sizeof(int_t));
  char* buf = &(*key)[0];
  memcpy(buf, &name[0], name.size());
  buf += name.size();
  *(int_t*)buf = id;
}

void WePSUtil::GetSparseParamKeys(const PullRequest& pull_request,
                                  std::vector<std::string>* keys) {
  int total_key_size = 0;
  for (auto& entry : pull_request.srm_map) {
    total_key_size += (int)entry.second.size();
  }
  keys->resize(total_key_size);
  int idx = 0;
  for (auto& entry : pull_request.srm_map) {
    const std::string& name = entry.first;
    const id_set_t& id_set = entry.second;
    for (auto id : id_set) {
      GetSparseParamKey(name, id, &(*keys)[idx++]);
    }
  }
}

/************************************************************************/
/* WePSUtil::DenseParamParser */
/************************************************************************/
void WePSUtil::DenseParamParser::Init(TensorMap* param) noexcept {
  param_ = param;
}

void WePSUtil::DenseParamParser::Parse(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<float_t>>& values, ParamParserStat* stat) {
  DXASSERT(keys.size() == values.size());
  stat->clear();
  for (size_t i = 0; i < keys.size(); ++i) {
    ++stat->key_exist;
    Parse(keys[i], values[i], stat);
  }
}

void WePSUtil::DenseParamParser::Parse(const std::string& key,
                                       const std::vector<float_t>& value,
                                       ParamParserStat* stat) {
  DXASSERT(!key.empty());
  DXASSERT(!value.empty());
  auto& W = param_->get<tsr_t>(key);
  if ((int)value.size() == W.total_dim()) {
    // copy, not view
    W.set_data(value);
  } else {
    ++stat->value_bad;
  }
}

/************************************************************************/
/* WePSUtil::SparseParamParser */
/************************************************************************/
void WePSUtil::SparseParamParser::Init(TensorMap* param) {
  param_ = param;

  for (auto& entry : *param_) {
    Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      W.zeros();
    }
  }
}

void WePSUtil::SparseParamParser::Parse(const std::vector<std::string>& keys,
                                        const std::vector<std::string>& values,
                                        const std::vector<int16_t>& codes,
                                        ParamParserStat* stat) {
  DXASSERT(keys.size() == values.size());
  DXASSERT(keys.size() == codes.size());
  stat->clear();
  for (size_t i = 0; i < keys.size(); ++i) {
    if (codes[i] == 0) {
      ++stat->key_exist;
      Parse(keys[i], values[i], stat);
    } else if (codes[i] == 1) {
      ++stat->key_not_exist;
    } else {
      ++stat->we_ps_client_error;
    }
  }
}

bool WePSUtil::SparseParamParser::GetSparseParamNameId(const std::string& key,
                                                       std::string* name,
                                                       int_t* id) noexcept {
  if (key.size() <= sizeof(int_t)) {
    return false;
  }
  size_t name_size = key.size() - sizeof(int_t);
  name->resize(name_size);
  memcpy(&(*name)[0], key.data(), name_size);
  *id = *(const int_t*)&key[name_size];
  return true;
}

void WePSUtil::SparseParamParser::Parse(const std::string& key,
                                        const std::string& value,
                                        ParamParserStat* stat) {
  DXASSERT(!key.empty());
  DXASSERT(!value.empty());
  std::string name;
  int_t id;
  if (!GetSparseParamNameId(key, &name, &id)) {
    ++stat->key_bad;
    return;
  }

  auto& W = param_->get<srm_t>(name);
  if (value.size() != sizeof(float_t) * W.col()) {
    ++stat->value_bad;
    return;
  }

  const char* embedding = value.data();
  if (!(W.col() == 1 && embedding[0] == 0)) {
    if (view_) {
      // view, zero-copy
      W.assign_view(id, (const float_t*)embedding);
    } else {
      // copy, not view
      W.assign(id, (const float_t*)embedding);
    }
  }
}

/************************************************************************/
/* WePSUtil */
/************************************************************************/
void WePSUtil::GetSparseParamValue(const std::vector<float_t>& values,
                                   std::string* value) {
  value->resize(sizeof(float_t) * values.size());
  char* buf = &(*value)[0];
  for (float_t entry : values) {
    memcpy(buf, (const char*)&entry, sizeof(float_t));
    buf += sizeof(float_t);
  }
}

}  // namespace deepx_core
