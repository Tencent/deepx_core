// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/feature_kv_util.h>
#include <cstring>  // memcpy
#include <limits>   // std::numeric_limits
#if HAVE_SAGE2 == 1
#include <sage2/half.h>
#endif

namespace deepx_core {

/************************************************************************/
/* FeatureKVUtil */
/************************************************************************/
void FeatureKVUtil::CheckVersion(int version) {
  if (version != 2 && version != 3) {
    DXTHROW_INVALID_ARGUMENT("Invalid feature kv protocol version: %d.",
                             version);
  }

  if (version == 3) {
#if HAVE_SAGE2 == 0
    DXTHROW_RUNTIME_ERROR(
        "Please recompile with sage2 for feature kv protocol version 3.");
#endif
  }
}

std::string FeatureKVUtil::GetVersionKey() { return "version"; }

bool FeatureKVUtil::GetVersion(const std::string& value,
                               int* version) noexcept {
  if (value.size() != sizeof(int)) {
    return false;
  }
  *version = *(const int*)value.data();
  return true;
}

std::string FeatureKVUtil::GetGraphKey() { return "graph"; }

bool FeatureKVUtil::GetGraph(const std::string& value, Graph* graph) {
  return graph->ParseFromString(value);
}

const std::string& FeatureKVUtil::GetDenseParamKey(
    const std::string& name) noexcept {
  return name;
}

const std::string& FeatureKVUtil::GetDenseParamName(
    const std::string& key) noexcept {
  return key;
}

void FeatureKVUtil::GetSparseParamKey(int_t id, std::string* key) {
  const char* begin = (const char*)&id;
  const char* end = begin + sizeof(int_t);
  key->assign(begin, end);
}

bool FeatureKVUtil::GetSparseParamId(const std::string& key,
                                     int_t* id) noexcept {
  if (key.size() != sizeof(int_t)) {
    return false;
  }
  *id = *(const int_t*)key.data();
  return true;
}

void FeatureKVUtil::GetDenseParamKeys(const TensorMap& param,
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

void FeatureKVUtil::GetSparseParamKeys(const id_set_t& id_set,
                                       std::vector<std::string>* keys) {
  auto it = id_set.begin();
  keys->resize(id_set.size());
  for (size_t i = 0; i < id_set.size(); ++i, ++it) {
    int_t id = *it;
    std::string& key = (*keys)[i];
    GetSparseParamKey(id, &key);
  }
}

void FeatureKVUtil::GetSparseParamKeys(const std::vector<int_t>& ids,
                                       std::vector<std::string>* keys) {
  keys->resize(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    int_t id = ids[i];
    std::string& key = (*keys)[i];
    GetSparseParamKey(id, &key);
  }
}

void FeatureKVUtil::GetSparseParamKeys(const PullRequest& pull_request,
                                       std::vector<std::string>* keys) {
  id_set_t id_set;
  for (const auto& entry : pull_request.srm_map) {
    id_set.insert(entry.second.begin(), entry.second.end());
  }
  GetSparseParamKeys(id_set, keys);
}

void FeatureKVUtil::GetItem(const std::string& key, const char* value,
                            size_t value_size, std::string* item) {
  item->resize(sizeof(FeatureKVHead) + key.size() + value_size);
  char* buf = &(*item)[0];

  // head
  FeatureKVHead* head = (FeatureKVHead*)buf;  // NOLINT
  head->magic = 0xb2;
  head->flag = 0;
  DXASSERT(0 < key.size() &&
           key.size() <= (size_t)std::numeric_limits<uint8_t>::max());
  head->key_size = (uint8_t)key.size();
  DXASSERT(0 < value_size &&
           value_size <= (size_t)std::numeric_limits<uint32_t>::max());
  head->value_size = (uint32_t)value_size;
  buf += sizeof(FeatureKVHead);

  // key
  memcpy(buf, key.data(), key.size());
  buf += key.size();

  // value
  memcpy(buf, value, value_size);
}

void FeatureKVUtil::GetVersionItem(int version, std::string* item) {
  GetItem(GetVersionKey(), (const char*)&version, sizeof(version), item);
}

bool FeatureKVUtil::GetGraphItem(const Graph& graph, std::string* item) {
  OutputStringStream os;
  if (!graph.Write(os)) {
    return false;
  }
  GetItem(GetGraphKey(), os.GetData(), os.GetSize(), item);
  return true;
}

void FeatureKVUtil::GetDenseParamItem(const std::string& key, const tsr_t& W,
                                      std::string* item) {
  GetItem(key, (const char*)W.data(), sizeof(float_t) * W.total_dim(), item);
}

void FeatureKVUtil::GetSparseParamItem(const std::string& key,
                                       const sparse_values_t& values,
                                       std::string* item, int version) {
  size_t value_size = 0;
  if (version == 2) {
    for (const sparse_value_t& value : values) {
      value_size += sizeof(uint16_t) +                     // node_id
                    sizeof(uint16_t) +                     // embedding_col
                    sizeof(float_t) * std::get<1>(value);  // embedding
    }
  } else if (version == 3) {
#if HAVE_SAGE2 == 1
    for (const sparse_value_t& value : values) {
      value_size += sizeof(uint16_t) +                          // node_id
                    sizeof(uint16_t) +                          // embedding_col
                    sizeof(sage2_half_t) * std::get<1>(value);  // embedding
    }
#endif
  }

  item->resize(sizeof(FeatureKVHead) + key.size() + value_size);
  char* buf = &(*item)[0];

  // head
  FeatureKVHead* head = (FeatureKVHead*)buf;  // NOLINT
  head->magic = 0xb2;
  head->flag = 0;
  DXASSERT(0 < key.size() &&
           key.size() <= (size_t)std::numeric_limits<uint8_t>::max());
  head->key_size = (uint8_t)key.size();
  DXASSERT(0 < value_size &&
           value_size <= (size_t)std::numeric_limits<uint32_t>::max());
  head->value_size = (uint32_t)value_size;
  buf += sizeof(FeatureKVHead);

  // key
  memcpy(buf, key.data(), key.size());
  buf += key.size();

  // value
  for (const sparse_value_t& value : values) {
    uint16_t node_id = std::get<0>(value);
    uint16_t embedding_col = std::get<1>(value);
    const float_t* embedding = std::get<2>(value);

    *(uint16_t*)buf = node_id;
    buf += sizeof(uint16_t);

    *(uint16_t*)buf = embedding_col;
    buf += sizeof(uint16_t);

    if (version == 2) {
      memcpy(buf, embedding, sizeof(float_t) * embedding_col);
      buf += sizeof(float_t) * embedding_col;
    } else if (version == 3) {
#if HAVE_SAGE2 == 1
      sage2_half_convert((uint64_t)embedding_col, embedding,
                         (sage2_half_t*)buf);
      buf += sizeof(sage2_half_t) * embedding_col;
#endif
    }
  }
}

bool FeatureKVUtil::WriteVersion(OutputStream& os, int version) {
  CheckVersion(version);

  DXINFO("Writing version...");
  std::string item;
  GetVersionItem(version, &item);
  os.Write(item.data(), item.size());
  if (!os) {
    DXERROR("Failed to write version.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool FeatureKVUtil::WriteGraph(OutputStream& os, const Graph& graph) {
  DXINFO("Writing graph...");
  std::string item;
  if (!GetGraphItem(graph, &item)) {
    DXERROR("Failed to write graph.");
    return false;
  }
  os.Write(item.data(), item.size());
  if (!os) {
    DXERROR("Failed to write graph.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool FeatureKVUtil::WriteDenseParam(OutputStream& os, const TensorMap& param) {
  DXINFO("Writing dense param...");
  std::string item;
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      const auto& W = Wany.unsafe_to_ref<tsr_t>();
      GetDenseParamItem(GetDenseParamKey(name), W, &item);
      os.Write(item.data(), item.size());
      if (!os) {
        break;
      }
    }
  }

  if (!os) {
    DXERROR("Failed to write dense param.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool FeatureKVUtil::WriteSparseParam(OutputStream& os, const Graph& graph,
                                     const TensorMap& param, int version) {
  CheckVersion(version);

  DXINFO("Collecting sparse param...");
  id_2_sparse_values_t id_2_sparse_values;
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      uint16_t node_id = graph.find_node(name)->node_id();
      uint16_t embedding_col = (uint16_t)W.col();  // NOLINT
      for (const auto& _entry : W) {
        int_t id = _entry.first;
        const float_t* embedding = _entry.second;
        id_2_sparse_values[id].emplace_back(node_id, embedding_col, embedding);
      }
    }
  }
  DXINFO("Collected %zu ids.", id_2_sparse_values.size());
  return WriteSparseParam(os, id_2_sparse_values, version);
}

bool FeatureKVUtil::WriteSparseParam(OutputStream& os, const Graph& graph,
                                     const TensorMap& param,
                                     const id_set_t& id_set, int version) {
  CheckVersion(version);

  DXINFO("Collecting sparse param within %zu ids...", id_set.size());
  using sparse_meta_t = std::tuple<const srm_t*, uint16_t>;
  using sparse_metas_t = std::vector<sparse_meta_t>;
  sparse_metas_t sparse_metas;
  sparse_metas.reserve(param.size());
  for (const auto& entry : param) {
    const std::string& name = entry.first;
    const Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      const auto& W = Wany.unsafe_to_ref<srm_t>();
      uint16_t node_id = graph.find_node(name)->node_id();
      sparse_metas.emplace_back(&W, node_id);
    }
  }

  id_2_sparse_values_t id_2_sparse_values;
  for (int_t id : id_set) {
    for (const sparse_meta_t& sparse_meta : sparse_metas) {
      const srm_t* W = std::get<0>(sparse_meta);
      uint16_t node_id = std::get<1>(sparse_meta);
      uint16_t embedding_col = (uint16_t)W->col();  // NOLINT
      auto it = W->find(id);
      if (it != W->end()) {
        id_2_sparse_values[id].emplace_back(node_id, embedding_col, it->second);
      }
    }
  }
  DXINFO("Collected %zu ids.", id_2_sparse_values.size());
  return WriteSparseParam(os, id_2_sparse_values, version);
}

bool FeatureKVUtil::WriteSparseParam(
    OutputStream& os, const id_2_sparse_values_t& id_2_sparse_values,
    int version) {
  DXINFO("Writing sparse param...");
  std::string key;
  std::string item;
  for (const auto& entry : id_2_sparse_values) {
    int_t id = entry.first;
    const sparse_values_t& values = entry.second;
    GetSparseParamKey(id, &key);
    GetSparseParamItem(key, values, &item, version);
    os.Write(item.data(), item.size());
    if (!os) {
      break;
    }
  }

  if (!os) {
    DXERROR("Failed to write sparse param.");
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool FeatureKVUtil::WriteModel(OutputStream& os, const Graph& graph,
                               const TensorMap& param, int version) {
  return WriteVersion(os, version) && WriteGraph(os, graph) &&
         WriteDenseParam(os, param) &&
         WriteSparseParam(os, graph, param, version);
}

bool FeatureKVUtil::WriteModel(OutputStream& os, const Graph& graph,
                               const TensorMap& param, const id_set_t& id_set,
                               int version) {
  return WriteVersion(os, version) && WriteGraph(os, graph) &&
         WriteDenseParam(os, param) &&
         WriteSparseParam(os, graph, param, id_set, version);
}

bool FeatureKVUtil::SaveModel(const std::string& file, const Graph& graph,
                              const TensorMap& param, int version) {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving model to %s...", file.c_str());
  if (!WriteModel(os, graph, param, version)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool FeatureKVUtil::SaveModel(const std::string& file, const Graph& graph,
                              const TensorMap& param, const id_set_t& id_set,
                              int version) {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving model to %s...", file.c_str());
  if (!WriteModel(os, graph, param, id_set, version)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

/************************************************************************/
/* FeatureKVUtil::DenseParamParser */
/************************************************************************/
void FeatureKVUtil::DenseParamParser::Init(const Graph* graph,
                                           TensorMap* param) noexcept {
  graph_ = graph;
  param_ = param;
}

void FeatureKVUtil::DenseParamParser::Parse(
    const std::vector<std::string>& keys,
    const std::vector<std::string>& values, const std::vector<int16_t>& codes,
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
      ++stat->feature_kv_client_error;
    }
  }
}

void FeatureKVUtil::DenseParamParser::Parse(const std::string& key,
                                            const std::string& value,
                                            ParamParserStat* stat) {
  DXASSERT(!key.empty());
  DXASSERT(!value.empty());
  auto& W = param_->get<tsr_t>(GetDenseParamName(key));
  const float_t* data = (const float_t*)value.data();  // NOLINT
  int total_dim = (int)(value.size() / sizeof(float_t));
  if (total_dim == W.total_dim()) {
    // copy, not view
    W.set_data(data, total_dim);
  } else {
    ++stat->value_bad;
  }
}

/************************************************************************/
/* FeatureKVUtil::SparseParamParser */
/************************************************************************/
void FeatureKVUtil::SparseParamParser::Init(const Graph* graph,
                                            TensorMap* param, int version) {
  CheckVersion(version);

  graph_ = graph;
  param_ = param;
  version_ = version;

  size_t node_size = graph_->name_2_node().size();
  W_.assign(node_size, nullptr);
  for (auto& entry : *param_) {
    const std::string& name = entry.first;
    Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      uint16_t node_id = graph_->find_node(name)->node_id();
      DXASSERT(node_id < node_size);
      auto& W = Wany.unsafe_to_ref<srm_t>();
      W.zeros();
      W_[node_id] = &W;
    }
  }
}

void FeatureKVUtil::SparseParamParser::Parse(
    const std::vector<std::string>& keys,
    const std::vector<std::string>& values, const std::vector<int16_t>& codes,
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
      ++stat->feature_kv_client_error;
    }
  }
}

void FeatureKVUtil::SparseParamParser::Parse(const std::string& key,
                                             const std::string& value,
                                             ParamParserStat* stat) {
  DXASSERT(!key.empty());
  DXASSERT(!value.empty());
  int_t id;
  const char* buf = value.data();
  size_t buf_size = value.size();
  uint16_t node_id;
  uint16_t embedding_col;
  const char* embedding;
  srm_t* W;
  size_t float_size = 0;

  if (!GetSparseParamId(key, &id)) {
    ++stat->key_bad;
    return;
  }

  if (version_ == 2) {
    float_size = sizeof(float_t);
  } else if (version_ == 3) {
#if HAVE_SAGE2 == 1
    float_size = sizeof(sage2_half_t);
#endif
  }

  for (;;) {
    if (buf_size == 0) {
      break;
    }

    if (buf_size < sizeof(uint16_t) + sizeof(uint16_t)) {
      ++stat->value_bad;
      break;
    }

    node_id = *(const uint16_t*)buf;
    buf += sizeof(uint16_t);
    buf_size -= sizeof(uint16_t);
    embedding_col = *(const uint16_t*)buf;
    buf += sizeof(uint16_t);
    buf_size -= sizeof(uint16_t);

    if ((size_t)node_id >= W_.size() || (W = W_[node_id]) == nullptr ||
        (int)embedding_col != W->col()) {
      ++stat->value_bad;
      break;
    }

    if (buf_size < float_size * embedding_col) {
      ++stat->value_bad;
      break;
    }

    embedding = buf;
    buf += float_size * embedding_col;
    buf_size -= float_size * embedding_col;

    if (version_ == 2) {
      if (!(W->col() == 1 && embedding[0] == 0)) {
        if (view_) {
          // view, zero-copy
          W->assign_view(id, (const float_t*)embedding);
        } else {
          // copy, not view
          W->assign(id, (const float_t*)embedding);
        }
      }
    } else if (version_ == 3) {
#if HAVE_SAGE2 == 1
      sage2_half_convert((uint64_t)embedding_col,
                         (const sage2_half_t*)embedding,
                         W->get_row_no_init(id));
#endif
    }
  }
}

/************************************************************************/
/* FeatureKVUtil */
/************************************************************************/
bool FeatureKVUtil::_GetKeyValue(const char*& item_buf, size_t& item_buf_size,
                                 std::string* key, std::string* value) {
  if (item_buf_size < sizeof(FeatureKVHead)) {
    return false;
  }

  FeatureKVHead* head = (FeatureKVHead*)item_buf;  // NOLINT
  size_t key_size = (size_t)head->key_size;        // NOLINT
  size_t value_size = (size_t)head->value_size;    // NOLINT
  item_buf += sizeof(FeatureKVHead);
  item_buf_size -= sizeof(FeatureKVHead);
  if (item_buf_size < key_size + value_size) {
    return false;
  }

  key->assign(item_buf, key_size);
  item_buf += key_size;
  item_buf_size -= key_size;

  value->assign(item_buf, value_size);
  item_buf += value_size;
  item_buf_size -= value_size;
  return true;
}

bool FeatureKVUtil::GetKeyValue(const char* item_buf, size_t item_buf_size,
                                std::string* key, std::string* value) {
  return _GetKeyValue(item_buf, item_buf_size, key, value);
}

bool FeatureKVUtil::GetKeyValue(const std::string& item, std::string* key,
                                std::string* value) {
  return GetKeyValue(item.data(), item.size(), key, value);
}

bool FeatureKVUtil::GetKeyValues(const char* item_buf, size_t item_buf_size,
                                 std::vector<std::string>* keys,
                                 std::vector<std::string>* values) {
  keys->clear();
  values->clear();
  for (;;) {
    if (item_buf_size == 0) {
      break;
    }

    std::string key, value;
    if (!_GetKeyValue(item_buf, item_buf_size, &key, &value)) {
      return false;
    }

    keys->emplace_back(std::move(key));
    values->emplace_back(std::move(value));
  }
  return true;
}

bool FeatureKVUtil::GetKeyValues(const std::string& item,
                                 std::vector<std::string>* keys,
                                 std::vector<std::string>* values) {
  return GetKeyValues(item.data(), item.size(), keys, values);
}

}  // namespace deepx_core
