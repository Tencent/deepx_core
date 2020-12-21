// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace deepx_core {

// feature kv utility.
//
// feature kv manual.
// https://git.code.oa.com/wxg-td/featurekv/blob/master/docs/manual.md
//
// In deepx_core, there are 3 types of objects stored in feature kv.
// 1. meta data.
// 1.1. feature kv protocol version.
// 1.1.1. key is "version".
// 1.1.2. value is 2 of 'int' type for feature kv protocol version 2.
// 1.1.3. value is 3 of 'int' type for feature kv protocol version 3.
// 1.2. computational graph.
// 1.2.1. key is "graph".
// 1.2.2. value is the serialized buffer of the graph.
// 2. dense param.
// 2.1. key is the name of 'tsr_t'.
// 2.2. value is the 'float_t' buffer of 'tsr_t'.
// 3. sparse param.
// 3.1. key is the id of 'int_t' type.
// 3.2. value is the aggregated embedding.
//      Typical values are as follows.
// 3.2.1. feature kv protocol version 2.
//        node_id1(uint16_t)
//        embedding_col1(uint16_t)
//        embedding1(float_t * embedding_col1)
//        node_id2(uint16_t)
//        embedding_col2(uint16_t)
//        embedding2(float_t * embedding_col2)
//        node_id3(uint16_t)
//        embedding_col3(uint16_t)
//        embedding3(float_t * embedding_col3)
//        ...
// 3.2.2. feature kv protocol version 3.
//        node_id1(uint16_t)
//        embedding_col1(uint16_t)
//        embedding1(half_t * embedding_col1)
//        node_id2(uint16_t)
//        embedding_col2(uint16_t)
//        embedding2(half_t * embedding_col2)
//        node_id3(uint16_t)
//        embedding_col3(uint16_t)
//        embedding3(half_t * embedding_col3)
//        ...
//
// 'version' in this file means feature kv protocol version.
// 'id' in this file means feature id.
// 'embedding' in this file means feature embedding.
// 'item' in this file is exactly the same as the one in feature kv manual.
//
// NOTE: 'FeatureKVHead' is only little endian compatible, but not checked.
// NOTE: 'FeatureKVUtil' is endian sensitive.

/************************************************************************/
/* FeatureKVHead */
/************************************************************************/
#pragma pack(1)
struct FeatureKVHead {
  uint8_t magic;
  uint8_t flag;
  uint8_t key_size;
  uint32_t value_size;
};
#pragma pack()

/************************************************************************/
/* FeatureKVUtil */
/************************************************************************/
class FeatureKVUtil : public DataType {
 public:
  static void CheckVersion(int version);
  static std::string GetVersionKey();
  static bool GetVersion(const std::string& value, int* version) noexcept;
  static std::string GetGraphKey();
  static bool GetGraph(const std::string& value, Graph* graph);

 private:
  static const std::string& GetDenseParamKey(const std::string& name) noexcept;
  static const std::string& GetDenseParamName(const std::string& key) noexcept;
  static void GetSparseParamKey(int_t id, std::string* key);
  static bool GetSparseParamId(const std::string& key, int_t* id) noexcept;

 public:
  static void GetDenseParamKeys(const TensorMap& param,
                                std::vector<std::string>* keys);
  static void GetSparseParamKeys(const id_set_t& id_set,
                                 std::vector<std::string>* keys);
  static void GetSparseParamKeys(const std::vector<int_t>& ids,
                                 std::vector<std::string>* keys);
  static void GetSparseParamKeys(const PullRequest& pull_request,
                                 std::vector<std::string>* keys);

 private:
  static void GetItem(const std::string& key, const char* value,
                      size_t value_size, std::string* item);
  static void GetVersionItem(int version, std::string* item);
  static bool GetGraphItem(const Graph& graph, std::string* item);
  static void GetDenseParamItem(const std::string& key, const tsr_t& W,
                                std::string* item);
  // <node_id, embedding_col, embedding>
  using sparse_value_t = std::tuple<uint16_t, uint16_t, const float_t*>;
  using sparse_values_t = std::vector<sparse_value_t>;
  using id_2_sparse_values_t = std::unordered_map<int_t, sparse_values_t>;
  static void GetSparseParamItem(const std::string& key,
                                 const sparse_values_t& values,
                                 std::string* item, int version);

 public:
  // for unit test
  static bool WriteVersion(OutputStream& os,  // NOLINT
                           int version);
  // for unit test
  static bool WriteGraph(OutputStream& os,  // NOLINT
                         const Graph& graph);
  // for unit test
  static bool WriteDenseParam(OutputStream& os,  // NOLINT
                              const TensorMap& param);
  // for unit test
  static bool WriteSparseParam(OutputStream& os,  // NOLINT
                               const Graph& graph, const TensorMap& param,
                               int version);
  // for unit test
  static bool WriteSparseParam(OutputStream& os,  // NOLINT
                               const Graph& graph, const TensorMap& param,
                               const id_set_t& id_set, int version);

 private:
  static bool WriteSparseParam(OutputStream& os,  // NOLINT
                               const id_2_sparse_values_t& id_2_sparse_values,
                               int version);

 public:
  static bool WriteModel(OutputStream& os,  // NOLINT
                         const Graph& graph, const TensorMap& param,
                         int version);
  static bool WriteModel(OutputStream& os,  // NOLINT
                         const Graph& graph, const TensorMap& param,
                         const id_set_t& id_set, int version);
  static bool SaveModel(const std::string& file, const Graph& graph,
                        const TensorMap& param, int version);
  static bool SaveModel(const std::string& file, const Graph& graph,
                        const TensorMap& param, const id_set_t& id_set,
                        int version);

 public:
  struct ParamParserStat {
    int key_exist = 0;
    int key_not_exist = 0;
    int key_bad = 0;
    int value_bad = 0;
    int feature_kv_client_error = 0;

    void clear() noexcept {
      key_exist = 0;
      key_not_exist = 0;
      key_bad = 0;
      value_bad = 0;
      feature_kv_client_error = 0;
    }
  };

 public:
  class DenseParamParser {
   private:
    const Graph* graph_ = nullptr;
    TensorMap* param_ = nullptr;

   public:
    void Init(const Graph* graph, TensorMap* param) noexcept;
    void Parse(const std::vector<std::string>& keys,
               const std::vector<std::string>& values,
               const std::vector<int16_t>& codes, ParamParserStat* stat);

   private:
    void Parse(const std::string& key, const std::string& value,
               ParamParserStat* stat);
  };

 public:
  class SparseParamParser {
   private:
    int view_ = 0;
    const Graph* graph_ = nullptr;
    TensorMap* param_ = nullptr;
    int version_ = 0;
    std::vector<srm_t*> W_;  // indexed by node id

   public:
    void set_view(int view) noexcept { view_ = view; }
    int view() const noexcept { return view_; }

   public:
    void Init(const Graph* graph, TensorMap* param, int version);
    void Parse(const std::vector<std::string>& keys,
               const std::vector<std::string>& values,
               const std::vector<int16_t>& codes, ParamParserStat* stat);

   private:
    void Parse(const std::string& key, const std::string& value,
               ParamParserStat* stat);
  };

 private:
  static bool _GetKeyValue(const char*& item_buf,  // NOLINT
                           size_t& item_buf_size,  // NOLINT
                           std::string* key, std::string* value);

 public:
  // Get the 1st key and value.
  // for unit test and demo
  static bool GetKeyValue(const char* item_buf, size_t item_buf_size,
                          std::string* key, std::string* value);
  // for unit test and demo
  static bool GetKeyValue(const std::string& item, std::string* key,
                          std::string* value);
  // Get all keys and values.
  // for unit test and demo
  static bool GetKeyValues(const char* item_buf, size_t item_buf_size,
                           std::vector<std::string>* keys,
                           std::vector<std::string>* values);
  // for unit test and demo
  static bool GetKeyValues(const std::string& item,
                           std::vector<std::string>* keys,
                           std::vector<std::string>* values);
};

}  // namespace deepx_core
