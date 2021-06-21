// Copyright 2021 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* WePSUtil */
/************************************************************************/
class WePSUtil : public DataType {
 public:
  static std::string GetGraphKey();
  static bool GetGraph(const std::string& value, Graph* graph);

 private:
  static const std::string& GetDenseParamKey(const std::string& name) noexcept;

 public:
  static void GetDenseParamKeys(const TensorMap& param,
                                std::vector<std::string>* keys);
  static void GetSparseParamKey(const std::string& name, int_t id,
                                std::string* key);
  static void GetSparseParamKeys(const PullRequest& pull_request,
                                 std::vector<std::string>* keys);

 public:
  struct ParamParserStat {
    int key_exist = 0;
    int key_not_exist = 0;
    int key_bad = 0;
    int value_bad = 0;
    int we_ps_client_error = 0;

    void clear() noexcept {
      key_exist = 0;
      key_not_exist = 0;
      key_bad = 0;
      value_bad = 0;
      we_ps_client_error = 0;
    }
  };

 public:
  class DenseParamParser {
   private:
    TensorMap* param_ = nullptr;

   public:
    void Init(TensorMap* param) noexcept;
    void Parse(const std::vector<std::string>& keys,
               const std::vector<std::vector<float>>& values,
               ParamParserStat* stat);

   private:
    void Parse(const std::string& key, const std::vector<float>& value,
               ParamParserStat* stat);
  };

 public:
  class SparseParamParser {
   private:
    int view_ = 0;
    TensorMap* param_ = nullptr;

   public:
    void set_view(int view) noexcept { view_ = view; }
    int view() const noexcept { return view_; }

   public:
    void Init(TensorMap* param);
    void Parse(const std::vector<std::string>& keys,
               const std::vector<std::string>& values,
               const std::vector<int16_t>& codes, ParamParserStat* stat);

   private:
    static bool GetSparseParamNameId(const std::string& key, std::string* name,
                                     int_t* id) noexcept;
    void Parse(const std::string& key, const std::string& value,
               ParamParserStat* stat);
  };

 public:
  // for unit test
  static void GetSparseParamValue(const std::vector<float_t>& values,
                                  std::string* value);
};

}  // namespace deepx_core
