// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace deepx_core {

/************************************************************************/
/* WePSClient */
/************************************************************************/
class WePSClient : public DataType {
 public:
  virtual ~WePSClient() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual bool InitConfig(const AnyMap& config) = 0;
  virtual bool InitConfig(const StringMap& config) = 0;

 public:
  virtual bool SetTSR(const std::string& name, const tsr_t& W) = 0;
  virtual bool GetTSR(const std::string& name, tsr_t* W) = 0;
  virtual bool UpdateTSR(const std::string& name, const tsr_t& delta_W,
                         tsr_t* new_W) = 0;

  virtual bool SetTSR(const TensorMap& param) = 0;
  virtual bool GetTSR(TensorMap* param) = 0;
  virtual bool UpdateTSR(const TensorMap& delta_param,
                         TensorMap* new_param) = 0;

  virtual bool SetSRM(const std::string& name, const srm_t& W) = 0;
  virtual bool GetSRM(const std::string& name, const id_set_t& id_set,
                      srm_t* W) = 0;
  virtual bool UpdateSRM(const std::string& name, const srm_t& delta_W) = 0;

  virtual bool SetSRM(const TensorMap& param) = 0;
  virtual bool GetSRM(
      const std::unordered_map<std::string, id_set_t>& id_set_map,
      TensorMap* param) = 0;
  virtual bool UpdateSRM(const TensorMap& delta_param) = 0;

  static std::string GetGraphKey() noexcept { return "graph"; }
  virtual bool SetGraph(const Graph& graph) = 0;
  virtual bool GetGraph(Graph* graph, int* exist) = 0;
};

/************************************************************************/
/* WePSClient functions */
/************************************************************************/
std::unique_ptr<WePSClient> NewWePSClient(const std::string& name);

}  // namespace deepx_core
