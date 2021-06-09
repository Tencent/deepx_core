// Copyright 2021 the deepx authors.
// Author: Xingfei Li (xingfeili@tencent.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <functional>
#include <memory>
#include <string>

namespace deepx_core {

/************************************************************************/
/* WePSOptimizer */
/************************************************************************/
class WePSOptimizer : public DataType {
 public:
  virtual ~WePSOptimizer() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual void Init(const Graph* graph, TensorMap* param) = 0;
  virtual bool InitConfig(const AnyMap& config) = 0;
  virtual bool InitConfig(const StringMap& config) = 0;
  virtual bool InitParam() = 0;
  virtual bool Write(OutputStream& os) const = 0;  // NOLINT
  virtual bool Read(InputStream& is) = 0;          // NOLINT

 public:
  virtual void Update(TensorMap* grad, TensorMap* delta_param) = 0;
  virtual void ForEachSRM(
      const std::function<void(const std::string&, srm_t*)>& func) = 0;
};

/************************************************************************/
/* WePSOptimizer functions */
/************************************************************************/
std::unique_ptr<WePSOptimizer> NewWePSOptimizer(const std::string& name);
bool SaveWePSOptimizer(const std::string& file, const WePSOptimizer& optimizer);
bool LoadWePSOptimizer(const std::string& file, WePSOptimizer* optimizer);
bool LoadWePSOptimizerName(const std::string& file, std::string* name);

}  // namespace deepx_core
