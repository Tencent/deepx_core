// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
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
/* Optimizer */
/************************************************************************/
class Optimizer : public DataType {
 public:
  virtual ~Optimizer() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual void Init(const Graph* graph, TensorMap* param) = 0;
  virtual bool InitConfig(const AnyMap& config) = 0;
  virtual bool InitConfig(const StringMap& config) = 0;
  virtual bool InitParam() = 0;
  virtual void InitLock(AnyMap* param_lock) = 0;
  virtual bool Write(OutputStream& os) const = 0;  // NOLINT
  virtual bool Read(InputStream& is) = 0;          // NOLINT
  virtual void Warmup(Optimizer* other) = 0;

 public:
  // thread safe after 'InitLock'
  virtual void Update(TensorMap* grad) = 0;
  virtual void ForEachSRM(
      const std::function<void(const std::string&, srm_t*)>& func) = 0;
};

/************************************************************************/
/* Optimizer functions */
/************************************************************************/
std::unique_ptr<Optimizer> NewOptimizer(const std::string& name);
bool SaveOptimizer(const std::string& file, const Optimizer& optimizer);
std::unique_ptr<Optimizer> LoadOptimizer(const std::string& file);

}  // namespace deepx_core
