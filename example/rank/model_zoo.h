// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/graph/graph.h>
#include <memory>
#include <string>

namespace deepx_core {

/************************************************************************/
/* ModelZoo */
/************************************************************************/
class ModelZoo {
 public:
  virtual ~ModelZoo() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual bool InitConfig(const AnyMap& config) = 0;
  virtual bool InitConfig(const StringMap& config) = 0;
  virtual bool InitGraph(Graph* graph) const = 0;
};

/************************************************************************/
/* ModelZoo functions */
/************************************************************************/
std::unique_ptr<ModelZoo> NewModelZoo(const std::string& name);

}  // namespace deepx_core
