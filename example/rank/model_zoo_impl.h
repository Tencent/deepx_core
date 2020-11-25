// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// include all headers needed by model zoos
#include <deepx_core/common/class_factory.h>
#include <deepx_core/common/group_config.h>
#include <deepx_core/common/misc.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph_module_creator.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/variable_scope.h>
#include <cmath>
#include <vector>
#include "model_zoo.h"

namespace deepx_core {

#define MODEL_ZOO_REGISTER(class_name, name) \
  CLASS_FACTORY_REGISTER(ModelZoo, class_name, name)
#define MODEL_ZOO_NEW(name) CLASS_FACTORY_NEW(ModelZoo, name)
#define MODEL_ZOO_NAMES() CLASS_FACTORY_NAMES(ModelZoo)
#define DEFINE_MODEL_ZOO_LIKE(clazz_name) \
  const char* class_name() const noexcept override { return #clazz_name; }

/************************************************************************/
/* ModelZooImpl */
/************************************************************************/
class ModelZooImpl : public ModelZoo {
 protected:
  std::vector<GroupConfigItem3> items_;
  int item_is_fm_ = 0;
  int item_m_ = 0;
  int item_k_ = 0;
  int item_mk_ = 0;
  int has_w_ = 0;
  int sparse_ = 0;

 public:
  bool InitConfig(const AnyMap& config) override;
  bool InitConfig(const StringMap& config) override;

 protected:
  virtual bool PreInitConfig() { return true; }
  virtual bool InitConfigKV(const std::string& k, const std::string& v) = 0;
  virtual bool PostInitConfig() { return true; }
};

}  // namespace deepx_core
