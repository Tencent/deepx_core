// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/class_factory.h>
#include <deepx_core/contrib/we_ps/client/we_ps_client.h>

namespace deepx_core {

#define WE_PS_CLIENT_REGISTER(class_name, name) \
  CLASS_FACTORY_REGISTER(WePSClient, class_name, name)
#define WE_PS_CLIENT_NEW(name) CLASS_FACTORY_NEW(WePSClient, name)
#define WE_PS_CLIENT_NAMES() CLASS_FACTORY_NAMES(WePSClient)
#define DEFINE_WE_PS_CLIENT_LIKE(clazz_name) \
  const char* class_name() const noexcept override { return #clazz_name; }

}  // namespace deepx_core
