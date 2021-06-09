// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/contrib/we_ps/client/we_ps_client_impl.h>
#include <deepx_core/dx_log.h>

namespace deepx_core {

/************************************************************************/
/* WePSClient functions */
/************************************************************************/
std::unique_ptr<WePSClient> NewWePSClient(const std::string& name) {
  std::unique_ptr<WePSClient> we_ps_client(WE_PS_CLIENT_NEW(name));
  if (!we_ps_client) {
    DXERROR("Invalid WePS client name: %s.", name.c_str());
    DXERROR("WePS client name can be:");
    for (const std::string& _name : WE_PS_CLIENT_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return we_ps_client;
}

}  // namespace deepx_core
