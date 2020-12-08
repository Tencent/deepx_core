// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/ps/rpc_client.h>

namespace deepx_core {

/************************************************************************/
/* Rpc client functions for TcpConnection */
/************************************************************************/
int WriteTerminationNotify(TcpConnection* conn) {
  conn->mutable_out_message()->set_type(DIST_MESSAGE_TYPE_TERMINATION_NOTIFY);
  if (conn->RpcWrite() == -1) {
    return -1;
  }
  return 0;
}

/************************************************************************/
/* Rpc client functions for TcpConnections */
/************************************************************************/
int WriteTerminationNotify(TcpConnections* conns,
                           const std::vector<int>* masks) {
  if (masks) {
    DXASSERT(conns->size() == masks->size());
  }

  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      conn->mutable_out_message()->set_type(
          DIST_MESSAGE_TYPE_TERMINATION_NOTIFY);
      if (conn->RpcWrite() == -1) {
        return -1;
      }
    }
  }
  return 0;
}

}  // namespace deepx_core
