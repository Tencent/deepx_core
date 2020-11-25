// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/ps/tcp_server.h>

namespace deepx_core {

/************************************************************************/
/* ParamServer */
/************************************************************************/
class ParamServer : public TcpServer {
 public:
  void Run() override;

 protected:
  int OnReadMessage(conn_t conn) override;
  virtual void OnPullRequest(conn_t conn) = 0;
  virtual void OnPushNotify(conn_t conn) = 0;
  virtual void OnModelSaveRequest(conn_t conn) = 0;
  virtual void OnTerminationNotify(conn_t conn) = 0;
  virtual void OnUserRequest(conn_t conn);
  virtual void OnUserResponse(conn_t conn);
  virtual void OnUserNotify(conn_t conn);
};

}  // namespace deepx_core
