// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/ps/file_dispatcher.h>
#include <deepx_core/ps/tcp_server.h>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* CoordServerConfig */
/************************************************************************/
struct CoordServerConfig {
  TcpEndpoint listen_endpoint;
  std::vector<TcpEndpoint> ps_endpoints;
  int epoch = 0;
  std::vector<std::string> file_dispatcher_files;
  int file_dispatcher_reverse = 0;
  int file_dispatcher_shuffle = 0;
  int file_dispatcher_timeout = 0;
  int dump_model = 0;
};

/************************************************************************/
/* CoordServer */
/************************************************************************/
class CoordServer : public TcpServer {
 protected:
  CoordServerConfig cs_config_;
  FileDispatcher file_dispatcher_;
  int epoch_ = 0;

 public:
  void set_config(const CoordServerConfig& cs_config);

 public:
  void Run() override;

 protected:
  virtual void PreTrain();
  virtual void PostTrain();
  virtual void PreEpoch();
  virtual void PostEpoch();
  virtual void DumpModel();
  virtual void TerminatePS();
  void DeleteConnection(conn_t conn) override;
  int OnReadMessage(conn_t conn) override;
  virtual void OnFileRequest(conn_t conn);
  virtual void OnFileFinishNotify(conn_t conn);
  virtual void OnUserRequest(conn_t conn);
  virtual void OnUserResponse(conn_t conn);
  virtual void OnUserNotify(conn_t conn);
};

}  // namespace deepx_core
