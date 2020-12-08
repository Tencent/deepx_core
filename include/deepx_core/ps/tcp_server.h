// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/ps/tcp_connection.h>
#include <memory>

namespace deepx_core {

/************************************************************************/
/* TcpServerConfig */
/************************************************************************/
struct TcpServerConfig {
  TcpEndpoint listen_endpoint;
  int thread = 1;
};

/************************************************************************/
/* TcpServer */
/************************************************************************/
class TcpServer {
 public:
  using conn_t = const std::shared_ptr<TcpConnection>&;

 protected:
  TcpServerConfig config_;
  std::unique_ptr<IoContext> io_;
  std::unique_ptr<TcpAcceptor> acceptor_;

 public:
  void set_config(const TcpServerConfig& config);

 public:
  TcpServer() = default;
  TcpServer(const TcpServer& other) = delete;
  TcpServer& operator=(const TcpServer& other) = delete;
  virtual ~TcpServer() = default;

 public:
  virtual void Run() = 0;

 protected:
  virtual void DeleteConnection(conn_t conn);
  virtual void RunLoop();
  virtual void StopLoop();

  void AsyncAccept();
  virtual void OnAccept(conn_t conn);

  void AsyncRead(conn_t conn);
  virtual void OnRead(conn_t conn, size_t in_bytes);
  virtual int OnReadMessage(conn_t conn) = 0;

  void AsyncWrite(conn_t conn);
  virtual void OnWrite(conn_t conn, size_t out_bytes);
  void AsyncWriteMessage(conn_t conn);
  virtual void OnWriteMessage(conn_t conn);
};

}  // namespace deepx_core
