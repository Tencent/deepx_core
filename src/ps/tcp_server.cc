// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/ps/tcp_server.h>
#include <system_error>
#include <thread>
#include <vector>

namespace deepx_core {

void TcpServer::set_config(const TcpServerConfig& config) {
  DXCHECK_THROW(config.thread > 0);
  config_ = config;
}

void TcpServer::DeleteConnection(conn_t /*conn*/) {}

void TcpServer::RunLoop() {
  DXINFO("Listening at %s.", to_string(config_.listen_endpoint).c_str());
  io_.reset(new IoContext);
  acceptor_.reset(new TcpAcceptor(*io_, config_.listen_endpoint));
  AsyncAccept();

  DXINFO("Serving with %d threads.", config_.thread);
  std::vector<std::thread> threads;
  for (int i = 0; i < config_.thread; ++i) {
    threads.emplace_back([this] { io_->run(); });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  std::error_code ec;
  acceptor_->close(ec);
  // ignore 'ec'
  acceptor_.reset();
  DXINFO("Stopped.");
}

void TcpServer::StopLoop() {
  DXINFO("Stopping...");
  io_->stop();
}

void TcpServer::AsyncAccept() {
  auto conn = std::make_shared<TcpConnection>(io_.get());
  acceptor_->async_accept(
      conn->socket(), [this, conn](const std::error_code& ec) {
        AsyncAccept();
        if (ec) {
          DXERROR("Failed to accept: %s.", ec.message().c_str());
        } else {
          OnAccept(conn);
        }
      });
}

void TcpServer::OnAccept(conn_t conn) {
  conn->OnConnect();
  AsyncRead(conn);
}

void TcpServer::AsyncRead(conn_t conn) {
  conn->socket().async_read_some(
      conn->GetInBuf(),
      [this, conn](const std::error_code& ec, size_t in_bytes) {
        if (ec) {
          DeleteConnection(conn);
        } else {
          OnRead(conn, in_bytes);
        }
      });
}

void TcpServer::OnRead(conn_t conn, size_t in_bytes) {
  int read;
  for (;;) {
    switch (conn->TryReadMessage(in_bytes)) {
      case 0:
        // complete message
        read = OnReadMessage(conn);
        if (read == 0) {
          in_bytes = 0;
          // continue to process another possible message
          continue;
        } else if (read == 1) {
          // return
        } else {
          // error
          DeleteConnection(conn);
        }
        return;
      case 1:
        // incomplete message
        AsyncRead(conn);
        return;
      case -2:
      default:
        // error
        DeleteConnection(conn);
        return;
    }
  }
}

void TcpServer::AsyncWrite(conn_t conn) {
  conn->socket().async_write_some(
      conn->GetOutBuf(),
      [this, conn](const std::error_code& ec, size_t out_bytes) {
        if (ec) {
          DeleteConnection(conn);
        } else {
          OnWrite(conn, out_bytes);
        }
      });
}

void TcpServer::OnWrite(conn_t conn, size_t out_bytes) {
  if (conn->OnWritten(out_bytes) == 0) {
    // complete message
    OnWriteMessage(conn);
  } else {
    // incomplete message
    AsyncWrite(conn);
  }
}

void TcpServer::AsyncWriteMessage(conn_t conn) {
  conn->PrepareOutBuf();
  AsyncWrite(conn);
}

void TcpServer::OnWriteMessage(conn_t conn) { OnRead(conn, 0); }

}  // namespace deepx_core
