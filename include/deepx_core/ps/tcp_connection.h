// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/any.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/ps/dist_message.h>
#include <asio.hpp>
#include <memory>
#include <string>
#include <vector>

namespace deepx_core {

using IoContext = asio::io_context;
using IpAddress = asio::ip::address;
using TcpEndpoint = asio::ip::tcp::endpoint;
using TcpSocket = asio::ip::tcp::socket;
using MutableBuffers = asio::mutable_buffers_1;
using TcpAcceptor = asio::ip::tcp::acceptor;
using TcpNoDelay = asio::ip::tcp::no_delay;
using SteadyTimer = asio::steady_timer;

/************************************************************************/
/* TcpEndpoint functions */
/************************************************************************/
TcpEndpoint MakeTcpEndpoint(const std::string& ip, int port);
TcpEndpoint MakeTcpEndpoint(const std::string& addr);
std::vector<TcpEndpoint> MakeTcpEndpoints(const std::string& addrs);
std::string to_string(const TcpEndpoint& endpoint);

/************************************************************************/
/* TcpConnection */
/************************************************************************/
class TcpConnection {
 private:
  static constexpr size_t INITIAL_BUF_BYTES = 100 * 1024;    // magic number
  static constexpr size_t MAX_BUF_BYTES = 10 * 1024 * 1024;  // magic number

  IoContext* const io_;
  std::unique_ptr<TcpSocket> socket_;
  TcpEndpoint remote_;

  DistMessageView in_message_;
  std::string in_buf_;
  size_t in_buf_offset_ = 0;
  InputStringStream in_stream_;
  // bytes read for current message
  size_t in_bytes_ = 0;

  DistMessage out_message_;
  OutputStringStream out_stream_;
  // bytes written for current message
  size_t out_bytes_ = 0;

  // the file that remote worker is processing
  std::string file_;

  Any user_data_;

 public:
  TcpSocket& socket() noexcept { return *socket_; }
  const TcpSocket& socket() const noexcept { return *socket_; }
  const TcpEndpoint& remote() const noexcept { return remote_; }
  DistMessageView* mutable_in_message() noexcept { return &in_message_; }
  const DistMessageView& in_message() const noexcept { return in_message_; }
  DistMessage* mutable_out_message() noexcept { return &out_message_; }
  const DistMessage& out_message() const noexcept { return out_message_; }
  void clear_file() noexcept { file_.clear(); }
  void set_file(const std::string& file) { file_ = file; }
  const std::string& file() const noexcept { return file_; }
  Any* mutable_user_data() noexcept { return &user_data_; }
  const Any& user_data() const noexcept { return user_data_; }

 public:
  explicit TcpConnection(IoContext* io);
  ~TcpConnection();
  TcpConnection(const TcpConnection& other) = delete;
  TcpConnection& operator=(const TcpConnection& other) = delete;

 public:
  void Close();
  void Reset();
  MutableBuffers GetInBuf();
  void PrepareOutBuf();
  MutableBuffers GetOutBuf();

  // 'in_bytes' bytes has been read,
  // try to deserialize data to 'in_message_'.
  //
  // Return 0, success.
  // Return 1, incomplete message.
  // Return -2, message deserialization error.
  int TryReadMessage(size_t in_bytes);

  // Read data and deserialize to 'in_message_'.
  //
  // Return 0, success.
  // Return -1, socket error.
  // Return -2, message deserialization error.
  int ReadMessage();

  // Write 'out_message_'.
  //
  // Return 0, success.
  // Return -1, socket error.
  int WriteMessage();

  // 'out_bytes' bytes has been written.
  //
  // Return 0, completely written.
  // Return 1, incompletely written.
  int OnWritten(size_t out_bytes);

  // Connect to 'remote'.
  //
  // Return 0, success.
  // Return -1, socket error.
  int Connect(const TcpEndpoint& remote);

  // Connect to 'remote' with retries.
  //
  // Return 0, success.
  // Return -1, socket error.
  int ConnectRetry(const TcpEndpoint& remote, int retries = 100,
                   int second = 3);

  void OnConnect();

 public:
  // Rpc client functions.
  //
  // Return 0, success.
  // Return -1, error.
  int RpcWrite();
  int RpcRead();
  int Rpc(int type);
  int RpcEchoRequest();
  int RpcHeartBeatNotify();
  int RpcFileRequest();
  int RpcFileFinishNotify();
  int RpcPullRequest();
  int RpcPushNotify();
  int RpcModelSaveRequest();
  int RpcTerminationNotify();
  int RpcUserRequest();
  int RpcUserResponse();
  int RpcUserNotify();
};

/************************************************************************/
/* TcpConnections */
/************************************************************************/
class TcpConnections : public std::vector<std::unique_ptr<TcpConnection>> {
 private:
  IoContext* const io_;

 public:
  explicit TcpConnections(IoContext* io);
  ~TcpConnections();
  TcpConnections(const TcpConnections& other) = delete;
  TcpConnections& operator=(const TcpConnections& other) = delete;

 public:
  void Close();

  // Connect to 'remotes'.
  //
  // Return 0, success.
  // Return -1, socket error.
  int Connect(const std::vector<TcpEndpoint>& remotes);

  // Connect to 'remotes' with retries.
  //
  // Return 0, success.
  // Return -1, socket error.
  int ConnectRetry(const std::vector<TcpEndpoint>& remotes, int retries = 100,
                   int second = 3);

 public:
  // Rpc client functions.
  //
  // Return 0, success.
  // Return -1, error.
  int Rpc(int type, const std::vector<int>* masks = nullptr);
  int RpcPullRequest(const std::vector<int>* masks = nullptr);
  int RpcPushNotify(const std::vector<int>* masks = nullptr);
  int RpcModelSaveRequest(const std::vector<int>* masks = nullptr);
  int RpcTerminationNotify(const std::vector<int>* masks = nullptr);
  int RpcUserRequest(const std::vector<int>* masks = nullptr);
  int RpcUserResponse(const std::vector<int>* masks = nullptr);
  int RpcUserNotify(const std::vector<int>* masks = nullptr);
};

}  // namespace deepx_core
