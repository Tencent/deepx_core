// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/tcp_connection.h>
#include <chrono>
#include <cstdint>
#include <limits>  // std::numeric_limits
#include <sstream>
#include <system_error>
#include <thread>
#include <utility>

namespace deepx_core {

/************************************************************************/
/* TcpEndpoint functions */
/************************************************************************/
TcpEndpoint MakeTcpEndpoint(const std::string& ip, int port) {
  std::error_code ec;
  TcpEndpoint endpoint;
  endpoint.address(IpAddress::from_string(ip, ec));
  if (ec) {
    DXTHROW_INVALID_ARGUMENT("Invalid ip: %s.", ip.c_str());
  }

  if (port <= 0 || port > (int)std::numeric_limits<uint16_t>::max()) {
    DXTHROW_INVALID_ARGUMENT("Invalid port: %d.", port);
  }
  endpoint.port((uint16_t)port);
  return endpoint;
}

TcpEndpoint MakeTcpEndpoint(const std::string& addr) {
  size_t semi = addr.rfind(':');
  if (semi == std::string::npos) {
    DXTHROW_INVALID_ARGUMENT("Invalid addr: %s.", addr.c_str());
  }

  std::error_code ec;
  TcpEndpoint endpoint;
  endpoint.address(IpAddress::from_string(addr.substr(0, semi), ec));
  if (ec) {
    DXTHROW_INVALID_ARGUMENT("Invalid addr: %s.", addr.c_str());
  }

  int port = std::stoi(&addr[semi + 1]);
  if (port <= 0 || port > (int)std::numeric_limits<uint16_t>::max()) {
    DXTHROW_INVALID_ARGUMENT("Invalid addr: %s.", addr.c_str());
  }
  endpoint.port((uint16_t)port);
  return endpoint;
}

std::vector<TcpEndpoint> MakeTcpEndpoints(const std::string& _addrs) {
  std::vector<std::string> addrs;
  Split(_addrs, ";", &addrs, true);

  std::vector<TcpEndpoint> endpoints;
  endpoints.reserve(addrs.size());
  for (const std::string& addr : addrs) {
    endpoints.emplace_back(MakeTcpEndpoint(addr));
  }
  return endpoints;
}

std::string to_string(const TcpEndpoint& endpoint) {
  std::ostringstream os;
  os << endpoint;
  return os.str();
}

/************************************************************************/
/* TcpConnection */
/************************************************************************/
TcpConnection::TcpConnection(IoContext* io) : io_(io) { Reset(); }

TcpConnection::~TcpConnection() { Close(); }

void TcpConnection::Close() { socket_.reset(); }

void TcpConnection::Reset() {
  socket_.reset(new TcpSocket(*io_));
  in_buf_.resize(INITIAL_BUF_BYTES);
  in_buf_offset_ = 0;
  in_bytes_ = 0;
  out_bytes_ = 0;
  file_.clear();
  user_data_.reset();
}

MutableBuffers TcpConnection::GetInBuf() {
  if (in_buf_offset_ > 0) {
    // trim the leading 'in_buf_offset_' bytes
    in_buf_.erase(0, in_buf_offset_);
    in_buf_offset_ = 0;
  }

  size_t in_buf_bytes = in_buf_.size();
  if (in_buf_bytes == in_bytes_) {
    // double 'in_buf_'
    in_buf_bytes = in_buf_bytes * 2;
    in_buf_.resize(in_buf_bytes);
  } else if (in_bytes_ == 0 && in_buf_bytes > MAX_BUF_BYTES) {
    // shrink 'in_buf_'
    in_buf_bytes = INITIAL_BUF_BYTES;
    in_buf_.resize(in_buf_bytes);
    in_buf_.shrink_to_fit();
  }
  return MutableBuffers{&in_buf_[in_bytes_], in_buf_bytes - in_bytes_};
}

void TcpConnection::PrepareOutBuf() {
  out_stream_ << out_message_;
  DXCHECK_THROW(out_stream_);
  out_bytes_ = 0;
}

MutableBuffers TcpConnection::GetOutBuf() {
  return asio::buffer((void*)(out_stream_.GetData() + out_bytes_),
                      out_stream_.GetSize() - out_bytes_);
}

int TcpConnection::TryReadMessage(size_t in_bytes) {
  in_bytes_ += in_bytes;
  if (in_bytes_ < sizeof(int)) {
    return 1;
  }

  const char* in_packet = in_buf_.data() + in_buf_offset_;
  size_t packet_bytes = *(const int*)in_packet;
  if (in_bytes_ < packet_bytes) {
    return 1;
  }

  in_stream_.SetView(in_packet, packet_bytes);
  ReadView(in_stream_, in_message_);
  if (in_stream_) {
    if (in_bytes_ == packet_bytes) {
      in_buf_offset_ = 0;
      in_bytes_ = 0;
    } else {
      in_buf_offset_ += packet_bytes;
      in_bytes_ -= packet_bytes;
    }
    return 0;
  }

  return -2;
}

int TcpConnection::ReadMessage() {
  std::error_code ec;
  size_t n = 0;
  int read;
  for (;;) {
    switch (read = TryReadMessage(n)) {
      case 0:
        return read;
      case -2:
        Close();
        return read;
      case 1:
      default:
        n = socket_->receive(GetInBuf(), 0, ec);
        if (ec) {
          DXERROR("Failed to read from %s: %s.", to_string(remote_).c_str(),
                  ec.message().c_str());
          Close();
          return -1;
        }
    }
  }
}

int TcpConnection::WriteMessage() {
  std::error_code ec;
  size_t to_out_bytes, n;
  PrepareOutBuf();
  to_out_bytes = out_stream_.GetSize();
  while (to_out_bytes != 0) {
    n = socket_->write_some(GetOutBuf(), ec);
    if (ec) {
      DXERROR("Failed to write to %s: %s.", to_string(remote_).c_str(),
              ec.message().c_str());
      Close();
      return -1;
    }
    to_out_bytes -= n;
    out_bytes_ += n;
  }
  return 0;
}

int TcpConnection::OnWritten(size_t out_bytes) {
  if ((out_bytes_ += out_bytes) == out_stream_.GetSize()) {
    return 0;
  }
  return 1;
}

int TcpConnection::Connect(const TcpEndpoint& remote) {
  std::error_code ec;
  socket_.reset(new TcpSocket(*io_));
  socket_->connect(remote, ec);
  if (ec) {
    DXERROR("Failed to connect to %s: %s.", to_string(remote).c_str(),
            ec.message().c_str());
    return -1;
  }
  OnConnect();
  return 0;
}

int TcpConnection::ConnectRetry(const TcpEndpoint& remote, int retries,
                                int second) {
  for (int i = 0; i < retries; ++i) {
    if (Connect(remote) == 0) {
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::seconds(second));
  }
  DXERROR("Failed to connect to %s after %d retries.",
          to_string(remote).c_str(), retries);
  return -1;
}

void TcpConnection::OnConnect() {
  std::error_code ec;
  socket_->set_option(TcpNoDelay(true), ec);
  remote_ = socket_->remote_endpoint(ec);
  // ignore 'ec'
}

int TcpConnection::RpcWrite() {
  if (WriteMessage() == 0) {
    return 0;
  }
  return -1;
}

int TcpConnection::RpcRead() {
  if (ReadMessage() == 0) {
    return 0;
  }
  return -1;
}

int TcpConnection::Rpc(int type) {
  out_message_.set_type(type);
  if (RpcWrite() == -1) {
    return -1;
  }
  if (out_message_.HasResponse()) {
    if (RpcRead() == -1) {
      return -1;
    }
  }
  return 0;
}

int TcpConnection::RpcEchoRequest() {
  return Rpc(DIST_MESSAGE_TYPE_ECHO_REQUEST);
}

int TcpConnection::RpcHeartBeatNotify() {
  return Rpc(DIST_MESSAGE_TYPE_HEART_BEAT_NOTIFY);
}

int TcpConnection::RpcFileRequest() {
  return Rpc(DIST_MESSAGE_TYPE_FILE_REQUEST);
}

int TcpConnection::RpcFileFinishNotify() {
  return Rpc(DIST_MESSAGE_TYPE_FILE_FINISH_NOTIFY);
}

int TcpConnection::RpcPullRequest() {
  return Rpc(DIST_MESSAGE_TYPE_PULL_REQUEST);
}

int TcpConnection::RpcPushNotify() {
  return Rpc(DIST_MESSAGE_TYPE_PUSH_NOTIFY);
}

int TcpConnection::RpcModelSaveRequest() {
  return Rpc(DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST);
}

int TcpConnection::RpcTerminationNotify() {
  return Rpc(DIST_MESSAGE_TYPE_TERMINATION_NOTIFY);
}

int TcpConnection::RpcUserRequest() {
  return Rpc(DIST_MESSAGE_TYPE_USER_REQUEST);
}

int TcpConnection::RpcUserResponse() {
  return Rpc(DIST_MESSAGE_TYPE_USER_RESPONSE);
}

int TcpConnection::RpcUserNotify() {
  return Rpc(DIST_MESSAGE_TYPE_USER_NOTIFY);
}

/************************************************************************/
/* TcpConnections */
/************************************************************************/
TcpConnections::TcpConnections(IoContext* io) : io_(io) {}

TcpConnections::~TcpConnections() { Close(); }

void TcpConnections::Close() { clear(); }

int TcpConnections::Connect(const std::vector<TcpEndpoint>& remotes) {
  Close();
  reserve(remotes.size());
  for (const TcpEndpoint& remote : remotes) {
    std::unique_ptr<TcpConnection> conn(new TcpConnection(io_));
    if (conn->Connect(remote) != 0) {
      Close();
      return -1;
    }
    emplace_back(std::move(conn));
  }
  return 0;
}

int TcpConnections::ConnectRetry(const std::vector<TcpEndpoint>& remotes,
                                 int retries, int second) {
  Close();
  reserve(remotes.size());
  for (const TcpEndpoint& remote : remotes) {
    std::unique_ptr<TcpConnection> conn(new TcpConnection(io_));
    if (conn->ConnectRetry(remote, retries, second) != 0) {
      Close();
      return -1;
    }
    emplace_back(std::move(conn));
  }
  return 0;
}

int TcpConnections::Rpc(int type, const std::vector<int>* masks) {
  if (masks) {
    DXASSERT(size() == masks->size());
  }

  for (size_t i = 0; i < size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*this)[i].get();
      conn->mutable_out_message()->set_type(type);
      if (conn->RpcWrite() == -1) {
        return -1;
      }
    }
  }
  if (DistMessage::HasResponse(type)) {
    for (size_t i = 0; i < size(); ++i) {
      if (masks == nullptr || (*masks)[i]) {
        TcpConnection* conn = (*this)[i].get();
        if (conn->RpcRead() == -1) {
          return -1;
        }
      }
    }
  }
  return 0;
}

int TcpConnections::RpcPullRequest(const std::vector<int>* masks) {
  return Rpc(DIST_MESSAGE_TYPE_PULL_REQUEST, masks);
}

int TcpConnections::RpcPushNotify(const std::vector<int>* masks) {
  return Rpc(DIST_MESSAGE_TYPE_PUSH_NOTIFY, masks);
}

int TcpConnections::RpcModelSaveRequest(const std::vector<int>* masks) {
  return Rpc(DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST, masks);
}

int TcpConnections::RpcTerminationNotify(const std::vector<int>* masks) {
  return Rpc(DIST_MESSAGE_TYPE_TERMINATION_NOTIFY, masks);
}

int TcpConnections::RpcUserRequest(const std::vector<int>* masks) {
  return Rpc(DIST_MESSAGE_TYPE_USER_REQUEST, masks);
}

int TcpConnections::RpcUserResponse(const std::vector<int>* masks) {
  return Rpc(DIST_MESSAGE_TYPE_USER_RESPONSE, masks);
}

int TcpConnections::RpcUserNotify(const std::vector<int>* masks) {
  return Rpc(DIST_MESSAGE_TYPE_USER_NOTIFY, masks);
}

}  // namespace deepx_core
