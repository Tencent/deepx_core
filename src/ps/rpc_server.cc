// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/ps/dist_message.h>
#include <deepx_core/ps/rpc_server.h>
#include <string>

namespace deepx_core {

void RpcServer::Run() { RunLoop(); }

int RpcServer::OnReadMessage(conn_t conn) {
  const DistMessageView& in = conn->in_message();
  switch (in.type()) {
    case DIST_MESSAGE_TYPE_TERMINATION_NOTIFY:
      StopLoop();
      break;
    case DIST_MESSAGE_TYPE_USER_REQUEST:
      conn->mutable_out_message()->set_type(DIST_MESSAGE_TYPE_USER_RESPONSE);
      if (OnUserRequest(conn) == -1) {
        DeleteConnection(conn);
        return -1;
      }
      break;
    case DIST_MESSAGE_TYPE_USER_NOTIFY:
      if (OnUserNotify(conn) == -1) {
        DeleteConnection(conn);
        return -1;
      }
      break;
    default:
      DeleteConnection(conn);
      return -1;
  }
  if (in.HasResponse()) {
    AsyncWriteMessage(conn);
    return 1;
  }
  return 0;
}

int RpcServer::OnUserRequest(conn_t conn) {
  const const_string_view& in_buf = conn->in_message().user_request().buf;
  InputStringStream is;
  is.SetView(in_buf.data(), in_buf.size());

  std::string& out_buf =
      conn->mutable_out_message()->mutable_user_response()->buf;
  out_buf.clear();
  OutputStringStream os;
  os.SetView(&out_buf);

  int rpc_type;
  is >> rpc_type;
  if (!is) {
    DXERROR("Failed to deserialize message.");
    return -1;
  }

  os << rpc_type;
  if (!os) {
    DXERROR("Failed to serialize message.");
    return -1;
  }

  auto it = request_handler_map_.find(rpc_type);
  if (it == request_handler_map_.end()) {
    DXERROR("Unregistered rpc type: %d.", rpc_type);
    return -1;
  }

  if (it->second(conn, is, os) == -1) {
    return -1;
  }
  return 0;
}

int RpcServer::OnUserNotify(conn_t conn) {
  const const_string_view& in_buf = conn->in_message().user_notify().buf;
  InputStringStream is;
  is.SetView(in_buf.data(), in_buf.size());

  int rpc_type;
  is >> rpc_type;
  if (!is) {
    DXERROR("Failed to deserialize message.");
    return -1;
  }

  auto it = notify_handler_map_.find(rpc_type);
  if (it == notify_handler_map_.end()) {
    DXERROR("Unregistered rpc type: %d.", rpc_type);
    return -1;
  }

  if (it->second(conn, is) == -1) {
    return -1;
  }
  return 0;
}

}  // namespace deepx_core
