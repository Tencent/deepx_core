// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/ps/param_server.h>

namespace deepx_core {

void ParamServer::Run() { RunLoop(); }

int ParamServer::OnReadMessage(conn_t conn) {
  const DistMessageView& in = conn->in_message();
  switch (in.type()) {
    case DIST_MESSAGE_TYPE_PULL_REQUEST:
      conn->mutable_out_message()->set_type(DIST_MESSAGE_TYPE_PULL_RESPONSE);
      OnPullRequest(conn);
      break;
    case DIST_MESSAGE_TYPE_PUSH_NOTIFY:
      OnPushNotify(conn);
      break;
    case DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST:
      conn->mutable_out_message()->set_type(
          DIST_MESSAGE_TYPE_MODEL_SAVE_RESPONSE);
      OnModelSaveRequest(conn);
      break;
    case DIST_MESSAGE_TYPE_TERMINATION_NOTIFY:
      OnTerminationNotify(conn);
      StopLoop();
      break;
    case DIST_MESSAGE_TYPE_USER_REQUEST:
      conn->mutable_out_message()->set_type(DIST_MESSAGE_TYPE_USER_RESPONSE);
      OnUserRequest(conn);
      break;
    case DIST_MESSAGE_TYPE_USER_RESPONSE:
      OnUserResponse(conn);
      break;
    case DIST_MESSAGE_TYPE_USER_NOTIFY:
      OnUserNotify(conn);
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

void ParamServer::OnUserRequest(conn_t /*conn*/) {
  // input
  //   conn->in_message().user_request()
  // output
  //   conn->mutable_out_message()->mutable_user_response()
}

void ParamServer::OnUserResponse(conn_t /*conn*/) {
  // input
  //   conn->in_message().user_response()
}

void ParamServer::OnUserNotify(conn_t /*conn*/) {
  // input
  //   conn->in_message().user_notify()
}

}  // namespace deepx_core
