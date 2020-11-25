// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/ps/coord_server.h>

namespace deepx_core {

void CoordServer::set_config(const CoordServerConfig& cs_config) {
  DXCHECK_THROW(!cs_config.ps_endpoints.empty());
  DXCHECK_THROW(cs_config.epoch > 0);
  DXCHECK_THROW(!cs_config.file_dispatcher_files.empty());
  config_.listen_endpoint = cs_config.listen_endpoint;
  config_.thread = 1;
  cs_config_ = cs_config;
}

void CoordServer::Run() {
  file_dispatcher_.set_reverse(cs_config_.file_dispatcher_reverse);
  file_dispatcher_.set_shuffle(cs_config_.file_dispatcher_shuffle);
  file_dispatcher_.set_timeout(cs_config_.file_dispatcher_timeout);
  PreTrain();
  for (epoch_ = 0; epoch_ < cs_config_.epoch; ++epoch_) {
    PreEpoch();
    DXINFO("Epoch %d begins.", epoch_ + 1);
    RunLoop();
    DXINFO("Epoch %d completed.", epoch_ + 1);
    PostEpoch();
  }
  PostTrain();
  if (cs_config_.dump_model) {
    DumpModel();
  }
  TerminatePS();
}

void CoordServer::PreTrain() {
  file_dispatcher_.PreTrain(cs_config_.file_dispatcher_files);
}

void CoordServer::PostTrain() {}

void CoordServer::PreEpoch() { file_dispatcher_.PreEpoch(); }

void CoordServer::PostEpoch() {}

void CoordServer::DumpModel() {
  TcpConnections conns(io_.get());
  DXCHECK_THROW(conns.ConnectRetry(cs_config_.ps_endpoints) == 0);
  for (auto& conn : conns) {
    conn->mutable_out_message()->mutable_model_save_request()->epoch = epoch_;
  }
  DXCHECK_THROW(conns.RpcModelSaveRequest() == 0);
}

void CoordServer::TerminatePS() {
  TcpConnections conns(io_.get());
  DXCHECK_THROW(conns.ConnectRetry(cs_config_.ps_endpoints) == 0);
  DXCHECK_THROW(conns.RpcTerminationNotify() == 0);
}

void CoordServer::DeleteConnection(conn_t conn) {
  if (!conn->file().empty()) {
    file_dispatcher_.WorkerFailureFile(conn->file());
  }
  TcpServer::DeleteConnection(conn);
}

int CoordServer::OnReadMessage(conn_t conn) {
  const DistMessageView& in = conn->in_message();
  switch (in.type()) {
    case DIST_MESSAGE_TYPE_FILE_REQUEST:
      conn->mutable_out_message()->set_type(DIST_MESSAGE_TYPE_FILE_RESPONSE);
      OnFileRequest(conn);
      break;
    case DIST_MESSAGE_TYPE_FILE_FINISH_NOTIFY:
      OnFileFinishNotify(conn);
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

void CoordServer::OnFileRequest(conn_t conn) {
  auto* response = conn->mutable_out_message()->mutable_file_response();
  response->epoch = epoch_;
  response->file.clear();
  if (file_dispatcher_.WorkerDispatchFile(&response->file)) {
    conn->set_file(response->file);
  }
}

void CoordServer::OnFileFinishNotify(conn_t conn) {
  const auto& notify = conn->in_message().file_finish_notify();
  conn->clear_file();
  if (file_dispatcher_.WorkerFinishFile(notify.file)) {
    StopLoop();
  }
  DXINFO("file=%s, loss=%f", notify.file.c_str(),
         notify.loss / notify.loss_weight);
}

void CoordServer::OnUserRequest(conn_t /*conn*/) {
  // input
  //   conn->in_message().user_request()
  // output
  //   conn->mutable_out_message()->mutable_user_response()
}

void CoordServer::OnUserResponse(conn_t /*conn*/) {
  // input
  //   conn->in_message().user_response()
}

void CoordServer::OnUserNotify(conn_t /*conn*/) {
  // input
  //   conn->in_message().user_notify()
}

}  // namespace deepx_core
