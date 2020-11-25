// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/ps/rpc_server.h>
#include <deepx_core/ps/tcp_connection.h>
#include <deepx_core/ps/tcp_server.h>
#include <gflags/gflags.h>
#include "echo_proto.h"

DEFINE_string(listen, "127.0.0.1:8888", "listening address");
DEFINE_int32(thread, 1, "# of threads");

namespace deepx_core {
namespace {

int Echo(const EchoRequest& echo_request, EchoResponse* echo_response) {
  echo_response->s = echo_request.s;
  return 0;
}

int ReverseEcho(const EchoRequest& echo_request, EchoResponse* echo_response) {
  echo_response->s.assign(echo_request.s.rbegin(), echo_request.s.rend());
  return 0;
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  TcpServerConfig config;
  config.listen_endpoint = MakeTcpEndpoint(FLAGS_listen);
  config.thread = FLAGS_thread;

  RpcServer rpc_server;
  rpc_server.set_config(config);
  rpc_server.RegisterRequestHandler<EchoRequest, EchoResponse>(RPC_TYPE_ECHO,
                                                               &Echo);
  rpc_server.RegisterRequestHandler<EchoRequest, EchoResponse>(
      RPC_TYPE_REVERSE_ECHO, &ReverseEcho);
  rpc_server.Run();

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
