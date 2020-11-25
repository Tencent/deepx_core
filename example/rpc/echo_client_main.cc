// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/ps/rpc_client.h>
#include <deepx_core/ps/tcp_connection.h>
#include <gflags/gflags.h>
#include <random>
#include <string>
#include "echo_proto.h"

DEFINE_string(server, "127.0.0.1:8888", "server address");

namespace deepx_core {
namespace {

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  TcpEndpoint endpoint = MakeTcpEndpoint(FLAGS_server);
  IoContext io;
  TcpConnection conn(&io);
  EchoRequest echo_request;
  EchoResponse echo_response;
  std::default_random_engine engine;
  std::uniform_int_distribution<int> dist;

  DXCHECK_THROW(conn.Connect(endpoint) == 0);
  for (int i = 0; i < 100; ++i) {
    echo_request.s = std::to_string(dist(engine));
    DXCHECK_THROW(WriteRequestReadResponse(&conn, RPC_TYPE_ECHO, echo_request,
                                           &echo_response) == 0);
    DXINFO("RPC_TYPE_ECHO: %s -> %s", echo_request.s.c_str(),
           echo_response.s.c_str());
    DXCHECK_THROW(WriteRequestReadResponse(&conn, RPC_TYPE_REVERSE_ECHO,
                                           echo_request, &echo_response) == 0);
    DXINFO("RPC_TYPE_REVERSE_ECHO: %s -> %s", echo_request.s.c_str(),
           echo_response.s.c_str());
  }
  DXCHECK_THROW(WriteTerminationNotify(&conn) == 0);

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
