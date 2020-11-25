// Copyright 2020 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/str_util.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/rpc_client.h>
#include <deepx_core/ps/rpc_server.h>
#include <deepx_core/ps/tcp_connection.h>
#include <deepx_core/ps/tcp_server.h>
#include <gflags/gflags.h>
#include <algorithm>  // std::sort, ...
#include <chrono>
#include <cstdint>
#include <cstdlib>  // std::getenv
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace deepx_core {
namespace {

// 'wk' in this file means worker.
// 'ps' in this file means param server.
// 'cs' in this file means coord server/scheduler.

constexpr int RPC_TYPE_PS_ADDR_REQUEST = 0x04a1e6c3;  // magic number

/************************************************************************/
/* PSAddrRequest */
/************************************************************************/
struct PSAddrRequest {
  int is_ps = 0;         // 1: from ps, 0: from wk
  int ps_id = 0;         // only for ps
  uint16_t ps_port = 0;  // only for ps
};

OutputStream& operator<<(OutputStream& os, const PSAddrRequest& request) {
  os << request.is_ps << request.ps_id << request.ps_port;
  return os;
}

InputStream& operator>>(InputStream& is, PSAddrRequest& request) {
  is >> request.is_ps >> request.ps_id >> request.ps_port;
  return is;
}

/************************************************************************/
/* PSAddrResponse */
/************************************************************************/
struct PSAddrResponse {
  int ps_id = 0;  // only for ps
  std::vector<std::string> ps_addrs;
};

OutputStream& operator<<(OutputStream& os, const PSAddrResponse& response) {
  os << response.ps_id << response.ps_addrs;
  return os;
}

InputStream& operator>>(InputStream& is, PSAddrResponse& response) {
  is >> response.ps_id >> response.ps_addrs;
  return is;
}

/************************************************************************/
/* PSItem */
/************************************************************************/
class PSItem {
 private:
  int id_ = 0;
  TcpEndpoint addr_;

 public:
  void set_id(int id) noexcept { id_ = id; }
  int id() const noexcept { return id_; }
  const TcpEndpoint& addr() const noexcept { return addr_; }
  std::string addr_str() const { return to_string(addr_); }

 public:
  PSItem() = default;
  PSItem(int id, TcpEndpoint addr) : id_(id), addr_(std::move(addr)) {}
};

/************************************************************************/
/* PSItems */
/************************************************************************/
class PSItems : public std::vector<PSItem> {
 public:
  const_iterator find(const TcpEndpoint& addr) const noexcept {
    return std::find_if(begin(), end(), [&addr](const PSItem& ps_item) {
      return ps_item.addr() == addr;
    });
  }

  int GetId(const TcpEndpoint& addr) const noexcept {
    auto it = find(addr);
    if (it != end()) {
      return it->id();
    }
    return -1;
  }

  std::vector<std::string> GetAddrs() const {
    std::vector<std::string> ps_addrs(size());
    for (std::size_t i = 0; i < size(); ++i) {
      ps_addrs[i] = (*this)[i].addr_str();
    }
    return ps_addrs;
  }

  void AllocId() {
    int has_id = 1;
    for (const PSItem& ps_item : *this) {
      if (ps_item.id() == -1) {
        has_id = 0;
        break;
      }
    }

    if (has_id) {
      std::sort(begin(), end(), [](const PSItem& left, const PSItem& right) {
        return left.id() < right.id();
      });
    } else {
      std::sort(begin(), end(), [](const PSItem& left, const PSItem& right) {
        return left.addr() < right.addr();
      });
    }

    for (std::size_t i = 0; i < size(); ++i) {
      PSItem& ps_item = (*this)[i];
      ps_item.set_id((int)i);
      DXINFO("ps_addr=%s, ps_id=%d", ps_item.addr_str().c_str(), ps_item.id());
    }
  }
};

/************************************************************************/
/* CSAdapter */
/************************************************************************/
class CSAdapter {
 private:
  TcpEndpoint cs_addr_;
  int ps_size_ = 0;
  int wk_size_ = 0;
  PSItems ps_items_;
  enum STATE {
    STATE_NONE = 0,
    STATE_PHASE1 = 1,
    STATE_PHASE2 = 2,
  };
  int state_ = STATE_NONE;
  int phase1_finish_size_ = 0;
  int phase2_finish_size_ = 0;

 private:
  void TerminateCS() const {
    IoContext io;
    TcpConnection conn(&io);
    DXCHECK_THROW(conn.ConnectRetry(cs_addr_) == 0);
    DXCHECK_THROW(WriteTerminationNotify(&conn) == 0);
  }

 public:
  std::vector<std::string> GetPSAddrs() const { return ps_items_.GetAddrs(); }

 public:
  void Init(const TcpEndpoint& cs_addr, int ps_size, int wk_size) {
    cs_addr_ = cs_addr;
    ps_size_ = ps_size;
    wk_size_ = wk_size;
    ps_items_.clear();
    state_ = STATE_PHASE1;
    phase1_finish_size_ = 0;
    phase2_finish_size_ = 0;
  }

  int OnPSAddrRequest(RpcServer::conn_t conn, const PSAddrRequest& request,
                      PSAddrResponse* response) {
    TcpEndpoint addr = conn->remote();
    if (request.is_ps) {
      addr.port(request.ps_port);
    }

    switch (state_) {
      case STATE_PHASE1:
        response->ps_id = -1;
        response->ps_addrs.clear();
        if (request.is_ps) {
          if (ps_items_.find(addr) == ps_items_.end()) {
            PSItem ps_item(request.ps_id, addr);
            ps_items_.emplace_back(ps_item);
            if (++phase1_finish_size_ == ps_size_) {
              ps_items_.AllocId();
              state_ = STATE_PHASE2;
            }
          }
        }
        break;
      case STATE_PHASE2:
        if (request.is_ps) {
          response->ps_id = ps_items_.GetId(addr);
        } else {
          response->ps_id = -1;
        }
        response->ps_addrs = ps_items_.GetAddrs();
        if (++phase2_finish_size_ == ps_size_ + wk_size_) {
          TerminateCS();
        }
        break;
    }
    return 0;
  }
};

/************************************************************************/
/* CS */
/************************************************************************/
void RunCS(const TcpEndpoint& cs_addr, int ps_size, int wk_size) {
  CSAdapter cs_adapter;
  cs_adapter.Init(cs_addr, ps_size, wk_size);

  TcpServerConfig config;
  config.listen_endpoint = cs_addr;
  config.thread = 1;

  RpcServer rpc_server;
  rpc_server.set_config(config);
  rpc_server.RegisterRequestHandler<PSAddrRequest, PSAddrResponse>(
      RPC_TYPE_PS_ADDR_REQUEST,
      std::bind(&CSAdapter::OnPSAddrRequest, &cs_adapter, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));
  rpc_server.Run();

  std::cout << "role=cs" << std::endl;
  std::cout << "cs_addr=" << cs_addr << std::endl;
  std::cout << "ps_addrs=" << Join(cs_adapter.GetPSAddrs(), ";") << std::endl;
}

/************************************************************************/
/* PS */
/************************************************************************/
uint16_t GetListenablePort(int is_v4) {
  IoContext io;
  TcpAcceptor acceptor(io);
  TcpEndpoint addr;
  if (is_v4) {
    addr = TcpEndpoint(asio::ip::tcp::v4(), 0);
  } else {
    addr = TcpEndpoint(asio::ip::tcp::v6(), 0);
  }
  acceptor.open(addr.protocol());
  acceptor.set_option(TcpAcceptor::reuse_address(true));
  acceptor.bind(addr);
  acceptor.listen();
  return acceptor.local_endpoint().port();
}

void RunPS(const TcpEndpoint& cs_addr, int ps_id) {
  IoContext io;
  TcpConnection conn(&io);
  DXCHECK_THROW(conn.ConnectRetry(cs_addr) == 0);

  PSAddrRequest request;
  request.is_ps = 1;
  request.ps_id = ps_id;
  request.ps_port = GetListenablePort(cs_addr.address().is_v4() ? 1 : 0);

  PSAddrResponse response;
  for (;;) {
    DXCHECK_THROW(WriteRequestReadResponse(&conn, RPC_TYPE_PS_ADDR_REQUEST,
                                           request, &response) == 0);
    if (response.ps_id != -1 && !response.ps_addrs.empty()) {
      break;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(100));  // magic number
  }

  std::cout << "role=ps" << std::endl;
  std::cout << "cs_addr=" << cs_addr << std::endl;
  std::cout << "ps_id=" << response.ps_id << std::endl;
  std::cout << "ps_addrs=" << Join(response.ps_addrs, ";") << std::endl;
}

/************************************************************************/
/* WK */
/************************************************************************/
void RunWK(const TcpEndpoint& cs_addr) {
  IoContext io;
  TcpConnection conn(&io);
  DXCHECK_THROW(conn.ConnectRetry(cs_addr) == 0);

  PSAddrRequest request;
  request.is_ps = 0;
  request.ps_id = -1;
  request.ps_port = 0;

  PSAddrResponse response;
  for (;;) {
    DXCHECK_THROW(WriteRequestReadResponse(&conn, RPC_TYPE_PS_ADDR_REQUEST,
                                           request, &response) == 0);
    if (!response.ps_addrs.empty()) {
      break;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(100));  // magic number
  }

  std::cout << "role=wk" << std::endl;
  std::cout << "cs_addr=" << cs_addr << std::endl;
  std::cout << "ps_addrs=" << Join(response.ps_addrs, ";") << std::endl;
}

/************************************************************************/
/* main */
/************************************************************************/
template <typename T>
bool getenv(const char* name, T* value) {
  const char* env = std::getenv(name);
  if (env == nullptr) {
    return false;
  }

  DXINFO("%s=%s", name, env);
  std::istringstream is(env);
  is >> *value;
  return is && is.eof();
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::string role;
  DXCHECK_THROW(getenv("DMLC_ROLE", &role));
  DXCHECK_THROW(role == "scheduler" || role == "server" || role == "worker");

  std::string cs_ip;
  int cs_port;
  TcpEndpoint cs_addr;
  DXCHECK_THROW(getenv("DMLC_PS_ROOT_URI", &cs_ip));
  DXCHECK_THROW(getenv("DMLC_PS_ROOT_PORT", &cs_port));
  cs_addr = MakeTcpEndpoint(cs_ip, cs_port);

  if (role == "scheduler") {
    int ps_size = 0;
    int wk_size = 0;
    DXCHECK_THROW(getenv("DMLC_NUM_SERVER", &ps_size));
    DXCHECK_THROW(getenv("DMLC_NUM_WORKER", &wk_size));
    RunCS(cs_addr, ps_size, wk_size);
  } else if (role == "server") {
    int ps_id;
    if (!getenv("DMLC_SERVER_ID", &ps_id)) {
      ps_id = -1;
    }
    RunPS(cs_addr, ps_id);
  } else {
    RunWK(cs_addr);
  }

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
