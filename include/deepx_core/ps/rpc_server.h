// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/tcp_server.h>
#include <functional>
#include <unordered_map>

namespace deepx_core {

/************************************************************************/
/* RpcServer */
/************************************************************************/
class RpcServer : public TcpServer {
 public:
  void Run() override;

 protected:
  int OnReadMessage(conn_t conn) override;
  int OnUserRequest(conn_t conn);
  int OnUserNotify(conn_t conn);

 protected:
  std::unordered_map<
      int, std::function<int(conn_t, InputStringStream&, OutputStringStream&)>>
      request_handler_map_;
  std::unordered_map<int, std::function<int(conn_t, InputStringStream&)>>
      notify_handler_map_;

 public:
  template <class Request, class Response>
  void RegisterRequestHandler(
      int rpc_type,
      const std::function<int(const Request&, Response*)>& request_handler) {
    auto handler = [request_handler](conn_t /*conn*/, InputStringStream& is,
                                     OutputStringStream& os) {
      Request request;
      is >> request;
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      Response response;
      if (request_handler(request, &response) == -1) {
        return -1;
      }

      os << response;
      if (!os) {
        DXERROR("Failed to serialize message.");
        return -1;
      }
      return 0;
    };
    request_handler_map_[rpc_type] = handler;
  }

  template <class Request, class Response>
  void RegisterRequestHandler(
      int rpc_type,
      const std::function<int(conn_t conn, const Request&, Response*)>&
          request_handler) {
    auto handler = [request_handler](conn_t conn, InputStringStream& is,
                                     OutputStringStream& os) {
      Request request;
      is >> request;
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      Response response;
      if (request_handler(conn, request, &response) == -1) {
        return -1;
      }

      os << response;
      if (!os) {
        DXERROR("Failed to serialize message.");
        return -1;
      }
      return 0;
    };
    request_handler_map_[rpc_type] = handler;
  }

  template <class Request, class Response>
  void RegisterRequestViewHandler(
      int rpc_type,
      const std::function<int(const Request&, Response*)>& request_handler) {
    auto handler = [request_handler](conn_t /*conn*/, InputStringStream& is,
                                     OutputStringStream& os) {
      Request request;
      ReadView(is, request);
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      Response response;
      if (request_handler(request, &response) == -1) {
        return -1;
      }

      os << response;
      if (!os) {
        DXERROR("Failed to serialize message.");
        return -1;
      }
      return 0;
    };
    request_handler_map_[rpc_type] = handler;
  }

  template <class Request, class Response>
  void RegisterRequestViewHandler(
      int rpc_type,
      const std::function<int(conn_t conn, const Request&, Response*)>&
          request_handler) {
    auto handler = [request_handler](conn_t conn, InputStringStream& is,
                                     OutputStringStream& os) {
      Request request;
      ReadView(is, request);
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      Response response;
      if (request_handler(conn, request, &response) == -1) {
        return -1;
      }

      os << response;
      if (!os) {
        DXERROR("Failed to serialize message.");
        return -1;
      }
      return 0;
    };
    request_handler_map_[rpc_type] = handler;
  }

  template <class Notify>
  void RegisterNotifyHandler(
      int rpc_type, const std::function<int(const Notify&)>& notify_handler) {
    auto handler = [notify_handler](conn_t /*conn*/, InputStringStream& is) {
      Notify notify;
      is >> notify;
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      if (notify_handler(notify) == -1) {
        return -1;
      }
      return 0;
    };
    notify_handler_map_[rpc_type] = handler;
  }

  template <class Notify>
  void RegisterNotifyHandler(
      int rpc_type,
      const std::function<int(conn_t conn, const Notify&)>& notify_handler) {
    auto handler = [notify_handler](conn_t conn, InputStringStream& is) {
      Notify notify;
      is >> notify;
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      if (notify_handler(conn, notify) == -1) {
        return -1;
      }
      return 0;
    };
    notify_handler_map_[rpc_type] = handler;
  }

  template <class Notify>
  void RegisterNotifyViewHandler(
      int rpc_type, const std::function<int(const Notify&)>& notify_handler) {
    auto handler = [notify_handler](conn_t /*conn*/, InputStringStream& is) {
      Notify notify;
      ReadView(is, notify);
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      if (notify_handler(notify) == -1) {
        return -1;
      }
      return 0;
    };
    notify_handler_map_[rpc_type] = handler;
  }

  template <class Notify>
  void RegisterNotifyViewHandler(
      int rpc_type,
      const std::function<int(conn_t conn, const Notify&)>& notify_handler) {
    auto handler = [notify_handler](conn_t conn, InputStringStream& is) {
      Notify notify;
      ReadView(is, notify);
      if (!is) {
        DXERROR("Failed to deserialize message.");
        return -1;
      }

      if (notify_handler(conn, notify) == -1) {
        return -1;
      }
      return 0;
    };
    notify_handler_map_[rpc_type] = handler;
  }
};

}  // namespace deepx_core
