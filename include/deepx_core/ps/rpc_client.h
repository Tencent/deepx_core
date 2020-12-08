// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/array_view.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/dist_message.h>
#include <deepx_core/ps/tcp_connection.h>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* Rpc client functions for TcpConnection */
/************************************************************************/
// Return 0, success.
// Return -1, error.

int WriteTerminationNotify(TcpConnection* conn);

template <class Request>
int WriteRequest(TcpConnection* conn, int rpc_type, const Request& request) {
  std::string& out_buf =
      conn->mutable_out_message()->mutable_user_request()->buf;
  out_buf.clear();
  OutputStringStream os;
  os.SetView(&out_buf);

  os << rpc_type;
  os << request;
  if (!os) {
    DXERROR("Failed to serialize message.");
    return -1;
  }

  conn->mutable_out_message()->set_type(DIST_MESSAGE_TYPE_USER_REQUEST);
  if (conn->RpcWrite() == -1) {
    return -1;
  }
  return 0;
}

template <class Response>
int ReadResponse(TcpConnection* conn, Response* response) {
  if (conn->RpcRead() == -1) {
    return -1;
  }

  const const_string_view& in_buf = conn->in_message().user_response().buf;
  InputStringStream is;
  is.SetView(in_buf.data(), in_buf.size());

  int rpc_type;
  is >> rpc_type;
  is >> *response;
  if (!is) {
    DXERROR("Failed to deserialize message.");
    return -1;
  }
  return 0;
}

template <class Response>
int ReadResponseView(TcpConnection* conn, Response* response) {
  if (conn->RpcRead() == -1) {
    return -1;
  }

  const const_string_view& in_buf = conn->in_message().user_response().buf;
  InputStringStream is;
  is.SetView(in_buf.data(), in_buf.size());

  int rpc_type;
  ReadView(is, rpc_type);
  ReadView(is, *response);
  if (!is) {
    DXERROR("Failed to deserialize message.");
    return -1;
  }
  return 0;
}

template <class Request, class Response>
int WriteRequestReadResponse(TcpConnection* conn, int rpc_type,
                             const Request& request, Response* response) {
  if (WriteRequest(conn, rpc_type, request) == -1) {
    return -1;
  }
  if (ReadResponse(conn, response) == -1) {
    return -1;
  }
  return 0;
}

template <class Request, class Response>
int WriteRequestReadResponseView(TcpConnection* conn, int rpc_type,
                                 const Request& request, Response* response) {
  if (WriteRequest(conn, rpc_type, request) == -1) {
    return -1;
  }
  if (ReadResponseView(conn, response) == -1) {
    return -1;
  }
  return 0;
}

template <class Notify>
int WriteNotify(TcpConnection* conn, int rpc_type, const Notify& notify) {
  std::string& out_buf =
      conn->mutable_out_message()->mutable_user_notify()->buf;
  out_buf.clear();
  OutputStringStream os;
  os.SetView(&out_buf);

  os << rpc_type;
  os << notify;
  if (!os) {
    DXERROR("Failed to serialize message.");
    return -1;
  }

  conn->mutable_out_message()->set_type(DIST_MESSAGE_TYPE_USER_NOTIFY);
  if (conn->RpcWrite() == -1) {
    return -1;
  }
  return 0;
}

/************************************************************************/
/* Rpc client functions for TcpConnections */
/************************************************************************/
// Return 0, success.
// Return -1, error.

int WriteTerminationNotify(TcpConnections* conns,
                           const std::vector<int>* masks = nullptr);

template <class Request>
int WriteRequest(TcpConnections* conns, int rpc_type,
                 const std::vector<Request>& requests,
                 const std::vector<int>* masks = nullptr) {
  DXASSERT(conns->size() == requests.size());
  if (masks) {
    DXASSERT(conns->size() == masks->size());
  }

  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      const Request& request = requests[i];
      if (WriteRequest(conn, rpc_type, request) == -1) {
        return -1;
      }
    }
  }
  return 0;
}

template <class Response>
int ReadResponse(TcpConnections* conns, std::vector<Response>* responses,
                 const std::vector<int>* masks = nullptr) {
  DXASSERT(conns->size() == responses->size());
  if (masks) {
    DXASSERT(conns->size() == masks->size());
  }

  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      Response* response = &(*responses)[i];
      if (ReadResponse(conn, response) == -1) {
        return -1;
      }
    }
  }
  return 0;
}

template <class Response>
int ReadResponseView(TcpConnections* conns, std::vector<Response>* responses,
                     const std::vector<int>* masks = nullptr) {
  DXASSERT(conns->size() == responses->size());
  if (masks) {
    DXASSERT(conns->size() == masks->size());
  }

  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      Response* response = &(*responses)[i];
      if (ReadResponseView(conn, response) == -1) {
        return -1;
      }
    }
  }
  return 0;
}

template <class Request, class Response>
int WriteRequestReadResponse(TcpConnections* conns, int rpc_type,
                             const std::vector<Request>& requests,
                             std::vector<Response>* responses,
                             const std::vector<int>* masks = nullptr) {
  DXASSERT(conns->size() == requests.size());
  DXASSERT(conns->size() == responses->size());
  if (masks) {
    DXASSERT(conns->size() == masks->size());
  }

  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      const Request& request = requests[i];
      if (WriteRequest(conn, rpc_type, request) == -1) {
        return -1;
      }
    }
  }
  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      Response* response = &(*responses)[i];
      if (ReadResponse(conn, response) == -1) {
        return -1;
      }
    }
  }
  return 0;
}

template <class Request, class Response>
int WriteRequestReadResponseView(TcpConnections* conns, int rpc_type,
                                 const std::vector<Request>& requests,
                                 std::vector<Response>* responses,
                                 const std::vector<int>* masks = nullptr) {
  DXASSERT(conns->size() == requests.size());
  DXASSERT(conns->size() == responses->size());
  if (masks) {
    DXASSERT(conns->size() == masks->size());
  }

  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      const Request& request = requests[i];
      if (WriteRequest(conn, rpc_type, request) == -1) {
        return -1;
      }
    }
  }
  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      Response* response = &(*responses)[i];
      if (ReadResponseView(conn, response) == -1) {
        return -1;
      }
    }
  }
  return 0;
}

template <class Notify>
int WriteNotify(TcpConnections* conns, int rpc_type,
                const std::vector<Notify>& notifies,
                const std::vector<int>* masks = nullptr) {
  DXASSERT(conns->size() == notifies.size());
  if (masks) {
    DXASSERT(conns->size() == masks->size());
  }

  for (size_t i = 0; i < conns->size(); ++i) {
    if (masks == nullptr || (*masks)[i]) {
      TcpConnection* conn = (*conns)[i].get();
      const Notify& notify = notifies[i];
      if (WriteNotify(conn, rpc_type, notify) == -1) {
        return -1;
      }
    }
  }
  return 0;
}

}  // namespace deepx_core
