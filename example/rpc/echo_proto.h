// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <string>

namespace deepx_core {

constexpr int RPC_TYPE_ECHO = 0x438bdb53;          // magic number
constexpr int RPC_TYPE_REVERSE_ECHO = 0x17342474;  // magic number

struct EchoRequest {
  std::string s;
};

struct EchoResponse {
  std::string s;
};

inline OutputStream& operator<<(OutputStream& os,
                                const EchoRequest& echo_request) {
  os << echo_request.s;
  return os;
}

inline InputStream& operator>>(InputStream& is, EchoRequest& echo_request) {
  is >> echo_request.s;
  return is;
}

inline InputStringStream& ReadView(InputStringStream& is,        // NOLINT
                                   EchoRequest& echo_request) {  // NOLINT
  is >> echo_request.s;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const EchoResponse& echo_response) {
  os << echo_response.s;
  return os;
}

inline InputStream& operator>>(InputStream& is, EchoResponse& echo_response) {
  is >> echo_response.s;
  return is;
}

inline InputStream& ReadView(InputStream& is,                // NOLINT
                             EchoResponse& echo_response) {  // NOLINT
  is >> echo_response.s;
  return is;
}

}  // namespace deepx_core
