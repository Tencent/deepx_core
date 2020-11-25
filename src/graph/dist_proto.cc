// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/graph/dist_proto.h>

namespace deepx_core {

OutputStream& operator<<(OutputStream& os, const PullRequest& pull_request) {
  os << pull_request.is_train << pull_request.tsr_set << pull_request.srm_map
     << pull_request.id_freq_map;
  return os;
}

InputStream& operator>>(InputStream& is, PullRequest& pull_request) {
  is >> pull_request.is_train >> pull_request.tsr_set >> pull_request.srm_map >>
      pull_request.id_freq_map;
  return is;
}

}  // namespace deepx_core
