// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

/************************************************************************/
/* OpImpl */
/************************************************************************/
void OpImpl::GetPullRequest(PullRequest* pull_request) const {
  for (const GraphNode* node : node_->input()) {
    if (node->node_type() == GRAPH_NODE_TYPE_PARAM) {
      switch (node->tensor_type()) {
        case TENSOR_TYPE_TSR:
          pull_request->tsr_set.emplace(node->name());
          break;
        default:
          DXTHROW_RUNTIME_ERROR("Please override %s::GetPullRequest.",
                                class_name());
          break;
      }
    }
  }
}

/************************************************************************/
/* Op functions */
/************************************************************************/
std::unique_ptr<Op> NewOp(const std::string& name) {
  std::unique_ptr<Op> op(OP_NEW(name));
  if (!op) {
    DXERROR("Invalid op name: %s.", name.c_str());
    DXERROR("Op name can be:");
    for (const std::string& _name : OP_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return op;
}

}  // namespace deepx_core
