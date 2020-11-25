// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <string>

namespace deepx_core {

/************************************************************************/
/* Op */
/************************************************************************/
class Op : public DataType {
 public:
  virtual ~Op() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual void Init(const Graph* graph, const GraphNode* node, TensorMap* param,
                    Hidden* hidden, TensorMap* ptr, TensorMap* grad,
                    TensorMap* grad_ptr, TensorMap* overwritten_param,
                    TensorMap* overwritten_ptr) = 0;
  virtual void InitForward() = 0;
  virtual void InitPredict() = 0;
  virtual void InitBackward() = 0;
  virtual void Forward() = 0;
  virtual void Predict() = 0;
  virtual void Backward() = 0;
  virtual void GetPullRequest(PullRequest* pull_request) const = 0;
};

/************************************************************************/
/* Op functions */
/************************************************************************/
std::unique_ptr<Op> NewOp(const std::string& name);

}  // namespace deepx_core
