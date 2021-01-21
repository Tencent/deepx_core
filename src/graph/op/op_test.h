// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// include all headers needed by operator UTs
#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_gtest.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <cstdint>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace deepx_core {

using param_initializer_t =
    std::function<void(std::default_random_engine&, TensorMap*)>;
using inst_initializer_t = std::function<void(Instance*)>;

void CheckOpForward(GraphNode* node, int on_heap,
                    const DataType::tsr_t& expected_forward,
                    const param_initializer_t& pre_param_initializer = nullptr,
                    const param_initializer_t& post_param_initializer = nullptr,
                    const inst_initializer_t& inst_initializer = nullptr);
void CheckOpBackward(
    GraphNode* node, int on_heap,
    const param_initializer_t& pre_param_initializer = nullptr,
    const param_initializer_t& post_param_initializer = nullptr,
    const inst_initializer_t& inst_initializer = nullptr);

}  // namespace deepx_core
