// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/tensor/csr_matrix.h>
#include <deepx_core/tensor/ll_math.h>
#include <deepx_core/tensor/ll_tensor.h>
#include <deepx_core/tensor/sparse_row_matrix.h>
#include <deepx_core/tensor/tensor.h>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace deepx_core {

/************************************************************************/
/* DataTypeT */
/************************************************************************/
template <typename T, typename I, int TOKEN>
class DataTypeT {
 public:
  using float_t = T;
  using int_t = I;
  using tsr_t = Tensor<float_t>;
  using srm_t = SparseRowMatrix<float_t, int_t>;
  using csr_t = CSRMatrix<float_t, int_t>;
  using tsri_t = Tensor<int_t>;
  using tsrs_t = Tensor<std::string>;
  using ll_math_t = LLMath<float_t>;
  using ll_tensor_t = LLTensor<float_t>;
  using ll_sparse_tensor_t = LLSparseTensor<float_t, int_t>;
  using ll_optimizer_t = LLOptimizer<float_t, int_t>;

  using id_set_t = std::unordered_set<int_t>;
  using freq_t = uint32_t;
  using id_freq_map_t = std::unordered_map<int_t, freq_t>;
  using ts_t = uint32_t;
  using id_ts_map_t = std::unordered_map<int_t, ts_t>;

  using tsr_shard_func_t = std::function<int(const std::string&, int)>;
  using srm_shard_func_t = std::function<int(int_t, int)>;

  static constexpr int DATA_TYPE_TOKEN = TOKEN;
};

template <typename T, typename I, int TOKEN>
constexpr int DataTypeT<T, I, TOKEN>::DATA_TYPE_TOKEN;

using DataTypeS = DataTypeT<float, uint64_t, 1>;
using DataTypeD = DataTypeT<double, uint64_t, 2>;
#if HAVE_FLOAT64 == 1
using DataType = DataTypeD;
#else
using DataType = DataTypeS;
#endif

}  // namespace deepx_core
