// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph_module_creator.h>
#include <deepx_core/graph/instance_reader.h>
#include <deepx_core/graph/variable_scope.h>
#include <cstdint>

namespace deepx_core {

/************************************************************************/
/* InstanceNode creator */
/************************************************************************/
GraphNode* GetX() {
  return new InstanceNode(X_NAME, Shape(BATCH_PLACEHOLDER, 0), TENSOR_TYPE_CSR);
}

GraphNode* GetX(int i) {
  return new InstanceNode(X_NAME + std::to_string(i),
                          Shape(BATCH_PLACEHOLDER, 0), TENSOR_TYPE_CSR);
}

GraphNode* GetXUser() {
  return new InstanceNode(X_USER_NAME, Shape(BATCH_PLACEHOLDER, 0),
                          TENSOR_TYPE_CSR);
}

GraphNode* GetXCand() {
  return new InstanceNode(X_CAND_NAME, Shape(BATCH_PLACEHOLDER, 0),
                          TENSOR_TYPE_CSR);
}

GraphNode* GetXHist(int i) {
  return new InstanceNode(X_HIST_NAME + std::to_string(i),
                          Shape(BATCH_PLACEHOLDER, 0), TENSOR_TYPE_CSR);
}

GraphNode* GetXHistSize() {
  return new InstanceNode(X_HIST_SIZE_NAME, Shape(BATCH_PLACEHOLDER),
                          TENSOR_TYPE_TSR);
}

GraphNode* GetY(int label_size) {
  return new InstanceNode(Y_NAME, Shape(BATCH_PLACEHOLDER, label_size),
                          TENSOR_TYPE_TSR);
}

GraphNode* GetW(int label_size) {
  return new InstanceNode(W_NAME, Shape(BATCH_PLACEHOLDER, label_size),
                          TENSOR_TYPE_TSR);
}

GraphNode* GetInstance(const std::string& name, const Shape& shape,
                       int tensor_type) {
  return new InstanceNode(name, shape, tensor_type);
}

/************************************************************************/
/* embedding creator */
/************************************************************************/
GraphNode* WideGroupEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                    const std::vector<GroupConfigItem3>& items,
                                    int sparse, int need_grad) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!items.empty());
  std::vector<GraphNode*> W(items.size());
  std::vector<uint16_t> group_ids(items.size());
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  for (size_t i = 0; i < items.size(); ++i) {
    group_ids[i] = items[i].group_id;
    auto ii = std::to_string(group_ids[i]);
    W[i] = GetVariable(prefix + "W" + ii, Shape(items[i].embedding_row, 1),
                       tensor_type, TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
    W[i]->set_need_grad(need_grad);
  }
  return GroupEmbeddingLookup("", X, W, group_ids);
}

GraphNode* WideGroupEmbeddingLookup2(const std::string& prefix, GraphNode* X,
                                     const std::vector<GroupConfigItem3>& items,
                                     int sparse, int need_grad) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!items.empty());
  DXCHECK_THROW(IsFMGroupConfig(items));
  std::vector<uint16_t> group_ids(items.size());
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  for (size_t i = 0; i < items.size(); ++i) {
    group_ids[i] = items[i].group_id;
  }
  auto* W = GetVariable(prefix + "W", Shape(items[0].embedding_row, 1),
                        tensor_type, TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
  W->set_need_grad(need_grad);
  return GroupEmbeddingLookup2("", X, W, group_ids);
}

GraphNode* DeepGroupEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                    const std::vector<GroupConfigItem3>& items,
                                    int sparse, int need_grad) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!items.empty());
  std::vector<GraphNode*> W(items.size());
  std::vector<uint16_t> group_ids(items.size());
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  for (size_t i = 0; i < items.size(); ++i) {
    group_ids[i] = items[i].group_id;
    auto ii = std::to_string(group_ids[i]);
    W[i] = GetVariable(prefix + "W" + ii,
                       Shape(items[i].embedding_row, items[i].embedding_col),
                       tensor_type, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1e-3);
    W[i]->set_need_grad(need_grad);
  }
  return GroupEmbeddingLookup("", X, W, group_ids);
}

GraphNode* DeepGroupEmbeddingLookup2(const std::string& prefix, GraphNode* X,
                                     const std::vector<GroupConfigItem3>& items,
                                     int sparse, int need_grad) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!items.empty());
  DXCHECK_THROW(IsFMGroupConfig(items));
  std::vector<uint16_t> group_ids(items.size());
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  for (size_t i = 0; i < items.size(); ++i) {
    group_ids[i] = items[i].group_id;
  }
  auto* W = GetVariable(prefix + "W",
                        Shape(items[0].embedding_row, items[0].embedding_col),
                        tensor_type, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1e-3);
  W->set_need_grad(need_grad);
  return GroupEmbeddingLookup2("", X, W, group_ids);
}

/************************************************************************/
/* building block creator */
/************************************************************************/
GraphNode* StackedFullyConnect(const std::string& prefix, GraphNode* X,
                               const std::vector<int>& deep_dims,
                               const std::string& activation) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!deep_dims.empty());
  DXCHECK_THROW(activation == "sigmoid" || activation == "tanh" ||
                activation == "relu");
  std::vector<int> _deep_dims;
  if (deep_dims[0] != X->shape()[1]) {
    _deep_dims.emplace_back(X->shape()[1]);
  }
  _deep_dims.insert(_deep_dims.end(), deep_dims.begin(), deep_dims.end());
  GraphNode* Z = X;
  int deep_size = (int)_deep_dims.size() - 1;
  for (int i = 0; i < deep_size; ++i) {
    auto ii = std::to_string(i);
    auto* W = GetVariableRandXavier(prefix + "W" + ii,
                                    Shape(_deep_dims[i], _deep_dims[i + 1]));
    auto* b = GetVariableZeros(prefix + "b" + ii, Shape(1, _deep_dims[i + 1]));
    auto* H = new FullyConnectNode("", Z, W, b);
    if (_deep_dims.back() == 1 && i == deep_size - 1) {
      Z = H;
    } else {
      if (activation == "sigmoid") {
        Z = Sigmoid("", H);
      } else if (activation == "tanh") {
        Z = Tanh("", H);
      } else {
        Z = Relu("", H);
      }
    }
  }
  return Z;
}

GraphNode* FullyConnect(const std::string& prefix, GraphNode* X, int out_dim) {
  DXCHECK_THROW(X->shape().is_rank(2));
  auto* W = GetVariableRandXavier(prefix + "W", Shape(X->shape()[1], out_dim));
  auto* b = GetVariableZeros(prefix + "b", Shape(1, out_dim));
  return new FullyConnectNode("", X, W, b);
}

GraphNode* AddBias(const std::string& prefix, GraphNode* X) {
  DXCHECK_THROW(X->shape().is_rank(2));
  auto* b = GetVariableZeros(prefix + "b", Shape(1, X->shape()[1]));
  return BroadcastAdd("", X, b);
}

GraphNode* SelfAttention(const std::string& prefix, GraphNode* X, int n) {
  DXCHECK_THROW(X->shape().is_rank(3));
  int k = X->shape()[2];
  auto* Wq = GetVariableRandn(prefix + "Wq", Shape(k, n));
  auto* Wk = GetVariableRandn(prefix + "Wk", Shape(k, n));
  auto* Wv = GetVariableRandn(prefix + "Wv", Shape(k, n));
  auto* C = ConstantScalar("", 1 / std::sqrt(1.0 * n));
  auto* Q = Matmul("", X, Wq);
  auto* K = Matmul("", X, Wk);
  auto* V = Matmul("", X, Wv);
  auto* Z1 = BatchGEMM("", Q, K, 0, 1);
  auto* Z2 = BroadcastMul("", Z1, C);
  auto* Z3 = Softmax("", Z2, -1);
  auto* Z4 = BatchGEMM("", Z3, V, 0, 0);
  return Z4;
}

GraphNode* CrossNet(const std::string& prefix, GraphNode* X, int cross) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int m = X->shape()[1];
  auto* Xre = Reshape("", X, Shape(-1, m, 1));
  auto* Xi = Xre;
  for (int i = 0; i < cross; ++i) {
    auto ii = std::to_string(i);
    auto* W = GetVariableRandXavier(prefix + "W" + ii, Shape(m, 1));
    auto* b = GetVariableZeros(prefix + "b" + ii, Shape(m, 1));
    auto* Z1 = TensorDot("", Xi, W, Shape(1), Shape(0));  // (batch, 1, 1)
    auto* Z2 = Matmul("", Xre, Z1);                       // (batch, m, 1)
    auto* Z3 = Add("", Xi, Z2);                           // (batch, m, 1)
    auto* Z4 = BroadcastAdd("", Z3, b);                   // (batch, m, 1)
    Xi = Z4;
  }
  auto* Z5 = ReshapeFast("", Xi, Shape(-1, m));  // (batch, m)
  return Z5;
}

GraphNode* CIN(const std::string& prefix, GraphNode* X,
               const std::vector<int>& dims) {
  DXCHECK_THROW(X->shape().is_rank(3));
  DXCHECK_THROW(!dims.empty());
  auto* X0 = X;
  auto* Xi = X;
  std::vector<GraphNode*> Z5in(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    auto ii = std::to_string(i);
    int m0mi = X0->shape()[1] * Xi->shape()[1];
    auto* W = GetVariableRandXavier(prefix + "W" + ii, Shape(m0mi, dims[i]));
    auto* Z1 = BatchFMInteraction2("", X0, Xi);
    auto* Z2 = TensorDot("", Z1, W, Shape(1), Shape(0));
    auto* Z3 = Transpose("", Z2, Shape(0, 2, 1));
    auto* Z4 = ReduceSum("", Z3, -1, 0);
    Xi = Z3;
    Z5in[i] = Z4;
  }
  auto* Z5 = Concat("", Z5in);
  return Z5;
}

GraphNode* RNNCell(const std::string& prefix, GraphNode* X, GraphNode* Hin,
                   int n) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int batch = X->shape()[0];
  int m = X->shape()[1];
  auto* U = GetVariableRandXavier(prefix + "U", Shape(m, n));
  if (Hin) {
    DXCHECK_THROW(Hin->shape().same_shape(batch, n));
    auto* W = GetVariableRandXavier(prefix + "W", Shape(n, n));
    auto* H1 = GEMM("", X, U, 0, 0);
    auto* H2 = GEMM("", Hin, W, 0, 0);
    auto* H3 = Add("", H1, H2);
    auto* H4 = Tanh("", H3);
    return H4;
  } else {
    auto* H1 = GEMM("", X, U, 0, 0);
    auto* H4 = Tanh("", H1);
    return H4;
  }
}

std::vector<GraphNode*> LSTMCell(const std::string& prefix, GraphNode* X,
                                 const std::vector<GraphNode*>& CHin, int n,
                                 int mask) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(mask == 1 || mask == 2 || mask == 3);
  int batch = X->shape()[0];
  int m = X->shape()[1];
  auto* Ui = GetVariableRandXavier(prefix + "Ui", Shape(m, n));
  auto* Uc = GetVariableRandXavier(prefix + "Uc", Shape(m, n));
  if (CHin.size() == 2) {
    GraphNode* Cin = CHin[0];
    GraphNode* Hin = CHin[1];
    DXCHECK_THROW(Cin->shape().same_shape(batch, n));
    DXCHECK_THROW(Hin->shape().same_shape(batch, n));
    auto* Uf = GetVariableRandXavier(prefix + "Uf", Shape(m, n));
    auto* Wf = GetVariableRandXavier(prefix + "Wf", Shape(n, n));
    auto* Wi = GetVariableRandXavier(prefix + "Wi", Shape(n, n));
    auto* Wc = GetVariableRandXavier(prefix + "Wc", Shape(n, n));

    auto* Hf1 = GEMM("", X, Uf, 0, 0);
    auto* Hf2 = GEMM("", Hin, Wf, 0, 0);
    auto* Hf3 = Add("", Hf1, Hf2);
    auto* Hf4 = Sigmoid("", Hf3);

    auto* Hi1 = GEMM("", X, Ui, 0, 0);
    auto* Hi2 = GEMM("", Hin, Wi, 0, 0);
    auto* Hi3 = Add("", Hi1, Hi2);
    auto* Hi4 = Sigmoid("", Hi3);

    auto* Hc1 = GEMM("", X, Uc, 0, 0);
    auto* Hc2 = GEMM("", Hin, Wc, 0, 0);
    auto* Hc3 = Add("", Hc1, Hc2);
    auto* Hc4 = Tanh("", Hc3);

    auto* H1 = Mul("", Cin, Hf4);
    auto* H2 = Mul("", Hc4, Hi4);
    auto* H3 = Add("", H1, H2);
    if (mask == 1) {
      return {H3, nullptr};
    } else {
      auto* Uo = GetVariableRandXavier(prefix + "Uo", Shape(m, n));
      auto* Wo = GetVariableRandXavier(prefix + "Wo", Shape(n, n));

      auto* Ho1 = GEMM("", X, Uo, 0, 0);
      auto* Ho2 = GEMM("", Hin, Wo, 0, 0);
      auto* Ho3 = Add("", Ho1, Ho2);
      auto* Ho4 = Sigmoid("", Ho3);

      auto* H4 = Tanh("", H3);
      auto* H5 = Mul("", H4, Ho4);
      if (mask == 2) {
        return {nullptr, H5};
      } else {
        return {H3, H5};
      }
    }
  } else {
    auto* Hi1 = GEMM("", X, Ui, 0, 0);
    auto* Hi4 = Sigmoid("", Hi1);

    auto* Hc1 = GEMM("", X, Uc, 0, 0);
    auto* Hc4 = Tanh("", Hc1);

    auto* H2 = Mul("", Hc4, Hi4);
    if (mask == 1) {
      return {H2, nullptr};
    } else {
      auto* Uo = GetVariableRandXavier(prefix + "Uo", Shape(m, n));

      auto* Ho1 = GEMM("", X, Uo, 0, 0);
      auto* Ho4 = Sigmoid("", Ho1);

      auto* H4 = Tanh("", H2);
      auto* H5 = Mul("", H4, Ho4);
      if (mask == 2) {
        return {nullptr, H5};
      } else {
        return {H2, H5};
      }
    }
  }
}

GraphNode* GRUCell(const std::string& prefix, GraphNode* X, GraphNode* Hin,
                   int n) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int batch = X->shape()[0];
  int m = X->shape()[1];
  auto* Uz = GetVariableRandXavier(prefix + "Uz", Shape(m, n));
  auto* Uh = GetVariableRandXavier(prefix + "Uh", Shape(m, n));
  auto* ONE = ConstantScalar("", 1);
  if (Hin) {
    DXCHECK_THROW(Hin->shape().same_shape(batch, n));
    auto* Ur = GetVariableRandXavier(prefix + "Ur", Shape(m, n));
    auto* Wz = GetVariableRandXavier(prefix + "Wz", Shape(n, n));
    auto* Wr = GetVariableRandXavier(prefix + "Wr", Shape(n, n));
    auto* Wh = GetVariableRandXavier(prefix + "Wh", Shape(n, n));

    auto* Hz1 = GEMM("", X, Uz, 0, 0);
    auto* Hz2 = GEMM("", Hin, Wz, 0, 0);
    auto* Hz3 = Add("", Hz1, Hz2);
    auto* Hz4 = Sigmoid("", Hz3);

    auto* Hr1 = GEMM("", X, Ur, 0, 0);
    auto* Hr2 = GEMM("", Hin, Wr, 0, 0);
    auto* Hr3 = Add("", Hr1, Hr2);
    auto* Hr4 = Sigmoid("", Hr3);

    auto* Hh1 = GEMM("", X, Uh, 0, 0);
    auto* HHr = Mul("", Hin, Hr4);
    auto* Hh2 = GEMM("", HHr, Wh, 0, 0);
    auto* Hh3 = Add("", Hh1, Hh2);
    auto* Hh4 = Tanh("", Hh3);

    auto* H1 = BroadcastSub("", ONE, Hz4);
    auto* H2 = Mul("", H1, Hh4);
    auto* H3 = Mul("", Hz4, Hin);
    auto* H4 = Add("", H2, H3);
    return H4;
  } else {
    auto* Hz1 = GEMM("", X, Uz, 0, 0);
    auto* Hz4 = Sigmoid("", Hz1);

    auto* Hh1 = GEMM("", X, Uh, 0, 0);
    auto* Hh4 = Tanh("", Hh1);

    auto* H1 = BroadcastSub("", ONE, Hz4);
    auto* H2 = Mul("", H1, Hh4);
    return H2;
  }
}

std::vector<GraphNode*> Split(const std::string& prefix, GraphNode* X, int axis,
                              int n) {
  DXCHECK_THROW(X->shape().real_axis(&axis));
  int k = X->shape()[axis];
  DXCHECK_THROW(k % n == 0);
  int split_dim = k / n;

  std::vector<GraphNode*> nodes(n);
  std::string name;
  int begin_index = 0;
  int end_index = 0;
  for (int i = 0; i < n; ++i) {
    name = prefix + std::to_string(i);
    begin_index = end_index;
    end_index += split_dim;
    nodes[i] = SubscriptRange(name, X, axis, begin_index, end_index);
  }
  return nodes;
}

std::vector<GraphNode*> Split(GraphNode* X, int axis, int n) {
  DXCHECK_THROW(X->shape().real_axis(&axis));
  int k = X->shape()[axis];
  DXCHECK_THROW(k % n == 0);
  int split_dim = k / n;

  std::vector<GraphNode*> nodes(n);
  int begin_index = 0;
  int end_index = 0;
  for (int i = 0; i < n; ++i) {
    begin_index = end_index;
    end_index += split_dim;
    nodes[i] = SubscriptRange("", X, axis, begin_index, end_index);
  }
  return nodes;
}

std::vector<GraphNode*> Split(const std::string& prefix, GraphNode* X, int axis,
                              const std::vector<int>& split_dims) {
  DXCHECK_THROW(X->shape().real_axis(&axis));
  int k = X->shape()[axis];
  int n = (int)split_dims.size();
  int total_split_dim = 0;
  for (int split_dim : split_dims) {
    total_split_dim += split_dim;
  }
  DXCHECK_THROW(k == total_split_dim);

  std::vector<GraphNode*> nodes(n);
  std::string name;
  int begin_index = 0;
  int end_index = 0;
  for (int i = 0; i < n; ++i) {
    name = prefix + std::to_string(i);
    begin_index = end_index;
    end_index += split_dims[i];
    nodes[i] = SubscriptRange(name, X, axis, begin_index, end_index);
  }
  return nodes;
}

std::vector<GraphNode*> Split(GraphNode* X, int axis,
                              const std::vector<int>& split_dims) {
  DXCHECK_THROW(X->shape().real_axis(&axis));
  int k = X->shape()[axis];
  int n = (int)split_dims.size();
  int total_split_dim = 0;
  for (int split_dim : split_dims) {
    total_split_dim += split_dim;
  }
  DXCHECK_THROW(k == total_split_dim);

  std::vector<GraphNode*> nodes(n);
  int begin_index = 0;
  int end_index = 0;
  for (int i = 0; i < n; ++i) {
    begin_index = end_index;
    end_index += split_dims[i];
    nodes[i] = SubscriptRange("", X, axis, begin_index, end_index);
  }
  return nodes;
}

GraphNode* BatchNorm(const std::string& prefix, GraphNode* X,
                     double moving_decay) {
  DXCHECK_THROW(X->shape().rank() >= 2);
  int m = X->shape().total_dim() / X->shape()[0];
  auto* gamma = GetVariableOnes(prefix + "gamma", Shape(m));
  auto* beta = GetVariableZeros(prefix + "beta", Shape(m));
  auto* mean = GetVariableZeros(prefix + "mean", Shape(m));
  mean->set_need_grad(0);
  auto* var = GetVariableOnes(prefix + "var", Shape(m));
  var->set_need_grad(0);
  return new BatchNormNode("", X, gamma, beta, mean, var, moving_decay);
}

/************************************************************************/
/* target creator */
/************************************************************************/
std::vector<GraphNode*> BinaryClassificationTarget(const std::string& prefix,
                                                   GraphNode* X, int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(X->shape()[1] == 1);
  auto* Y = GetY(1);
  auto* L = SigmoidBCELoss(prefix + "L", X, Y);
  auto* P = Sigmoid(prefix + "P", X);
  if (has_w) {
    auto* W = GetW(1);
    auto* WL = Mul(prefix + "WL", L, W);
    auto* WM = ReduceMean(prefix + "WM", WL);
    return {WM, P};
  } else {
    auto* M = ReduceMean(prefix + "M", L);
    return {M, P};
  }
}

std::vector<GraphNode*> BinaryClassificationTarget(GraphNode* X, int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(X->shape()[1] == 1);
  auto* Y = GetY(1);
  auto* L = SigmoidBCELoss("", X, Y);
  auto* P = Sigmoid("", X);
  if (has_w) {
    auto* W = GetW(1);
    auto* WL = Mul("", L, W);
    auto* WM = ReduceMean("", WL);
    return {WM, P};
  } else {
    auto* M = ReduceMean("", L);
    return {M, P};
  }
}

}  // namespace deepx_core
