// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

/************************************************************************/
/* GroupEmbeddingLookup */
/************************************************************************/
namespace {

struct GroupEmbeddingLookupAux {
  Shape Z;
  std::vector<int> Zoffset;  // indexed by group id

  int GetZoffset(uint16_t group_id) const noexcept {
    if ((size_t)group_id >= Zoffset.size()) {
      return -1;
    }
    return Zoffset[(size_t)group_id];
  }
};

bool GroupEmbeddingLookupPrepare(int Xrow,
                                 const std::vector<uint16_t>& group_ids,
                                 const std::vector<const Shape*>& W,
                                 GroupEmbeddingLookupAux* aux) {
  if (group_ids.size() != W.size()) {
    DXERROR("Invalid group_ids and W: inconsistent size %zu vs %zu.",
            group_ids.size(), W.size());
    return false;
  }

  uint16_t max_group_id =
      *std::max_element(group_ids.begin(), group_ids.end()) + 1;
  int n = 0;
  aux->Zoffset.assign(max_group_id, -1);
  for (size_t i = 0; i < group_ids.size(); ++i) {
    if (!W[i]->is_rank(2)) {
      DXERROR("Invalid W: rank of each W %d must be 2.", W[i]->rank());
      return false;
    }
    aux->Zoffset[group_ids[i]] = n;
    n += W[i]->dim(1);
  }
  aux->Z.resize(Xrow, n);
  return true;
}

bool GroupEmbeddingLookupInferShape(int Xrow,
                                    const std::vector<uint16_t>& group_ids,
                                    const std::vector<const Shape*>& W,
                                    Shape* Z) {
  GroupEmbeddingLookupAux aux;
  if (!GroupEmbeddingLookupPrepare(Xrow, group_ids, W, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T, typename I>
void GroupEmbeddingLookup(const CSRMatrix<T, I>& X,
                          const std::vector<const Tensor<T>*>& W, Tensor<T>* Z,
                          const GroupEmbeddingLookupAux& aux) noexcept {
  int n = Z->dim(1);
  T* _Z = Z->data();
  Z->zeros();
  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      I j = CSR_COL(X);
      T Xij = CSR_VALUE(X);
      uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
      int Zoffset = aux.GetZoffset(group_id);
      if (Zoffset < 0) {
        continue;
      }

      const auto& _W = *W[group_id];
      int Wrow = _W.dim(0);
      int Wcol = _W.dim(1);
      j = j % Wrow;
      if (Wcol == 1) {
        _Z[Zoffset] += Xij * _W.data(j);
      } else {
        LLMath<T>::axpy(Wcol, Xij, _W.data() + j * Wcol, _Z + Zoffset);
      }
    }
    _Z += n;
  }
}

template <typename T, typename I>
void GroupEmbeddingLookupBackward(const CSRMatrix<T, I>& X,
                                  const std::vector<const Tensor<T>*>& W,
                                  const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                                  std::vector<SparseRowMatrix<T, I>*>* gW,
                                  const GroupEmbeddingLookupAux& aux) noexcept {
  int n = gZ.dim(1);
  const T* _gZ = gZ.data();
  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      I j = CSR_COL(X);
      T Xij = CSR_VALUE(X);
      uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
      int Zoffset = aux.GetZoffset(group_id);
      if (Zoffset < 0) {
        continue;
      }

      const auto& _W = *W[group_id];
      int Wrow = _W.dim(0);
      int Wcol = _W.dim(1);
      j = j % Wrow;
      auto* _gW = (*gW)[group_id];
      if (_gW) {
        if (Wcol == 1) {
          _gW->get_scalar_no_init(j) += Xij * _gZ[Zoffset];
        } else {
          LLMath<T>::axpy(Wcol, Xij, _gZ + Zoffset, _gW->get_row_no_init(j));
        }
      }
    }
    _gZ += n;
  }
}

template <typename T, typename I>
void GroupSparseEmbeddingLookup(
    const CSRMatrix<T, I>& X,
    const std::vector<const SparseRowMatrix<T, I>*>& W, Tensor<T>* Z,
    const GroupEmbeddingLookupAux& aux) noexcept {
  int n = Z->dim(1);
  T* _Z = Z->data();
  Z->zeros();
  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      I j = CSR_COL(X);
      T Xij = CSR_VALUE(X);
      uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
      int Zoffset = aux.GetZoffset(group_id);
      if (Zoffset < 0) {
        continue;
      }

      const auto& _W = *W[group_id];
      int Wcol = _W.col();
      if (Wcol == 1) {
        _Z[Zoffset] += Xij * _W.get_scalar_no_init(j);
      } else {
        const T* Wj = _W.get_row_no_init(j);
        if (Wj) {
          LLMath<T>::axpy(Wcol, Xij, Wj, _Z + Zoffset);
        }
      }
    }
    _Z += n;
  }
}

template <typename T, typename I>
void GroupSparseEmbeddingLookupBackward(
    const CSRMatrix<T, I>& X,
    const std::vector<const SparseRowMatrix<T, I>*>& /*W*/,
    const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
    std::vector<SparseRowMatrix<T, I>*>* gW,
    const GroupEmbeddingLookupAux& aux) noexcept {
  int n = gZ.dim(1);
  const T* _gZ = gZ.data();
  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      I j = CSR_COL(X);
      T Xij = CSR_VALUE(X);
      uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
      int Zoffset = aux.GetZoffset(group_id);
      if (Zoffset < 0) {
        continue;
      }

      auto* _gW = (*gW)[group_id];
      if (_gW) {
        int Wcol = _gW->col();
        if (Wcol == 1) {
          _gW->get_scalar_no_init(j) += Xij * _gZ[Zoffset];
        } else {
          LLMath<T>::axpy(Wcol, Xij, _gZ + Zoffset, _gW->get_row_no_init(j));
        }
      }
    }
    _gZ += n;
  }
}

}  // namespace

GroupEmbeddingLookupNode::GroupEmbeddingLookupNode(
    std::string name, GraphNode* X, const std::vector<GraphNode*>& W,
    std::vector<uint16_t> group_ids)
    : GraphNode(std::move(name)), group_ids_(std::move(group_ids)) {
  DXCHECK_THROW(X->node_type() == GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_CSR);
  DXCHECK_THROW(!W.empty());
  int W_node_type = W[0]->node_type();
  int W_tensor_type = W[0]->tensor_type();
  DXCHECK_THROW(W_tensor_type == TENSOR_TYPE_TSR ||
                W_tensor_type == TENSOR_TYPE_SRM);
  for (const GraphNode* _W : W) {
    DXCHECK_THROW(_W->node_type() == W_node_type);
    DXCHECK_THROW(_W->tensor_type() == W_tensor_type);
  }
  DXCHECK_THROW(group_ids_.size() == W.size());

  input_.reserve(1 + W.size());
  input_.emplace_back(X);
  input_.insert(input_.end(), W.begin(), W.end());
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (X->shape().is_rank(2) && HasShape(W)) {
    std::vector<const Shape*> Wshape(group_ids_.size());
    for (size_t i = 0; i < group_ids_.size(); ++i) {
      Wshape[i] = &W[i]->shape();
    }
    (void)GroupEmbeddingLookupInferShape(X->shape()[0], group_ids_, Wshape,
                                         &shape_);
  }
}

class GroupEmbeddingLookupOp : public OpImpl {
 private:
  std::vector<uint16_t> group_ids_;
  uint16_t max_group_id_ = 0;
  const csr_t* X_ = nullptr;
  int W_node_type_ = GRAPH_NODE_TYPE_NONE;
  int W_tensor_type_ = TENSOR_TYPE_NONE;
  int Wsize_ = 0;
  std::vector<const GraphNode*> Wnode1_;  // indexed by group id
  std::vector<const GraphNode*> Wnode2_;  // indexed by i
  std::vector<const Shape*> Wshape_;      // indexed by i
  std::vector<const tsr_t*> Wtsr_;        // indexed by group id
  std::vector<const srm_t*> Wsrm_;        // indexed by group id
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  std::vector<srm_t*> gW_;  // indexed by group id
  GroupEmbeddingLookupAux aux_;

 public:
  DEFINE_OP_LIKE(GroupEmbeddingLookupOp);

  void InitForward() override {
    DXCHECK_THROW(!node_->input(0)->need_grad());
    group_ids_ = ((const GroupEmbeddingLookupNode*)node_)->group_ids();
    max_group_id_ = *std::max_element(group_ids_.begin(), group_ids_.end()) + 1;
    X_ = GetPtrCSR(node_->input(0));
    W_node_type_ = node_->input(1)->node_type();
    W_tensor_type_ = node_->input(1)->tensor_type();
    Wsize_ = node_->input_size() - 1;
    Wnode1_.assign(max_group_id_, nullptr);
    Wnode2_.assign(Wsize_, nullptr);
    for (int i = 0; i < Wsize_; ++i) {
      const GraphNode* Wnode = node_->input(i + 1);
      Wnode1_[group_ids_[i]] = Wnode;
      Wnode2_[i] = Wnode;
    }
    Wshape_.assign(Wsize_, nullptr);
    Wtsr_.clear();
    Wsrm_.clear();
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        Wtsr_.assign(max_group_id_, nullptr);
        for (int i = 0; i < Wsize_; ++i) {
          tsr_t* W = GetPtrTSR(Wnode2_[i]);
          Wtsr_[group_ids_[i]] = W;
          Wshape_[i] = &W->shape();
        }
        break;
      case TENSOR_TYPE_SRM:
        Wsrm_.assign(max_group_id_, nullptr);
        for (int i = 0; i < Wsize_; ++i) {
          srm_t* W = GetPtrSRM(Wnode2_[i]);
          Wsrm_[group_ids_[i]] = W;
          Wshape_[i] = &W->shape();
        }
        break;
    }
    DXCHECK_THROW(
        GroupEmbeddingLookupPrepare(X_->row(), group_ids_, Wshape_, &aux_));
    Z_ = InitHiddenTSR(node_, aux_.Z);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gW_.assign(max_group_id_, nullptr);
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        for (int i = 0; i < Wsize_; ++i) {
          gW_[group_ids_[i]] =
              InitGradSRM(Wnode2_[i], Wtsr_[group_ids_[i]]->shape());
        }
        break;
      case TENSOR_TYPE_SRM:
        for (int i = 0; i < Wsize_; ++i) {
          gW_[group_ids_[i]] =
              InitGradSRM(Wnode2_[i], Wsrm_[group_ids_[i]]->shape());
        }
        break;
    }
  }

  void Forward() override {
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        GroupEmbeddingLookup(*X_, Wtsr_, Z_, aux_);
        break;
      case TENSOR_TYPE_SRM:
        GroupSparseEmbeddingLookup(*X_, Wsrm_, Z_, aux_);
        break;
    }
  }

  void Backward() override {
    if (gZ_) {
      switch (W_tensor_type_) {
        case TENSOR_TYPE_TSR:
          GroupEmbeddingLookupBackward(*X_, Wtsr_, *Z_, *gZ_, &gW_, aux_);
          break;
        case TENSOR_TYPE_SRM:
          GroupSparseEmbeddingLookupBackward(*X_, Wsrm_, *Z_, *gZ_, &gW_, aux_);
          break;
      }
    }
  }

  void GetPullRequest(PullRequest* pull_request) const override {
    if (W_node_type_ != GRAPH_NODE_TYPE_PARAM) {
      return;
    }

    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR: {
        for (int i = 0; i < Wsize_; ++i) {
          pull_request->tsr_set.emplace(Wnode2_[i]->name());
        }
      } break;
      case TENSOR_TYPE_SRM:
        CSR_FOR_EACH_ROW(*X_, i) {
          CSR_FOR_EACH_COL(*X_, i) {
            int_t j = CSR_COL(*X_);
            uint16_t group_id = ll_sparse_tensor_t::get_group_id(j);
            int Zoffset = aux_.GetZoffset(group_id);
            if (Zoffset < 0) {
              continue;
            }

            pull_request->srm_map[Wnode1_[group_id]->name()].emplace(j);
          }
        }
        break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(GroupEmbeddingLookup);

/************************************************************************/
/* GroupEmbeddingLookup2 */
/************************************************************************/
namespace {

bool GroupEmbeddingLookup2Prepare(int Xrow, const Shape& W,
                                  const std::vector<uint16_t>& group_ids,
                                  GroupEmbeddingLookupAux* aux) {
  if (!W.is_rank(2)) {
    DXERROR("Invalid W: rank of W %d must be 2.", W.rank());
    return false;
  }

  uint16_t max_group_id =
      *std::max_element(group_ids.begin(), group_ids.end()) + 1;
  int m = (int)group_ids.size();
  int n = W[1];
  aux->Zoffset.assign(max_group_id, -1);
  for (int i = 0; i < m; ++i) {
    aux->Zoffset[group_ids[i]] = i * n;
  }
  aux->Z.resize(Xrow, m * n);
  return true;
}

bool GroupEmbeddingLookup2InferShape(int Xrow, const Shape& W,
                                     const std::vector<uint16_t>& group_ids,
                                     Shape* Z) {
  GroupEmbeddingLookupAux aux;
  if (!GroupEmbeddingLookup2Prepare(Xrow, W, group_ids, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T, typename I>
void GroupEmbeddingLookup2(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                           Tensor<T>* Z,
                           const GroupEmbeddingLookupAux& aux) noexcept {
  int Wrow = W.dim(0);
  int Wcol = W.dim(1);
  int n = Z->dim(1);
  T* _Z = Z->data();
  Z->zeros();
  if (Wcol == 1) {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        j = j % Wrow;
        _Z[Zoffset] += Xij * W.data(j);
      }
      _Z += n;
    }
  } else {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        j = j % Wrow;
        LLMath<T>::axpy(Wcol, Xij, W.data() + j * Wcol, _Z + Zoffset);
      }
      _Z += n;
    }
  }
}

template <typename T, typename I>
void GroupEmbeddingLookup2Backward(
    const CSRMatrix<T, I>& X, const Tensor<T>& W, const Tensor<T>& /*Z*/,
    const Tensor<T>& gZ, SparseRowMatrix<T, I>* gW,
    const GroupEmbeddingLookupAux& aux) noexcept {
  int Wrow = W.dim(0);
  int Wcol = W.dim(1);
  int n = gZ.dim(1);
  const T* _gZ = gZ.data();
  if (Wcol == 1) {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        j = j % Wrow;
        gW->get_scalar_no_init(j) += Xij * _gZ[Zoffset];
      }
      _gZ += n;
    }
  } else {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        j = j % Wrow;
        LLMath<T>::axpy(Wcol, Xij, _gZ + Zoffset, gW->get_row_no_init(j));
      }
      _gZ += n;
    }
  }
}

template <typename T, typename I>
void GroupSparseEmbeddingLookup2(const CSRMatrix<T, I>& X,
                                 const SparseRowMatrix<T, I>& W, Tensor<T>* Z,
                                 const GroupEmbeddingLookupAux& aux) noexcept {
  int Wcol = W.col();
  int n = Z->dim(1);
  T* _Z = Z->data();
  Z->zeros();
  if (Wcol == 1) {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        _Z[Zoffset] += Xij * W.get_scalar_no_init(j);
      }
      _Z += n;
    }
  } else {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        const T* Wj = W.get_row_no_init(j);
        if (Wj) {
          LLMath<T>::axpy(Wcol, Xij, Wj, _Z + Zoffset);
        }
      }
      _Z += n;
    }
  }
}

template <typename T, typename I>
void GroupSparseEmbeddingLookup2Backward(
    const CSRMatrix<T, I>& X, const SparseRowMatrix<T, I>& /*W*/,
    const Tensor<T>& /*Z*/, const Tensor<T>& gZ, SparseRowMatrix<T, I>* gW,
    const GroupEmbeddingLookupAux& aux) noexcept {
  int Wcol = gW->col();
  int n = gZ.dim(1);
  const T* _gZ = gZ.data();
  if (Wcol == 1) {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        gW->get_scalar_no_init(j) += Xij * _gZ[Zoffset];
      }
      _gZ += n;
    }
  } else {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        I j = CSR_COL(X);
        T Xij = CSR_VALUE(X);
        uint16_t group_id = LLSparseTensor<T, I>::get_group_id(j);
        int Zoffset = aux.GetZoffset(group_id);
        if (Zoffset < 0) {
          continue;
        }

        LLMath<T>::axpy(Wcol, Xij, _gZ + Zoffset, gW->get_row_no_init(j));
      }
      _gZ += n;
    }
  }
}

}  // namespace

GroupEmbeddingLookup2Node::GroupEmbeddingLookup2Node(
    std::string name, GraphNode* X, GraphNode* W,
    std::vector<uint16_t> group_ids)
    : GraphNode(std::move(name)), group_ids_(std::move(group_ids)) {
  DXCHECK_THROW(X->node_type() == GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_CSR);
  DXCHECK_THROW(W->tensor_type() == TENSOR_TYPE_TSR ||
                W->tensor_type() == TENSOR_TYPE_SRM);
  DXCHECK_THROW(!group_ids_.empty());

  input_ = {X, W};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (X->shape().is_rank(2) && !W->shape().empty()) {
    (void)GroupEmbeddingLookup2InferShape(X->shape()[0], W->shape(), group_ids_,
                                          &shape_);
  }
}

class GroupEmbeddingLookup2Op : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Wnode_ = nullptr;
  int W_node_type_ = GRAPH_NODE_TYPE_NONE;
  int W_tensor_type_ = TENSOR_TYPE_NONE;
  const csr_t* X_ = nullptr;
  const tsr_t* Wtsr_ = nullptr;
  const srm_t* Wsrm_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  srm_t* gW_ = nullptr;
  GroupEmbeddingLookupAux aux_;

 public:
  DEFINE_OP_LIKE(GroupEmbeddingLookup2Op);

  void InitForward() override {
    const std::vector<uint16_t>& group_ids =
        ((const GroupEmbeddingLookup2Node*)node_)->group_ids();
    Xnode_ = node_->input(0);
    DXCHECK_THROW(!Xnode_->need_grad());
    Wnode_ = node_->input(1);
    W_node_type_ = Wnode_->node_type();
    W_tensor_type_ = Wnode_->tensor_type();
    X_ = GetPtrCSR(Xnode_);
    Wtsr_ = nullptr;
    Wsrm_ = nullptr;
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        Wtsr_ = GetPtrTSR(Wnode_);
        DXCHECK_THROW(GroupEmbeddingLookup2Prepare(X_->row(), Wtsr_->shape(),
                                                   group_ids, &aux_));
        break;
      case TENSOR_TYPE_SRM:
        Wsrm_ = GetPtrSRM(Wnode_);
        DXCHECK_THROW(GroupEmbeddingLookup2Prepare(X_->row(), Wsrm_->shape(),
                                                   group_ids, &aux_));
        break;
    }
    Z_ = InitHiddenTSR(node_, aux_.Z);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        gW_ = InitGradSRM(Wnode_, Wtsr_->shape());
        break;
      case TENSOR_TYPE_SRM:
        gW_ = InitGradSRM(Wnode_, Wsrm_->shape());
        break;
    }
  }

  void Forward() override {
    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR:
        GroupEmbeddingLookup2(*X_, *Wtsr_, Z_, aux_);
        break;
      case TENSOR_TYPE_SRM:
        GroupSparseEmbeddingLookup2(*X_, *Wsrm_, Z_, aux_);
        break;
    }
  }

  void Backward() override {
    if (gW_) {
      switch (W_tensor_type_) {
        case TENSOR_TYPE_TSR:
          GroupEmbeddingLookup2Backward(*X_, *Wtsr_, *Z_, *gZ_, gW_, aux_);
          break;
        case TENSOR_TYPE_SRM:
          GroupSparseEmbeddingLookup2Backward(*X_, *Wsrm_, *Z_, *gZ_, gW_,
                                              aux_);
          break;
      }
    }
  }

  void GetPullRequest(PullRequest* pull_request) const override {
    if (W_node_type_ != GRAPH_NODE_TYPE_PARAM) {
      return;
    }

    switch (W_tensor_type_) {
      case TENSOR_TYPE_TSR: {
        pull_request->tsr_set.emplace(Wnode_->name());
      } break;
      case TENSOR_TYPE_SRM: {
        auto& id_set = pull_request->srm_map[Wnode_->name()];
        CSR_FOR_EACH_ROW(*X_, i) {
          CSR_FOR_EACH_COL(*X_, i) {
            int_t j = CSR_COL(*X_);
            uint16_t group_id = ll_sparse_tensor_t::get_group_id(j);
            int Zoffset = aux_.GetZoffset(group_id);
            if (Zoffset < 0) {
              continue;
            }

            id_set.emplace(j);
          }
        }
      } break;
    }
  }
};

GRAPH_NODE_OP_REGISTER(GroupEmbeddingLookup2);

}  // namespace deepx_core
