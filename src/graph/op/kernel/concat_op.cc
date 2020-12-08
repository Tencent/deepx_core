// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct ConcatAux {
  Shape Z;
  int m = 0;             // pre axis dim
  std::vector<int> Xnk;  // axis & post axis dim of X
  std::vector<int> Zoffset;
  int Zcol = 0;
};

bool ConcatPrepare(const std::vector<const Shape*>& X, int axis,
                   ConcatAux* aux) {
  if (X.empty()) {
    DXERROR("Invalid X: X is empty.");
    return false;
  }

  const Shape& X0 = *X[0];
  int rank = X0.rank();
  if (rank == 0) {
    DXERROR("Invalid X: rank of X[0] is zero.");
    return false;
  }

  if (!X0.real_axis(&axis)) {
    DXERROR("Invalid axis: %d.", axis);
    return false;
  }

  size_t size = X.size();
  int k = 0;
  for (size_t i = 0; i < size; ++i) {
    const Shape& Xi = *X[i];
    if (Xi.rank() != rank) {
      DXERROR("Invalid X: inconsistent rank %d vs %d.", Xi.rank(), rank);
      return false;
    }
    for (int j = 0; j < rank; ++j) {
      if (j == axis) {
        k += Xi[j];
      } else if (Xi[j] != X0[j]) {
        DXERROR("Invalid X: inconsistent dim %d vs %d.", Xi[j], X0[j]);
        return false;
      }
    }
  }

  int Zdims[SHAPE_MAX_RANK];
  int m = 1;
  int n = 1;
  int Xnk, Zcol = 0;
  for (int j = 0; j < axis; ++j) {
    Zdims[j] = X0[j];
    m *= X0[j];
  }
  Zdims[axis] = k;
  for (int j = axis + 1; j < rank; ++j) {
    Zdims[j] = X0[j];
    n *= X0[j];
  }

  aux->Z.assign(&Zdims[0], &Zdims[rank]);
  aux->m = m;
  aux->Xnk.resize(size);
  aux->Zoffset.resize(size);
  for (size_t i = 0; i < size; ++i) {
    const Shape& Xi = *X[i];
    Xnk = Xi[axis] * n;
    aux->Xnk[i] = Xnk;
    aux->Zoffset[i] = Zcol;
    Zcol += Xnk;
  }
  aux->Zcol = Zcol;
  return true;
}

bool ConcatInferShape(const std::vector<const Shape*>& X, int axis, Shape* Z) {
  ConcatAux aux;
  if (!ConcatPrepare(X, axis, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void Concat(const std::vector<const Tensor<T>*>& X, Tensor<T>* Z,
            const ConcatAux& aux) noexcept {
  int m = aux.m;
  for (size_t i = 0; i < X.size(); ++i) {
    int Xnk = aux.Xnk[i];
    const T* _X = X[i]->data();
    T* _Z = Z->data() + aux.Zoffset[i];
    for (int j = 0; j < m; ++j) {
      LLMath<T>::copy(Xnk, _X, _Z);
      _X += Xnk;
      _Z += aux.Zcol;
    }
  }
}

template <typename T>
void ConcatBackward(const std::vector<const Tensor<T>*>& /*X*/,
                    const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                    std::vector<Tensor<T>*>* gX,
                    const ConcatAux& aux) noexcept {
  int m = aux.m;
  for (size_t i = 0; i < gX->size(); ++i) {
    if ((*gX)[i]) {
      int Xnk = aux.Xnk[i];
      const T* _gZ = gZ.data() + aux.Zoffset[i];
      T* _gX = (*gX)[i]->data();
      for (int j = 0; j < m; ++j) {
        LLMath<T>::add(Xnk, _gX, _gZ, _gX);
        _gZ += aux.Zcol;
        _gX += Xnk;
      }
    }
  }
}

}  // namespace

ConcatNode::ConcatNode(std::string name, std::vector<GraphNode*> X, int axis)
    : GraphNode(std::move(name)), axis_(axis) {
  DXCHECK_THROW(!X.empty());
  for (const GraphNode* _X : X) {
    DXCHECK_THROW(_X->tensor_type() == TENSOR_TYPE_TSR);
  }

  input_ = std::move(X);
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (HasShape(input_)) {
    std::vector<const Shape*> Xshape(input_size());
    for (int i = 0; i < input_size(); ++i) {
      Xshape[i] = &input_[i]->shape();
    }
    (void)ConcatInferShape(Xshape, axis_, &shape_);
  }
}

class ConcatOp : public OpImpl {
 private:
  std::vector<const Shape*> Xshape_;
  std::vector<const tsr_t*> X_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  std::vector<tsr_t*> gX_;
  ConcatAux aux_;

 public:
  DEFINE_OP_LIKE(ConcatOp);

  void InitForward() override {
    int input_size = node_->input_size();
    int axis = ((const ConcatNode*)node_)->axis();
    Xshape_.resize(input_size);
    X_.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
      const GraphNode* Xnode = node_->input(i);
      const tsr_t* X = GetPtrTSR(Xnode);
      Xshape_[i] = &X->shape();
      X_[i] = X;
    }
    DXCHECK_THROW(ConcatPrepare(Xshape_, axis, &aux_));
    Z_ = InitHiddenTSR(node_, aux_.Z);
  }

  void InitBackward() override {
    int input_size = node_->input_size();
    gZ_ = GetGradPtrTSR(node_);
    gX_.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
      const GraphNode* Xnode = node_->input(i);
      gX_[i] = InitGradTSR(Xnode, X_[i]->shape());
    }
  }

  void Forward() override { Concat(X_, Z_, aux_); }

  void Backward() override {
    if (gZ_) {
      ConcatBackward(X_, *Z_, *gZ_, &gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Concat);

}  // namespace deepx_core
