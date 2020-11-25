// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#include "broadcast.h"

namespace deepx_core {

/************************************************************************/
/* Broadcast */
/************************************************************************/
namespace {

bool BroadcastInferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
  BroadcastAux aux;
  if (!BroadcastPrepare(X, Y, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T, class Meta>
void _BroadcastRank1(const T* X, const T* Y, T* Z,
                     const BroadcastAux& aux) noexcept {
  if (aux.vectorization) {
    Meta::Forward(aux.Z[0], X, Y, Z);
  } else {
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      Meta::Forward(X, Y, Z);
      X += aux.Xstrides[0];
      Y += aux.Ystrides[0];
      Z += aux.Zstrides[0];
    }
  }
}

template <typename T, class Meta>
void _BroadcastRank2(const T* X, const T* Y, T* Z,
                     const BroadcastAux& aux) noexcept {
  const T* _X;
  const T* _Y;
  T* _Z;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    if (aux.vectorization) {
      Meta::Forward(aux.Z[1], X, Y, Z);
    } else {
      _X = X;
      _Y = Y;
      _Z = Z;
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        Meta::Forward(_X, _Y, _Z);
        _X += aux.Xstrides[1];
        _Y += aux.Ystrides[1];
        _Z += aux.Zstrides[1];
      }
    }
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    Z += aux.Zstrides[0];
  }
}

template <typename T, class Meta>
void _BroadcastRank3(const T* X, const T* Y, T* Z,
                     const BroadcastAux& aux) noexcept {
  const T* _X;
  const T* _Y;
  T* _Z;
  const T* __X;
  const T* __Y;
  T* __Z;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    _X = X;
    _Y = Y;
    _Z = Z;
    for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
      if (aux.vectorization) {
        Meta::Forward(aux.Z[2], _X, _Y, _Z);
      } else {
        __X = _X;
        __Y = _Y;
        __Z = _Z;
        for (int i2 = 0; i2 < aux.Z[2]; ++i2) {
          Meta::Forward(__X, __Y, __Z);
          __X += aux.Xstrides[2];
          __Y += aux.Ystrides[2];
          __Z += aux.Zstrides[2];
        }
      }
      _X += aux.Xstrides[1];
      _Y += aux.Ystrides[1];
      _Z += aux.Zstrides[1];
    }
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    Z += aux.Zstrides[0];
  }
}

template <typename T, class Meta>
void _BroadcastFallback(int level, const T* X, const T* Y, T* Z,
                        const BroadcastAux& aux) noexcept {
  if (level < aux.Z.rank() - 1) {
    for (int i = 0; i < aux.Z[level]; ++i) {
      _BroadcastFallback<T, Meta>(level + 1, X, Y, Z, aux);
      X += aux.Xstrides[level];
      Y += aux.Ystrides[level];
      Z += aux.Zstrides[level];
    }
  } else {
    if (aux.vectorization) {
      Meta::Forward(aux.Z[level], X, Y, Z);
    } else {
      for (int i = 0; i < aux.Z[level]; ++i) {
        Meta::Forward(X, Y, Z);
        X += aux.Xstrides[level];
        Y += aux.Ystrides[level];
        Z += aux.Zstrides[level];
      }
    }
  }
}

template <typename T, class Meta>
void _Broadcast(const T* X, const T* Y, T* Z,
                const BroadcastAux& aux) noexcept {
  if (aux.XY_same_shape) {
    Meta::Forward(aux.Z_total_dim, X, Y, Z);
  } else if (aux.Z.is_rank(1)) {
    _BroadcastRank1<T, Meta>(X, Y, Z, aux);
  } else if (aux.Z.is_rank(2)) {
    _BroadcastRank2<T, Meta>(X, Y, Z, aux);
  } else if (aux.Z.is_rank(3)) {
    _BroadcastRank3<T, Meta>(X, Y, Z, aux);
  } else {
    _BroadcastFallback<T, Meta>(0, X, Y, Z, aux);
  }
}

template <typename T, class Meta>
void _BroadcastBackwardRank1(const T* X, const T* Y, const T* Z, const T* gZ,
                             T* gX, T* gY, const BroadcastAux& aux) noexcept {
  if (aux.vectorization) {
    Meta::Backward(aux.Z[0], X, Y, Z, gZ, gX, gY);
  } else {
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      Meta::Backward(X, Y, Z, gZ, gX, gY);
      X += aux.Xstrides[0];
      Y += aux.Ystrides[0];
      Z += aux.Zstrides[0];
      gZ += aux.Zstrides[0];
      if (gX) {
        gX += aux.Xstrides[0];
      }
      if (gY) {
        gY += aux.Ystrides[0];
      }
    }
  }
}

template <typename T, class Meta>
void _BroadcastBackwardRank2(const T* X, const T* Y, const T* Z, const T* gZ,
                             T* gX, T* gY, const BroadcastAux& aux) noexcept {
  const T* _X;
  const T* _Y;
  const T* _Z;
  const T* _gZ;
  T* _gX;
  T* _gY;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    if (aux.vectorization) {
      Meta::Backward(aux.Z[1], X, Y, Z, gZ, gX, gY);
    } else {
      _X = X;
      _Y = Y;
      _Z = Z;
      _gZ = gZ;
      _gX = gX;
      _gY = gY;
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        Meta::Backward(_X, _Y, _Z, _gZ, _gX, _gY);
        _X += aux.Xstrides[1];
        _Y += aux.Ystrides[1];
        _Z += aux.Zstrides[1];
        _gZ += aux.Zstrides[1];
        if (_gX) {
          _gX += aux.Xstrides[1];
        }
        if (_gY) {
          _gY += aux.Ystrides[1];
        }
      }
    }
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    Z += aux.Zstrides[0];
    gZ += aux.Zstrides[0];
    if (gX) {
      gX += aux.Xstrides[0];
    }
    if (gY) {
      gY += aux.Ystrides[0];
    }
  }
}

template <typename T, class Meta>
void _BroadcastBackwardRank3(const T* X, const T* Y, const T* Z, const T* gZ,
                             T* gX, T* gY, const BroadcastAux& aux) noexcept {
  const T* _X;
  const T* _Y;
  const T* _Z;
  const T* _gZ;
  T* _gX;
  T* _gY;
  const T* __X;
  const T* __Y;
  const T* __Z;
  const T* __gZ;
  T* __gX;
  T* __gY;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    _X = X;
    _Y = Y;
    _Z = Z;
    _gZ = gZ;
    _gX = gX;
    _gY = gY;
    for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
      if (aux.vectorization) {
        Meta::Backward(aux.Z[2], _X, _Y, _Z, _gZ, _gX, _gY);
      } else {
        __X = _X;
        __Y = _Y;
        __Z = _Z;
        __gZ = _gZ;
        __gX = _gX;
        __gY = _gY;
        for (int i2 = 0; i2 < aux.Z[2]; ++i2) {
          Meta::Backward(__X, __Y, __Z, __gZ, __gX, __gY);
          __X += aux.Xstrides[2];
          __Y += aux.Ystrides[2];
          __Z += aux.Zstrides[2];
          __gZ += aux.Zstrides[2];
          if (__gX) {
            __gX += aux.Xstrides[2];
          }
          if (__gY) {
            __gY += aux.Ystrides[2];
          }
        }
      }
      _X += aux.Xstrides[1];
      _Y += aux.Ystrides[1];
      _Z += aux.Zstrides[1];
      _gZ += aux.Zstrides[1];
      if (_gX) {
        _gX += aux.Xstrides[1];
      }
      if (_gY) {
        _gY += aux.Ystrides[1];
      }
    }
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    Z += aux.Zstrides[0];
    gZ += aux.Zstrides[0];
    if (gX) {
      gX += aux.Xstrides[0];
    }
    if (gY) {
      gY += aux.Ystrides[0];
    }
  }
}

template <typename T, class Meta>
void _BroadcastBackwardFallback(int level, const T* X, const T* Y, const T* Z,
                                const T* gZ, T* gX, T* gY,
                                const BroadcastAux& aux) noexcept {
  if (level < aux.Z.rank() - 1) {
    for (int i = 0; i < aux.Z[level]; ++i) {
      _BroadcastBackwardFallback<T, Meta>(level + 1, X, Y, Z, gZ, gX, gY, aux);
      X += aux.Xstrides[level];
      Y += aux.Ystrides[level];
      Z += aux.Zstrides[level];
      gZ += aux.Zstrides[level];
      if (gX) {
        gX += aux.Xstrides[level];
      }
      if (gY) {
        gY += aux.Ystrides[level];
      }
    }
  } else {
    if (aux.vectorization) {
      Meta::Backward(aux.Z[level], X, Y, Z, gZ, gX, gY);
    } else {
      for (int i = 0; i < aux.Z[level]; ++i) {
        Meta::Backward(X, Y, Z, gZ, gX, gY);
        X += aux.Xstrides[level];
        Y += aux.Ystrides[level];
        Z += aux.Zstrides[level];
        gZ += aux.Zstrides[level];
        if (gX) {
          gX += aux.Xstrides[level];
        }
        if (gY) {
          gY += aux.Ystrides[level];
        }
      }
    }
  }
}

template <typename T, class Meta>
void _BroadcastBackward(const T* X, const T* Y, const T* Z, const T* gZ, T* gX,
                        T* gY, const BroadcastAux& aux) noexcept {
  if (aux.XY_same_shape) {
    Meta::Backward(aux.Z_total_dim, X, Y, Z, gZ, gX, gY);
  } else if (aux.Z.is_rank(1)) {
    _BroadcastBackwardRank1<T, Meta>(X, Y, Z, gZ, gX, gY, aux);
  } else if (aux.Z.is_rank(2)) {
    _BroadcastBackwardRank2<T, Meta>(X, Y, Z, gZ, gX, gY, aux);
  } else if (aux.Z.is_rank(3)) {
    _BroadcastBackwardRank3<T, Meta>(X, Y, Z, gZ, gX, gY, aux);
  } else {
    _BroadcastBackwardFallback<T, Meta>(0, X, Y, Z, gZ, gX, gY, aux);
  }
}

#define DEFINE_BROADCAST_OP(name)                                             \
  template <typename T>                                                       \
  void Broadcast##name(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,  \
                       const BroadcastAux& aux) noexcept {                    \
    _Broadcast<T, Binary##name##Meta<T>>(X.data(), Y.data(), Z->data(), aux); \
  }

#define DEFINE_BROADCAST_OP_BACKWARD(name)                                \
  template <typename T>                                                   \
  void Broadcast##name##Backward(const Tensor<T>& X, const Tensor<T>& Y,  \
                                 const Tensor<T>& Z, const Tensor<T>& gZ, \
                                 Tensor<T>* gX, Tensor<T>* gY,            \
                                 const BroadcastAux& aux) noexcept {      \
    T* _gX = gX ? gX->data() : nullptr;                                   \
    T* _gY = gY ? gY->data() : nullptr;                                   \
    if (_gX || _gY) {                                                     \
      _BroadcastBackward<T, Binary##name##Meta<T>>(                       \
          X.data(), Y.data(), Z.data(), gZ.data(), _gX, _gY, aux);        \
    }                                                                     \
  }

DEFINE_BROADCAST_OP(Add)
DEFINE_BROADCAST_OP_BACKWARD(Add)
DEFINE_BROADCAST_OP(Sub)
DEFINE_BROADCAST_OP_BACKWARD(Sub)
DEFINE_BROADCAST_OP(Mul)
DEFINE_BROADCAST_OP_BACKWARD(Mul)
DEFINE_BROADCAST_OP(Div)
DEFINE_BROADCAST_OP_BACKWARD(Div)
DEFINE_BROADCAST_OP(Pow)
DEFINE_BROADCAST_OP_BACKWARD(Pow)
DEFINE_BROADCAST_OP(Max)
DEFINE_BROADCAST_OP_BACKWARD(Max)
DEFINE_BROADCAST_OP(Min)
DEFINE_BROADCAST_OP_BACKWARD(Min)
DEFINE_BROADCAST_OP(Equal)
DEFINE_BROADCAST_OP(Greater)
DEFINE_BROADCAST_OP(GreaterEqual)
DEFINE_BROADCAST_OP(Less)
DEFINE_BROADCAST_OP(LessEqual)

}  // namespace

GraphNodeBroadcastBase::GraphNodeBroadcastBase(std::string name, GraphNode* X,
                                               GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BroadcastInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class OpBroadcastBase : public OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Ynode_ = nullptr;
  const tsr_t* X_ = nullptr;
  const tsr_t* Y_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* gY_ = nullptr;
  BroadcastAux aux_;

 public:
  void InitForward() override {
    Xnode_ = node_->input(0);
    Ynode_ = node_->input(1);
    X_ = GetPtrTSR(Xnode_);
    Y_ = GetPtrTSR(Ynode_);
    DXCHECK_THROW(BroadcastPrepare(X_->shape(), Y_->shape(), &aux_));
    Z_ = InitHiddenTSR(node_, aux_.Z);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
    gY_ = InitGradTSR(Ynode_, Y_->shape());
  }
};

#define DEFINE_BROADCAST_OP1(name)                                     \
  name##Node::name##Node(std::string name, GraphNode* X, GraphNode* Y) \
      : GraphNodeBroadcastBase(std::move(name), X, Y) {}               \
                                                                       \
  class name##Op : public OpBroadcastBase {                            \
   public:                                                             \
    DEFINE_OP_LIKE(name##Op);                                          \
    void Forward() override { name(*X_, *Y_, Z_, aux_); }              \
    void Backward() override {                                         \
      if (gZ_) {                                                       \
        name##Backward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_);           \
      }                                                                \
    }                                                                  \
  };                                                                   \
                                                                       \
  GRAPH_NODE_OP_REGISTER(name)

#define DEFINE_BROADCAST_OP2(name)                                     \
  name##Node::name##Node(std::string name, GraphNode* X, GraphNode* Y) \
      : GraphNodeBroadcastBase(std::move(name), X, Y) {}               \
                                                                       \
  class name##Op : public OpBroadcastBase {                            \
   public:                                                             \
    DEFINE_OP_LIKE(name##Op);                                          \
    void Forward() override { name(*X_, *Y_, Z_, aux_); }              \
  };                                                                   \
                                                                       \
  GRAPH_NODE_OP_REGISTER(name)

DEFINE_BROADCAST_OP1(BroadcastAdd);
DEFINE_BROADCAST_OP1(BroadcastSub);
DEFINE_BROADCAST_OP1(BroadcastMul);
DEFINE_BROADCAST_OP1(BroadcastDiv);
DEFINE_BROADCAST_OP1(BroadcastPow);
DEFINE_BROADCAST_OP1(BroadcastMax);
DEFINE_BROADCAST_OP1(BroadcastMin);
DEFINE_BROADCAST_OP2(BroadcastEqual);
DEFINE_BROADCAST_OP2(BroadcastGreater);
DEFINE_BROADCAST_OP2(BroadcastGreaterEqual);
DEFINE_BROADCAST_OP2(BroadcastLess);
DEFINE_BROADCAST_OP2(BroadcastLessEqual);

/************************************************************************/
/* BroadcastTo */
/************************************************************************/
namespace {

bool BroadcastToPrepare(const Shape& X, const Shape& Y,
                        BroadcastAux* aux) noexcept {
  int Xrank = X.rank();
  if (Xrank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  int Yrank = Y.rank();
  if (Yrank == 0) {
    DXERROR("Invalid Y: rank of Y is zero.");
    return false;
  }

  if (Xrank > Yrank) {
    DXERROR(
        "Invalid X and Y: rank of X %d must be less than or equal to rank of Y "
        "%d.",
        Xrank, Yrank);
    return false;
  }

  int Zrank = Yrank;
  int Zrank_remain = SHAPE_MAX_RANK - Zrank;
  using ai_t = std::array<int, SHAPE_MAX_RANK>;
  ai_t X_reverse_dims, Y_reverse_dims, Z_reverse_dims;
  ai_t X_reverse_strides, Y_reverse_strides, Z_reverse_strides;
  int Xstride = 1, Ystride = 1, Zstride = 1;

  std::copy(X.rbegin(), X.rend(), X_reverse_dims.begin());
  for (int i = Xrank; i < Zrank; ++i) {
    X_reverse_dims[i] = 1;
  }

  std::copy(Y.rbegin(), Y.rend(), Y_reverse_dims.begin());

  for (int i = 0; i < Zrank; ++i) {
    if (X_reverse_dims[i] == Y_reverse_dims[i]) {
      Z_reverse_dims[i] = X_reverse_dims[i];
    } else if (X_reverse_dims[i] == 1) {
      Z_reverse_dims[i] = Y_reverse_dims[i];
    } else {
      DXERROR("Couldn't unidirectional broadcast %s to %s.",
              to_string(X).c_str(), to_string(Y).c_str());
      return false;
    }
  }
  aux->Z.assign(Z_reverse_dims.rbegin() + Zrank_remain, Z_reverse_dims.rend());

  for (int i = 0; i < Zrank; ++i) {
    if (X_reverse_dims[i] == 1) {
      X_reverse_strides[i] = 0;
    } else {
      X_reverse_strides[i] = Xstride;
      Xstride *= X_reverse_dims[i];
    }

    if (Y_reverse_dims[i] == 1) {
      Y_reverse_strides[i] = 0;
    } else {
      Y_reverse_strides[i] = Ystride;
      Ystride *= Y_reverse_dims[i];
    }

    if (Z_reverse_dims[i] == 1) {
      Z_reverse_strides[i] = 0;
    } else {
      Z_reverse_strides[i] = Zstride;
      Zstride *= Z_reverse_dims[i];
    }
  }
  aux->Xstrides.assign(X_reverse_strides.rbegin() + Zrank_remain,
                       X_reverse_strides.rend());
  aux->Ystrides.assign(Y_reverse_strides.rbegin() + Zrank_remain,
                       Y_reverse_strides.rend());
  aux->Zstrides.assign(Z_reverse_strides.rbegin() + Zrank_remain,
                       Z_reverse_strides.rend());
  aux->Z_total_dim = Zstride;
  aux->XY_same_shape = X == Y;
  aux->vectorization = (X_reverse_strides[0] == 1) &&
                       (Y_reverse_strides[0] == 1) &&
                       (Z_reverse_strides[0] == 1);
  return true;
}

bool BroadcastToInferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
  BroadcastAux aux;
  if (!BroadcastToPrepare(X, Y, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
struct BroadcastToMeta {
  static void Forward(const T* X, T* Z) noexcept { *Z = *X; }

  static void Forward(int n, const T* X, T* Z) noexcept {
    LLMath<T>::copy(n, X, Z);
  }

  static void Backward(const T* gZ, T* gX) noexcept { *gX += *gZ; }

  static void Backward(int n, const T* gZ, T* gX) noexcept {
    LLMath<T>::add(n, gX, gZ, gX);
  }
};

template <typename T>
void _BroadcastToRank1(const T* X, T* Z, const BroadcastAux& aux) noexcept {
  if (aux.vectorization) {
    BroadcastToMeta<T>::Forward(aux.Z[0], X, Z);
  } else {
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      BroadcastToMeta<T>::Forward(X, Z);
      X += aux.Xstrides[0];
      Z += aux.Zstrides[0];
    }
  }
}

template <typename T>
void _BroadcastToRank2(const T* X, T* Z, const BroadcastAux& aux) noexcept {
  const T* _X;
  T* _Z;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    if (aux.vectorization) {
      BroadcastToMeta<T>::Forward(aux.Z[1], X, Z);
    } else {
      _X = X;
      _Z = Z;
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        BroadcastToMeta<T>::Forward(_X, _Z);
        _X += aux.Xstrides[1];
        _Z += aux.Zstrides[1];
      }
    }
    X += aux.Xstrides[0];
    Z += aux.Zstrides[0];
  }
}

template <typename T>
void _BroadcastToRank3(const T* X, T* Z, const BroadcastAux& aux) noexcept {
  const T* _X;
  T* _Z;
  const T* __X;
  T* __Z;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    _X = X;
    _Z = Z;
    for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
      if (aux.vectorization) {
        BroadcastToMeta<T>::Forward(aux.Z[2], _X, _Z);
      } else {
        __X = _X;
        __Z = _Z;
        for (int i2 = 0; i2 < aux.Z[2]; ++i2) {
          BroadcastToMeta<T>::Forward(__X, __Z);
          __X += aux.Xstrides[2];
          __Z += aux.Zstrides[2];
        }
      }
      _X += aux.Xstrides[1];
      _Z += aux.Zstrides[1];
    }
    X += aux.Xstrides[0];
    Z += aux.Zstrides[0];
  }
}

template <typename T>
void _BroadcastToFallback(int level, const T* X, T* Z,
                          const BroadcastAux& aux) noexcept {
  if (level < aux.Z.rank() - 1) {
    for (int i = 0; i < aux.Z[level]; ++i) {
      _BroadcastToFallback(level + 1, X, Z, aux);
      X += aux.Xstrides[level];
      Z += aux.Zstrides[level];
    }
  } else {
    if (aux.vectorization) {
      BroadcastToMeta<T>::Forward(aux.Z[level], X, Z);
    } else {
      for (int i = 0; i < aux.Z[level]; ++i) {
        BroadcastToMeta<T>::Forward(X, Z);
        X += aux.Xstrides[level];
        Z += aux.Zstrides[level];
      }
    }
  }
}

template <typename T>
void _BroadcastTo(const T* X, T* Z, const BroadcastAux& aux) noexcept {
  if (aux.XY_same_shape) {
    BroadcastToMeta<T>::Forward(aux.Z_total_dim, X, Z);
  } else if (aux.Z.is_rank(1)) {
    _BroadcastToRank1(X, Z, aux);
  } else if (aux.Z.is_rank(2)) {
    _BroadcastToRank2(X, Z, aux);
  } else if (aux.Z.is_rank(3)) {
    _BroadcastToRank3(X, Z, aux);
  } else {
    _BroadcastToFallback(0, X, Z, aux);
  }
}

template <typename T>
void _BroadcastToBackwardRank1(const T* gZ, T* gX,
                               const BroadcastAux& aux) noexcept {
  if (aux.vectorization) {
    BroadcastToMeta<T>::Backward(aux.Z[0], gZ, gX);
  } else {
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      BroadcastToMeta<T>::Backward(gZ, gX);
      gZ += aux.Zstrides[0];
      gX += aux.Xstrides[0];
    }
  }
}

template <typename T>
void _BroadcastToBackwardRank2(const T* gZ, T* gX,
                               const BroadcastAux& aux) noexcept {
  const T* _gZ;
  T* _gX;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    if (aux.vectorization) {
      BroadcastToMeta<T>::Backward(aux.Z[1], gZ, gX);
    } else {
      _gZ = gZ;
      _gX = gX;
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        BroadcastToMeta<T>::Backward(_gZ, _gX);
        _gZ += aux.Zstrides[1];
        _gX += aux.Xstrides[1];
      }
    }
    gZ += aux.Zstrides[0];
    gX += aux.Xstrides[0];
  }
}

template <typename T>
void _BroadcastToBackwardRank3(const T* gZ, T* gX,
                               const BroadcastAux& aux) noexcept {
  const T* _gZ;
  T* _gX;
  const T* __gZ;
  T* __gX;
  for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
    _gZ = gZ;
    _gX = gX;
    for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
      if (aux.vectorization) {
        BroadcastToMeta<T>::Backward(aux.Z[2], _gZ, _gX);
      } else {
        __gZ = _gZ;
        __gX = _gX;
        for (int i2 = 0; i2 < aux.Z[2]; ++i2) {
          BroadcastToMeta<T>::Backward(__gZ, __gX);
          __gZ += aux.Zstrides[2];
          __gX += aux.Xstrides[2];
        }
      }
      _gZ += aux.Zstrides[1];
      _gX += aux.Xstrides[1];
    }
    gZ += aux.Zstrides[0];
    gX += aux.Xstrides[0];
  }
}

template <typename T>
void _BroadcastToBackwardFallback(int level, const T* gZ, T* gX,
                                  const BroadcastAux& aux) noexcept {
  if (level < aux.Z.rank() - 1) {
    for (int i = 0; i < aux.Z[level]; ++i) {
      _BroadcastToBackwardFallback(level + 1, gZ, gX, aux);
      gZ += aux.Zstrides[level];
      gX += aux.Xstrides[level];
    }
  } else {
    if (aux.vectorization) {
      BroadcastToMeta<T>::Backward(aux.Z[level], gZ, gX);
    } else {
      for (int i = 0; i < aux.Z[level]; ++i) {
        BroadcastToMeta<T>::Backward(gZ, gX);
        gZ += aux.Zstrides[level];
        gX += aux.Xstrides[level];
      }
    }
  }
}

template <typename T>
void _BroadcastToBackward(const T* gZ, T* gX,
                          const BroadcastAux& aux) noexcept {
  if (aux.XY_same_shape) {
    BroadcastToMeta<T>::Backward(aux.Z_total_dim, gZ, gX);
  } else if (aux.Z.is_rank(1)) {
    _BroadcastToBackwardRank1(gZ, gX, aux);
  } else if (aux.Z.is_rank(2)) {
    _BroadcastToBackwardRank2(gZ, gX, aux);
  } else if (aux.Z.is_rank(3)) {
    _BroadcastToBackwardRank3(gZ, gX, aux);
  } else {
    _BroadcastToBackwardFallback(0, gZ, gX, aux);
  }
}

template <typename T>
void BroadcastTo(const Tensor<T>& X, Tensor<T>* Z,
                 const BroadcastAux& aux) noexcept {
  _BroadcastTo(X.data(), Z->data(), aux);
}

template <typename T>
void BroadcastToBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                         const Tensor<T>& gZ, Tensor<T>* gX,
                         const BroadcastAux& aux) noexcept {
  _BroadcastToBackward(gZ.data(), gX->data(), aux);
}

}  // namespace

BroadcastToNode::BroadcastToNode(std::string name, GraphNode* X,
                                 const Shape& shape)
    : GraphNodeUnaryBase(std::move(name), X), new_shape_(shape) {
  // 'shape' must be valid.
  DXCHECK_THROW(shape.rank() > 0);
  DXCHECK_THROW(shape.total_dim() > 0);
  if (!X->shape().empty()) {
    (void)BroadcastToInferShape(X->shape(), new_shape_, &shape_);
  }
}

class BroadcastToOp : public OpUnaryBase {
 private:
  BroadcastAux aux_;

 public:
  DEFINE_OP_LIKE(BroadcastToOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(BroadcastToPrepare(
        X_->shape(), ((const BroadcastToNode*)node_)->new_shape(), &aux_));
    return aux_.Z;
  }

  void Forward() override { BroadcastTo(*X_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      BroadcastToBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BroadcastTo);

/************************************************************************/
/* BroadcastToLike */
/************************************************************************/
BroadcastToLikeNode::BroadcastToLikeNode(std::string name, GraphNode* X,
                                         GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BroadcastToInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BroadcastToLikeOp : public OpBinaryBase {
 private:
  BroadcastAux aux_;

 public:
  DEFINE_OP_LIKE(BroadcastToLikeOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(BroadcastToPrepare(X_->shape(), Y_->shape(), &aux_));
    return aux_.Z;
  }

  void Forward() override { BroadcastTo(*X_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      BroadcastToBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BroadcastToLike);

}  // namespace deepx_core
