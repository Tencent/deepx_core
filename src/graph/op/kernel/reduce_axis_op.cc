// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct ReduceAxisAux {
  Shape Z;
  int m = 0;  // pre axis dim
  int n = 0;  // post axis dim
  int k = 0;  // axis dim
};

bool ReduceAxisPrepare(const Shape& X, int axis, int keep_dim,
                       ReduceAxisAux* aux) noexcept {
  int rank = X.rank();
  if (rank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  if (!X.real_axis(&axis)) {
    DXERROR("Invalid axis: %d.", axis);
    return false;
  }

  int Zrank;
  int Zdims[SHAPE_MAX_RANK];
  int m = 1, n = 1, k;
  for (int j = 0; j < axis; ++j) {
    Zdims[j] = X[j];
    m *= X[j];
  }
  k = X[axis];
  if (rank == 1 || keep_dim) {
    Zrank = rank;
    Zdims[axis] = 1;
    for (int j = axis + 1; j < rank; ++j) {
      Zdims[j] = X[j];
      n *= X[j];
    }
  } else {
    Zrank = rank - 1;
    for (int j = axis + 1; j < rank; ++j) {
      Zdims[j - 1] = X[j];
      n *= X[j];
    }
  }

  aux->Z.assign(&Zdims[0], &Zdims[Zrank]);
  aux->m = m;
  aux->n = n;
  aux->k = k;
  return true;
}

bool ReduceAxisInferShape(const Shape& X, int axis, int keep_dim,
                          Shape* Z) noexcept {
  ReduceAxisAux aux;
  if (!ReduceAxisPrepare(X, axis, keep_dim, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

bool ReduceAxisPrepare(const Shape& X, ReduceAxisAux* aux) noexcept {
  int rank = X.rank();
  if (rank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  aux->Z.resize(1);
  aux->m = 1;
  aux->n = 1;
  aux->k = X.total_dim();
  return true;
}

bool ReduceAxisInferShape(const Shape& X, Shape* Z) noexcept {
  ReduceAxisAux aux;
  if (!ReduceAxisPrepare(X, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
struct ReduceAxisMutableAux {
  Tensor<T> buf1;
  Tensor<T> buf2;
};

template <typename T>
void ReduceAxisPrepare(const ReduceAxisAux& aux,
                       ReduceAxisMutableAux<T>* maux) {
  if (aux.n != 1) {
    maux->buf1.resize(aux.k);
    maux->buf2.resize(aux.k);
  }
}

template <typename T, class Meta>
void ReduceAxis(const Tensor<T>& X, Tensor<T>* Z, const ReduceAxisAux& aux,
                ReduceAxisMutableAux<T>* maux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  T* _Z = Z->data();
  if (n == 1) {
    for (int i = 0; i < m; ++i) {
      Meta::Forward(k, _X, _Z);
      _X += k;
      _Z += 1;
    }
  } else {
    const T* __X;
    const T* ___X;
    T* __Z;
    T* buf1 = maux->buf1.data();
    for (int i = 0; i < m; ++i) {
      __X = _X;
      __Z = _Z;
      for (int jj = 0; jj < n; ++jj) {
        ___X = __X;
        for (int j = 0; j < k; ++j) {
          buf1[j] = *___X;
          ___X += n;
        }
        Meta::Forward(k, buf1, __Z);
        __X += 1;
        __Z += 1;
      }
      _X += k * n;
      _Z += n;
    }
  }
}

template <typename T, class Meta>
void ReduceAxisBackward(const Tensor<T>& X, const Tensor<T>& Z,
                        const Tensor<T>& gZ, Tensor<T>* gX,
                        const ReduceAxisAux& aux,
                        ReduceAxisMutableAux<T>* maux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  if (n == 1) {
    for (int i = 0; i < m; ++i) {
      Meta::Backward(k, _X, _Z, _gZ, _gX);
      _X += k;
      _Z += 1;
      _gZ += 1;
      _gX += k;
    }
  } else {
    const T* __X;
    const T* ___X;
    const T* __Z;
    const T* __gZ;
    T* __gX;
    T* ___gX;
    T* buf1 = maux->buf1.data();
    T* buf2 = maux->buf2.data();
    for (int i = 0; i < m; ++i) {
      __X = _X;
      __Z = _Z;
      __gZ = _gZ;
      __gX = _gX;
      for (int jj = 0; jj < n; ++jj) {
        ___X = __X;
        ___gX = __gX;
        for (int j = 0; j < k; ++j) {
          buf1[j] = *___X;
          buf2[j] = *___gX;
          ___X += n;
          ___gX += n;
        }
        Meta::Backward(k, buf1, __Z, __gZ, buf2);
        ___X = __X;
        ___gX = __gX;
        for (int j = 0; j < k; ++j) {
          *___gX = buf2[j];
          ___gX += n;
        }
        __X += 1;
        __Z += 1;
        __gZ += 1;
        __gX += 1;
      }
      _X += k * n;
      _Z += n;
      _gZ += n;
      _gX += k * n;
    }
  }
}

template <typename T>
struct ReduceAxisReduceSumMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    *Z = LLMath<T>::sum(n, X);
  }

  static void Backward(int n, const T* /*X*/, const T* /*Z*/, const T* gZ,
                       T* gX) noexcept {
    LLMath<T>::add_scalar(n, gX, *gZ, gX);
  }
};

template <typename T>
struct ReduceAxisReduceMeanMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    *Z = LLMath<T>::sum(n, X) / n;
  }

  static void Backward(int n, const T* /*X*/, const T* /*Z*/, const T* gZ,
                       T* gX) noexcept {
    LLMath<T>::add_scalar(n, gX, *gZ / n, gX);
  }
};

template <typename T>
struct ReduceAxisReduceMaxMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    *Z = LLMath<T>::max(n, X);
  }

  static void Backward(int n, const T* X, const T* Z, const T* gZ,
                       T* gX) noexcept {
    for (int i = 0; i < n; ++i) {
      if (X[i] == *Z) {
        gX[i] += *gZ;
      }
    }
  }
};

template <typename T>
struct ReduceAxisReduceMinMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    *Z = LLMath<T>::min(n, X);
  }

  static void Backward(int n, const T* X, const T* Z, const T* gZ,
                       T* gX) noexcept {
    for (int i = 0; i < n; ++i) {
      if (X[i] == *Z) {
        gX[i] += *gZ;
      }
    }
  }
};

template <typename T>
struct ReduceAxisReduceL1Meta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    *Z = LLMath<T>::norm1(n, X);
  }

  static void Backward(int n, const T* X, const T* /*Z*/, const T* gZ,
                       T* gX) noexcept {
    for (int i = 0; i < n; ++i) {
      if (X[i] > 0) {
        gX[i] += *gZ;
      } else {
        gX[i] -= *gZ;
      }
    }
  }
};

template <typename T>
struct ReduceAxisReduceL2Meta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    *Z = LLMath<T>::norm2(n, X);
  }

  static void Backward(int n, const T* X, const T* Z, const T* gZ,
                       T* gX) noexcept {
    for (int i = 0; i < n; ++i) {
      if (*Z > (T)1e-6) {
        gX[i] += X[i] * (*gZ / *Z);
      } else {
        gX[i] += X[i] * (*gZ / (T)1e-6);
      }
    }
  }
};

template <typename T>
struct ReduceAxisArgMaxMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    int j = 0;
    T m = X[0];
    for (int i = 1; i < n; ++i) {
      if (m < X[i]) {
        m = X[i];
        j = i;
      }
    }
    *Z = (T)j;
  }
};

template <typename T>
struct ReduceAxisArgMinMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    int j = 0;
    T m = X[0];
    for (int i = 1; i < n; ++i) {
      if (m > X[i]) {
        m = X[i];
        j = i;
      }
    }
    *Z = (T)j;
  }
};

#define DEFINE_REDUCE_AXIS_OP(name)                                     \
  template <typename T>                                                 \
  void name(const Tensor<T>& X, Tensor<T>* Z, const ReduceAxisAux& aux, \
            ReduceAxisMutableAux<T>* maux) noexcept {                   \
    ReduceAxis<T, ReduceAxis##name##Meta<T>>(X, Z, aux, maux);          \
  }

#define DEFINE_REDUCE_AXIS_OP_BACKWARD(name)                                   \
  template <typename T>                                                        \
  void name##Backward(const Tensor<T>& X, const Tensor<T>& Z,                  \
                      const Tensor<T>& gZ, Tensor<T>* gX,                      \
                      const ReduceAxisAux& aux,                                \
                      ReduceAxisMutableAux<T>* maux) noexcept {                \
    ReduceAxisBackward<T, ReduceAxis##name##Meta<T>>(X, Z, gZ, gX, aux, maux); \
  }

DEFINE_REDUCE_AXIS_OP(ReduceSum)
DEFINE_REDUCE_AXIS_OP_BACKWARD(ReduceSum)
DEFINE_REDUCE_AXIS_OP(ReduceMean)
DEFINE_REDUCE_AXIS_OP_BACKWARD(ReduceMean)
DEFINE_REDUCE_AXIS_OP(ReduceMax)
DEFINE_REDUCE_AXIS_OP_BACKWARD(ReduceMax)
DEFINE_REDUCE_AXIS_OP(ReduceMin)
DEFINE_REDUCE_AXIS_OP_BACKWARD(ReduceMin)
DEFINE_REDUCE_AXIS_OP(ReduceL1)
DEFINE_REDUCE_AXIS_OP_BACKWARD(ReduceL1)
DEFINE_REDUCE_AXIS_OP(ReduceL2)
DEFINE_REDUCE_AXIS_OP_BACKWARD(ReduceL2)
DEFINE_REDUCE_AXIS_OP(ArgMax)
DEFINE_REDUCE_AXIS_OP(ArgMin)

}  // namespace

GraphNodeReduceAxisBase::GraphNodeReduceAxisBase(std::string name, GraphNode* X,
                                                 int axis, int keep_dim)
    : GraphNodeUnaryBase(std::move(name), X), axis_(axis), keep_dim_(keep_dim) {
  if (!X->shape().empty()) {
    (void)ReduceAxisInferShape(X->shape(), axis_, keep_dim_, &shape_);
  }
}

GraphNodeReduceAxisBase::GraphNodeReduceAxisBase(std::string name, GraphNode* X)
    : GraphNodeUnaryBase(std::move(name), X), reduce_all_(1) {
  if (!X->shape().empty()) {
    (void)ReduceAxisInferShape(X->shape(), &shape_);
  }
}

class OpReduceAxisBase : public OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const tsr_t* X_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  ReduceAxisAux aux_;
  ReduceAxisMutableAux<float_t> maux_;

 public:
  void InitForward() override {
    const GraphNodeReduceAxisBase* node =  // NOLINT
        (const GraphNodeReduceAxisBase*)node_;
    int reduce_all = node->reduce_all();
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    if (reduce_all == 0) {
      int axis = node->axis();
      int keep_dim = node->keep_dim();
      DXCHECK_THROW(ReduceAxisPrepare(X_->shape(), axis, keep_dim, &aux_));
    } else {
      DXCHECK_THROW(ReduceAxisPrepare(X_->shape(), &aux_));
    }
    Z_ = InitHiddenTSR(node_, aux_.Z);
    ReduceAxisPrepare(aux_, &maux_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
  }
};

#define DEFINE_REDUCE_AXIS_OP1(name)                                   \
  name##Node::name##Node(std::string name, GraphNode* X, int axis,     \
                         int keep_dim)                                 \
      : GraphNodeReduceAxisBase(std::move(name), X, axis, keep_dim) {} \
                                                                       \
  name##Node::name##Node(std::string name, GraphNode* X)               \
      : GraphNodeReduceAxisBase(std::move(name), X) {}                 \
                                                                       \
  class name##Op : public OpReduceAxisBase {                           \
   public:                                                             \
    DEFINE_OP_LIKE(name##Op);                                          \
    void Forward() override { name(*X_, Z_, aux_, &maux_); }           \
    void Backward() override {                                         \
      if (gX_) {                                                       \
        name##Backward(*X_, *Z_, *gZ_, gX_, aux_, &maux_);             \
      }                                                                \
    }                                                                  \
  };                                                                   \
                                                                       \
  GRAPH_NODE_OP_REGISTER(name)

#define DEFINE_REDUCE_AXIS_OP2(name)                               \
  name##Node::name##Node(std::string name, GraphNode* X, int axis) \
      : GraphNodeReduceAxisBase(std::move(name), X, axis, 0) {}    \
                                                                   \
  class name##Op : public OpReduceAxisBase {                       \
   public:                                                         \
    DEFINE_OP_LIKE(name##Op);                                      \
    void Forward() override { name(*X_, Z_, aux_, &maux_); }       \
  };                                                               \
                                                                   \
  GRAPH_NODE_OP_REGISTER(name)

DEFINE_REDUCE_AXIS_OP1(ReduceMean);
DEFINE_REDUCE_AXIS_OP1(ReduceSum);
DEFINE_REDUCE_AXIS_OP1(ReduceMax);
DEFINE_REDUCE_AXIS_OP1(ReduceMin);
DEFINE_REDUCE_AXIS_OP1(ReduceL1);
DEFINE_REDUCE_AXIS_OP1(ReduceL2);
DEFINE_REDUCE_AXIS_OP2(ArgMax);
DEFINE_REDUCE_AXIS_OP2(ArgMin);

}  // namespace deepx_core
