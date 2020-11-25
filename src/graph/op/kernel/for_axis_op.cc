// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct ForAxisAux {
  Shape Z;
  int m = 0;  // pre axis dim
  int n = 0;  // post axis dim
  int k = 0;  // axis dim
};

bool ForAxisPrepare(const Shape& X, int axis, ForAxisAux* aux) noexcept {
  int rank = X.rank();
  if (rank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  if (!X.real_axis(&axis)) {
    DXERROR("Invalid axis: %d.", axis);
    return false;
  }

  int m = 1, n = 1, k;
  for (int j = 0; j < axis; ++j) {
    m *= X[j];
  }
  k = X[axis];
  for (int j = axis + 1; j < rank; ++j) {
    n *= X[j];
  }

  aux->Z = X;
  aux->m = m;
  aux->n = n;
  aux->k = k;
  return true;
}

bool ForAxisInferShape(const Shape& X, int axis, Shape* Z) noexcept {
  ForAxisAux aux;
  if (!ForAxisPrepare(X, axis, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
struct ForAxisMutableAux {
  Tensor<T> buf1;
  Tensor<T> buf2;
  Tensor<T> buf3;
  Tensor<T> buf4;
};

template <typename T>
void ForAxisPrepare(const ForAxisAux& aux, ForAxisMutableAux<T>* maux) {
  if (aux.n != 1) {
    maux->buf1.resize(aux.k);
    maux->buf2.resize(aux.k);
    maux->buf3.resize(aux.k);
    maux->buf4.resize(aux.k);
  }
}

template <typename T, class Meta>
void ForAxis(const Tensor<T>& X, Tensor<T>* Z, const ForAxisAux& aux,
             ForAxisMutableAux<T>* maux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  T* _Z = Z->data();
  if (n == 1) {
    for (int i = 0; i < m; ++i) {
      Meta::Forward(k, _X, _Z);
      _X += k;
      _Z += k;
    }
  } else {
    const T* __X;
    const T* ___X;
    T* __Z;
    T* ___Z;
    T* buf1 = maux->buf1.data();
    T* buf2 = maux->buf2.data();
    for (int i = 0; i < m; ++i) {
      __X = _X;
      __Z = _Z;
      for (int jj = 0; jj < n; ++jj) {
        ___X = __X;
        ___Z = __Z;
        for (int j = 0; j < k; ++j) {
          buf1[j] = *___X;
          ___X += n;
        }
        Meta::Forward(k, buf1, buf2);
        for (int j = 0; j < k; ++j) {
          *___Z = buf2[j];
          ___Z += n;
        }
        __X += 1;
        __Z += 1;
      }
      _X += k * n;
      _Z += k * n;
    }
  }
}

template <typename T, class Meta>
void ForAxisBackward(const Tensor<T>& X, const Tensor<T>& Z,
                     const Tensor<T>& gZ, Tensor<T>* gX, const ForAxisAux& aux,
                     ForAxisMutableAux<T>* maux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  if (n == 1) {
    for (int i = 0; i < m; ++i) {
      Meta::Backward(k, _X, _Z, _gZ, _gX);
      _X += k;
      _Z += k;
      _gZ += k;
      _gX += k;
    }
  } else {
    const T* __X;
    const T* ___X;
    const T* __Z;
    const T* ___Z;
    const T* __gZ;
    const T* ___gZ;
    T* __gX;
    T* ___gX;
    T* buf1 = maux->buf1.data();
    T* buf2 = maux->buf2.data();
    T* buf3 = maux->buf3.data();
    T* buf4 = maux->buf4.data();
    for (int i = 0; i < m; ++i) {
      __X = _X;
      __Z = _Z;
      __gZ = _gZ;
      __gX = _gX;
      for (int jj = 0; jj < n; ++jj) {
        ___X = __X;
        ___Z = __Z;
        ___gZ = __gZ;
        ___gX = __gX;
        for (int j = 0; j < k; ++j) {
          buf1[j] = *___X;
          buf2[j] = *___Z;
          buf3[j] = *___gZ;
          buf4[j] = *___gX;
          ___X += n;
          ___Z += n;
          ___gZ += n;
          ___gX += n;
        }
        Meta::Backward(k, buf1, buf2, buf3, buf4);
        ___gX = __gX;
        for (int j = 0; j < k; ++j) {
          *___gX = buf4[j];
          ___gX += n;
        }
        __X += 1;
        __Z += 1;
        __gZ += 1;
        __gX += 1;
      }
      _X += k * n;
      _Z += k * n;
      _gZ += k * n;
      _gX += k * n;
    }
  }
}

template <typename T>
struct ForAxisSoftmaxMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    LLMath<T>::softmax(n, X, Z);
  }

  static void Backward(int n, const T* /*X*/, const T* Z, const T* gZ,
                       T* gX) noexcept {
    LLMath<T>::xypz(n, gZ, Z, gX);
    LLMath<T>::axpy(n, -LLMath<T>::dot(n, gZ, Z), Z, gX);
  }
};

template <typename T>
struct ForAxisSoftmax2Meta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    LLMath<T>::softmax2(n, X, Z);
  }

  static void Backward(int n, const T* /*X*/, const T* Z, const T* gZ,
                       T* gX) noexcept {
    LLMath<T>::xypz(n, gZ, Z, gX);
    LLMath<T>::axpy(n, -LLMath<T>::dot(n, gZ, Z) / n, Z, gX);
  }
};

template <typename T>
struct ForAxisLogSoftmaxMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    LLMath<T>::softmax(n, X, Z);
    LLMath<T>::log(n, Z, Z);
  }

  static void Backward(int n, const T* /*X*/, const T* Z, const T* gZ,
                       T* gX) noexcept {
    T s = LLMath<T>::sum(n, gZ);
    for (int i = 0; i < n; ++i) {
      gX[i] += gZ[i] - std::exp(Z[i]) * s;
    }
  }
};

template <typename T>
struct ForAxisHardmaxMeta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    int j = 0;
    T m = X[0];
    Z[0] = 0;
    for (int i = 1; i < n; ++i) {
      if (m < X[i]) {
        m = X[i];
        j = i;
      }
      Z[i] = 0;
    }
    Z[j] = 1;
  }
};

template <typename T>
struct ForAxisNormalize2Meta {
  static void Forward(int n, const T* X, T* Z) noexcept {
    T norm2_x = LLMath<T>::norm2(n, X);
    norm2_x = norm2_x < (T)1e-6 ? (T)1e-6 : norm2_x;
    T a = 1 / norm2_x;
    LLMath<T>::mul_scalar(n, X, a, Z);
  }

  static void Backward(int n, const T* X, const T* Z, const T* gZ,
                       T* gX) noexcept {
    T norm2_x = LLMath<T>::norm2(n, X);
    norm2_x = norm2_x < (T)1e-6 ? (T)1e-6 : norm2_x;
    T a = 1 / norm2_x;
    LLMath<T>::axpy(n, a, gZ, gX);
    LLMath<T>::axpy(n, -LLMath<T>::dot(n, gZ, Z) * a, Z, gX);
  }
};

#define DEFINE_FOR_AXIS_OP(name)                                     \
  template <typename T>                                              \
  void name(const Tensor<T>& X, Tensor<T>* Z, const ForAxisAux& aux, \
            ForAxisMutableAux<T>* maux) noexcept {                   \
    ForAxis<T, ForAxis##name##Meta<T>>(X, Z, aux, maux);             \
  }

#define DEFINE_FOR_AXIS_OP_BACKWARD(name)                                \
  template <typename T>                                                  \
  void name##Backward(const Tensor<T>& X, const Tensor<T>& Z,            \
                      const Tensor<T>& gZ, Tensor<T>* gX,                \
                      const ForAxisAux& aux,                             \
                      ForAxisMutableAux<T>* maux) noexcept {             \
    ForAxisBackward<T, ForAxis##name##Meta<T>>(X, Z, gZ, gX, aux, maux); \
  }

DEFINE_FOR_AXIS_OP(Softmax)
DEFINE_FOR_AXIS_OP_BACKWARD(Softmax)
DEFINE_FOR_AXIS_OP(Softmax2)
DEFINE_FOR_AXIS_OP_BACKWARD(Softmax2)
DEFINE_FOR_AXIS_OP(LogSoftmax)
DEFINE_FOR_AXIS_OP_BACKWARD(LogSoftmax)
DEFINE_FOR_AXIS_OP(Hardmax)
DEFINE_FOR_AXIS_OP(Normalize2)
DEFINE_FOR_AXIS_OP_BACKWARD(Normalize2)

}  // namespace

GraphNodeForAxisBase::GraphNodeForAxisBase(std::string name, GraphNode* X,
                                           int axis)
    : GraphNodeUnaryBase(std::move(name), X), axis_(axis) {
  if (!X->shape().empty()) {
    (void)ForAxisInferShape(X->shape(), axis_, &shape_);
  }
}

class OpForAxisBase : public OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const tsr_t* X_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  ForAxisAux aux_;
  ForAxisMutableAux<float_t> maux_;

 public:
  void InitForward() override {
    int axis = ((const GraphNodeForAxisBase*)node_)->axis();
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    DXCHECK_THROW(ForAxisPrepare(X_->shape(), axis, &aux_));
    Z_ = InitHiddenTSR(node_, aux_.Z);
    ForAxisPrepare(aux_, &maux_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
  }
};

#define DEFINE_FOR_AXIS_OP1(name)                                  \
  name##Node::name##Node(std::string name, GraphNode* X, int axis) \
      : GraphNodeForAxisBase(std::move(name), X, axis) {}          \
                                                                   \
  class name##Op : public OpForAxisBase {                          \
   public:                                                         \
    DEFINE_OP_LIKE(name##Op);                                      \
    void Forward() override { name(*X_, Z_, aux_, &maux_); }       \
    void Backward() override {                                     \
      if (gX_) {                                                   \
        name##Backward(*X_, *Z_, *gZ_, gX_, aux_, &maux_);         \
      }                                                            \
    }                                                              \
  };                                                               \
                                                                   \
  GRAPH_NODE_OP_REGISTER(name)

DEFINE_FOR_AXIS_OP1(Softmax);
DEFINE_FOR_AXIS_OP1(Softmax2);
DEFINE_FOR_AXIS_OP1(LogSoftmax);
DEFINE_FOR_AXIS_OP1(Normalize2);

}  // namespace deepx_core
