// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#if HAVE_SAGE2 == 1
#include <sage2/vmf.h>
#endif

namespace deepx_core {
namespace {

template <typename T>
void Sigmoid(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  LLTensor<T>::sigmoid(X, Z);
}

template <typename T>
void SigmoidBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                     const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    _gX[i] += _Z[i] * (1 - _Z[i]) * _gZ[i];
  }
}

template <typename T>
void Tanh(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  LLTensor<T>::tanh(X, Z);
}

template <typename T>
void TanhBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                  const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    _gX[i] += (1 - _Z[i] * _Z[i]) * _gZ[i];
  }
}

template <typename T>
void Relu(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_X[i] > 0) {
      _Z[i] = _X[i];
    } else {
      _Z[i] = 0;
    }
  }
}

template <typename T>
void ReluBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                  const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    if (_Z[i] > 0) {
      _gX[i] += _gZ[i];
    }
  }
}

#if HAVE_SAGE2 == 1
template <>
void Relu(const Tensor<float>& X, Tensor<float>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const float* _X = X.data();
  float* _Z = Z->data();
  sage2_relu_ps(X.total_dim(), _X, _Z);
}
#endif

template <typename T>
void SoftPlus(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _Z[i] = std::log(1 + std::exp(_X[i]));
  }
}

template <typename T>
void SoftPlusBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                      const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    _gX[i] += (1 - 1 / std::exp(_Z[i])) * _gZ[i];
  }
}

template <typename T>
void Swish(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _Z[i] = _X[i] * LLMath<T>::sigmoid(_X[i]);
  }
}

template <typename T>
void SwishBackward(const Tensor<T>& X, const Tensor<T>& Z, const Tensor<T>& gZ,
                   Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(X, Z, gZ, *gX);
  const T* _X = X.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_X[i] != 0) {
      _gX[i] += _Z[i] * (1 + (1 - _Z[i]) / _X[i]) * _gZ[i];
    } else {
      _gX[i] += (T)0.5 * _gZ[i];
    }
  }
}

template <typename T>
void Exp(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  LLTensor<T>::exp(X, Z);
}

template <typename T>
void ExpBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                 const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  LLTensor<T>::xypz(Z, gZ, gX);
}

template <typename T>
void Log(const Tensor<T>& X, Tensor<T>* Z) noexcept {
#if !defined NDEBUG
  const T* _X = X.data();
  for (int i = 0; i < X.total_dim(); ++i) {
    DXASSERT(_X[i] > 0);
  }
#endif
  LLTensor<T>::log(X, Z);
}

template <typename T>
void LogBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                 const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  LLTensor<T>::xdypz(gZ, X, gX);
}

template <typename T>
void Negate(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _Z[i] = -_X[i];
  }
}

template <typename T>
void NegateBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                    const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(gZ, *gX);
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < gZ.total_dim(); ++i) {
    _gX[i] -= _gZ[i];
  }
}

template <typename T>
void Inv(const Tensor<T>& X, Tensor<T>* Z) noexcept {
#if !defined NDEBUG
  const T* _X = X.data();
  for (int i = 0; i < X.total_dim(); ++i) {
    DXASSERT(_X[i] != 0);
  }
#endif
  LLTensor<T>::inv(X, Z);
}

template <typename T>
void InvBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                 const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    _gX[i] -= _Z[i] * _Z[i] * _gZ[i];
  }
}

template <typename T>
void Sqrt(const Tensor<T>& X, Tensor<T>* Z) noexcept {
#if !defined NDEBUG
  const T* _X = X.data();
  for (int i = 0; i < X.total_dim(); ++i) {
    DXASSERT(_X[i] >= 0);
  }
#endif
  LLTensor<T>::sqrt(X, Z);
}

template <typename T>
void SqrtBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                  const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    if (_Z[i] > (T)1e-3) {
      _gX[i] += (T)0.5 / _Z[i] * _gZ[i];
    } else {
      _gX[i] += (T)0.5 / (T)1e-3 * _gZ[i];
    }
  }
}

template <typename T>
void Cbrt(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  LLTensor<T>::cbrt(X, Z);
}

template <typename T>
void CbrtBackward(const Tensor<T>& /*X*/, const Tensor<T>& Z,
                  const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    if (_Z[i] < (T)-1e-2 || _Z[i] > (T)1e-2) {
      _gX[i] += 1 / (3 * _Z[i] * _Z[i]) * _gZ[i];
    } else {
      _gX[i] += 1 / (3 * (T)1e-2 * (T)1e-2) * _gZ[i];
    }
  }
}

template <typename T>
void Square(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  LLTensor<T>::square(X, Z);
}

template <typename T>
void SquareBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                    const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(X, gZ, *gX);
  const T* _X = X.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _gX[i] += 2 * _X[i] * _gZ[i];
  }
}

template <typename T>
void Cubic(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  LLTensor<T>::cubic(X, Z);
}

template <typename T>
void CubicBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                   const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(X, gZ, *gX);
  const T* _X = X.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _gX[i] += 3 * _X[i] * _X[i] * _gZ[i];
  }
}

template <typename T>
void Abs(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  LLTensor<T>::abs(X, Z);
}

template <typename T>
void AbsBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                 const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(X, gZ, *gX);
  const T* _X = X.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_X[i] > 0) {
      _gX[i] += _gZ[i];
    } else {
      _gX[i] -= _gZ[i];
    }
  }
}

}  // namespace

#define DEFINE_UNARY_ELEMENT_WISE_OP(name)                   \
  name##Node::name##Node(std::string name, GraphNode* X)     \
      : GraphNodeUnaryElementWiseBase(std::move(name), X) {} \
                                                             \
  class name##Op : public OpUnaryElementWiseBase {           \
   public:                                                   \
    DEFINE_OP_LIKE(name##Op);                                \
    void Forward() override { name(*X_, Z_); }               \
    void Backward() override {                               \
      if (gX_) {                                             \
        name##Backward(*X_, *Z_, *gZ_, gX_);                 \
      }                                                      \
    }                                                        \
  };                                                         \
                                                             \
  GRAPH_NODE_OP_REGISTER(name)

DEFINE_UNARY_ELEMENT_WISE_OP(Sigmoid);
DEFINE_UNARY_ELEMENT_WISE_OP(Tanh);
DEFINE_UNARY_ELEMENT_WISE_OP(Relu);
DEFINE_UNARY_ELEMENT_WISE_OP(SoftPlus);
DEFINE_UNARY_ELEMENT_WISE_OP(Swish);
DEFINE_UNARY_ELEMENT_WISE_OP(Exp);
DEFINE_UNARY_ELEMENT_WISE_OP(Log);
DEFINE_UNARY_ELEMENT_WISE_OP(Negate);
DEFINE_UNARY_ELEMENT_WISE_OP(Inv);
DEFINE_UNARY_ELEMENT_WISE_OP(Sqrt);
DEFINE_UNARY_ELEMENT_WISE_OP(Cbrt);
DEFINE_UNARY_ELEMENT_WISE_OP(Square);
DEFINE_UNARY_ELEMENT_WISE_OP(Cubic);
DEFINE_UNARY_ELEMENT_WISE_OP(Abs);

/************************************************************************/
/* LeakyRelu */
/************************************************************************/
namespace {

template <typename T>
void LeakyRelu(T alpha, const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_X[i] > 0) {
      _Z[i] = _X[i];
    } else {
      _Z[i] = alpha * _X[i];
    }
  }
}

template <typename T>
void LeakyReluBackward(T alpha, const Tensor<T>& /*X*/, const Tensor<T>& Z,
                       const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    if (_Z[i] > 0) {
      _gX[i] += _gZ[i];
    } else {
      _gX[i] += alpha * _gZ[i];
    }
  }
}

}  // namespace

LeakyReluNode::LeakyReluNode(std::string name, GraphNode* X, double alpha)
    : GraphNodeUnaryElementWiseBase(std::move(name), X), alpha_(alpha) {
  DXCHECK_THROW(alpha_ >= 0);
}

class LeakyReluOp : public OpUnaryElementWiseBase {
 private:
  float_t alpha_ = 0;

 public:
  DEFINE_OP_LIKE(LeakyReluOp);

  void InitForward() override {
    OpUnaryElementWiseBase::InitForward();
    alpha_ = (float_t)((const LeakyReluNode*)node_)->alpha();
  }

 public:
  void Forward() override { LeakyRelu(alpha_, *X_, Z_); }

  void Backward() override {
    if (gX_) {
      LeakyReluBackward(alpha_, *X_, *Z_, *gZ_, gX_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(LeakyRelu);

/************************************************************************/
/* Elu */
/************************************************************************/
namespace {

template <typename T>
void Elu(T alpha, const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_X[i] > 0) {
      _Z[i] = _X[i];
    } else {
      _Z[i] = alpha * (std::exp(_X[i]) - 1);
    }
  }
}

template <typename T>
void EluBackward(T alpha, const Tensor<T>& /*X*/, const Tensor<T>& Z,
                 const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    if (_Z[i] > 0) {
      _gX[i] += _gZ[i];
    } else {
      _gX[i] += (_Z[i] + alpha) * _gZ[i];
    }
  }
}

}  // namespace

EluNode::EluNode(std::string name, GraphNode* X, double alpha)
    : GraphNodeUnaryElementWiseBase(std::move(name), X), alpha_(alpha) {
  DXCHECK_THROW(alpha_ >= 0);
}

class EluOp : public OpUnaryElementWiseBase {
 private:
  float_t alpha_ = 0;

 public:
  DEFINE_OP_LIKE(EluOp);

  void InitForward() override {
    OpUnaryElementWiseBase::InitForward();
    alpha_ = (float_t)((const EluNode*)node_)->alpha();
  }

 public:
  void Forward() override { Elu(alpha_, *X_, Z_); }

  void Backward() override {
    if (gX_) {
      EluBackward(alpha_, *X_, *Z_, *gZ_, gX_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Elu);

/************************************************************************/
/* Selu */
/************************************************************************/
namespace {

template <typename T>
void Selu(T lambda, T lambda_alpha, const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_X[i] > 0) {
      _Z[i] = lambda * _X[i];
    } else {
      _Z[i] = lambda_alpha * (std::exp(_X[i]) - 1);
    }
  }
}

template <typename T>
void SeluBackward(T lambda, T lambda_alpha, const Tensor<T>& /*X*/,
                  const Tensor<T>& Z, const Tensor<T>& gZ,
                  Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(Z, gZ, *gX);
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < Z.total_dim(); ++i) {
    if (_Z[i] > 0) {
      _gX[i] += lambda * _gZ[i];
    } else {
      _gX[i] += (_Z[i] + lambda_alpha) * _gZ[i];
    }
  }
}

}  // namespace

SeluNode::SeluNode(std::string name, GraphNode* X, double lambda, double alpha)
    : GraphNodeUnaryElementWiseBase(std::move(name), X),
      lambda_(lambda),
      alpha_(alpha) {
  DXCHECK_THROW(lambda_ > 0);
  DXCHECK_THROW(alpha_ >= 0);
}

class SeluOp : public OpUnaryElementWiseBase {
 private:
  float_t lambda_ = 0;
  float_t alpha_ = 0;
  float_t lambda_alpha_ = 0;

 public:
  DEFINE_OP_LIKE(SeluOp);

  void InitForward() override {
    OpUnaryElementWiseBase::InitForward();
    lambda_ = (float_t)((const SeluNode*)node_)->lambda();
    alpha_ = (float_t)((const SeluNode*)node_)->alpha();
    lambda_alpha_ = lambda_ * alpha_;
  }

 public:
  void Forward() override { Selu(lambda_, lambda_alpha_, *X_, Z_); }

  void Backward() override {
    if (gX_) {
      SeluBackward(lambda_, lambda_alpha_, *X_, *Z_, *gZ_, gX_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Selu);

/************************************************************************/
/* Gelu */
/************************************************************************/
namespace {

template <typename T>
struct GeluMutableAux {
  Tensor<T> buf1;  // X * X
  Tensor<T> buf2;  // tanh(...)
};

template <typename T>
void GeluPrepare(const Shape& X, GeluMutableAux<T>* maux) {
  maux->buf1.resize(X);
  maux->buf2.resize(X);
}

template <typename T>
void Gelu(const Tensor<T>& X, Tensor<T>* Z, GeluMutableAux<T>* maux) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  static constexpr T A = (T)0.7978845608028654;  // sqrt(2/PI)
  static constexpr T B = (T)0.044715 * A;
  const T* _X = X.data();
  T* _Z = Z->data();
  T* _buf1 = maux->buf1.data();
  T* _buf2 = maux->buf2.data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _buf1[i] = _X[i] * _X[i];
    _buf2[i] = std::tanh(_X[i] * (A + B * _buf1[i]));
    _Z[i] = (T)0.5 * _X[i] * (1 + _buf2[i]);
  }
}

template <typename T>
void GeluBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                  const Tensor<T>& gZ, Tensor<T>* gX,
                  const GeluMutableAux<T>& maux) noexcept {
  DXASSERT_SAME_SHAPE(X, gZ, *gX);
  static constexpr T A = (T)0.7978845608028654;  // sqrt(2/PI)
  static constexpr T B = (T)0.044715 * A;
  static constexpr T C = (T)3 * B;
  const T* _X = X.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _buf1 = maux.buf1.data();
  const T* _buf2 = maux.buf2.data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _gX[i] += (T)0.5 *
              ((1 + _buf2[i]) +
               _X[i] * (1 - _buf2[i] * _buf2[i]) * (A + C * _buf1[i])) *
              _gZ[i];
  }
}

}  // namespace

GeluNode::GeluNode(std::string name, GraphNode* X)
    : GraphNodeUnaryElementWiseBase(std::move(name), X) {}

class GeluOp : public OpUnaryElementWiseBase {
 private:
  GeluMutableAux<float_t> maux_;

 public:
  DEFINE_OP_LIKE(GeluOp);

  void InitForward() override {
    OpUnaryElementWiseBase::InitForward();
    GeluPrepare(X_->shape(), &maux_);
  }

  void Forward() override { Gelu(*X_, Z_, &maux_); }

  void Backward() override {
    if (gX_) {
      GeluBackward(*X_, *Z_, *gZ_, gX_, maux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Gelu);

/************************************************************************/
/* Dropout */
/************************************************************************/
namespace {

template <class RandomEngine, typename T>
void DropoutMask(RandomEngine&& engine, T keep_prob, Tensor<T>* mask) noexcept {
  DXASSERT(0 < keep_prob && keep_prob <= 1);
  T inv = 1 / keep_prob;
  T* _mask = mask->data();
  mask->rand(engine);
  for (int i = 0; i < mask->total_dim(); ++i) {
    _mask[i] = (_mask[i] <= keep_prob) ? inv : 0;
  }
}

template <typename T>
void Dropout(const Tensor<T>& mask, const Tensor<T>& X, Tensor<T>* Z) {
  LLTensor<T>::mul(mask, X, Z);
}

template <typename T>
void DropoutBackward(const Tensor<T>& mask, const Tensor<T>& /*X*/,
                     const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                     Tensor<T>* gX) {
  LLTensor<T>::xypz(mask, gZ, gX);
}

}  // namespace

DropoutNode::DropoutNode(std::string name, GraphNode* X, double keep_prob)
    : GraphNodeUnaryElementWiseBase(std::move(name), X), keep_prob_(keep_prob) {
  DXCHECK_THROW(0 < keep_prob_ && keep_prob_ <= 1);
}

class DropoutOp : public OpUnaryElementWiseBase {
 private:
  float_t keep_prob_ = 0;
  tsr_t mask_;

 public:
  DEFINE_OP_LIKE(DropoutOp);

  void InitForward() override {
    OpUnaryElementWiseBase::InitForward();
    keep_prob_ = (float_t)((const DropoutNode*)node_)->keep_prob();
    mask_.resize(X_->shape());
  }

  void InitPredict() override { OpUnaryElementWiseBase::InitForward(); }

  void Forward() override {
    DropoutMask(hidden_->engine(), keep_prob_, &mask_);
    Dropout(mask_, *X_, Z_);
  }

  void Predict() override { Z_->set_data(*X_); }

  void Backward() override {
    if (gX_) {
      DropoutBackward(mask_, *X_, *Z_, *gZ_, gX_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Dropout);

/************************************************************************/
/* Sign */
/************************************************************************/
namespace {

template <typename T>
void Sign(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_X[i] > 0) {
      _Z[i] = 1;
    } else if (_X[i] == 0) {
      _Z[i] = 0;
    } else {
      _Z[i] = -1;
    }
  }
}

}  // namespace

SignNode::SignNode(std::string name, GraphNode* X)
    : GraphNodeUnaryElementWiseBase(std::move(name), X) {}

class SignOp : public OpUnaryElementWiseBase {
 public:
  DEFINE_OP_LIKE(SignOp);

  void Forward() override { Sign(*X_, Z_); }
};

GRAPH_NODE_OP_REGISTER(Sign);

/************************************************************************/
/* ClipByValue */
/************************************************************************/
namespace {

template <typename T>
void ClipByValue(const Tensor<T>& X, Tensor<T>* Z, T clip_value_min,
                 T clip_value_max) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    T value = _X[i];
    if (value < clip_value_min) {
      _Z[i] = clip_value_min;
    } else if (value > clip_value_max) {
      _Z[i] = clip_value_max;
    } else {
      _Z[i] = value;
    }
  }
}

template <typename T>
void ClipByValue(const Tensor<T>& X, Tensor<T>* Z, T clip_value_min,
                 T clip_value_max, Tensor<T>* aux) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z, *aux);
  const T* _X = X.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    T value = _X[i];
    _aux[i] = 0;
    if (value < clip_value_min) {
      _Z[i] = clip_value_min;
    } else if (value > clip_value_max) {
      _Z[i] = clip_value_max;
    } else {
      _Z[i] = value;
      _aux[i] = 1;
    }
  }
}

template <typename T>
void ClipByValueBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                         const Tensor<T>& gZ, Tensor<T>* gX,
                         const Tensor<T>& aux) noexcept {
  LLTensor<T>::xypz(aux, gZ, gX);
}

}  // namespace

ClipByValueNode::ClipByValueNode(std::string name, GraphNode* X,
                                 double clip_value_min, double clip_value_max)
    : GraphNodeUnaryElementWiseBase(std::move(name), X),
      clip_value_min_(clip_value_min),
      clip_value_max_(clip_value_max) {
  DXCHECK_THROW(clip_value_min <= clip_value_max);
}

class ClipByValueOp : public OpUnaryElementWiseBase {
 private:
  float_t clip_value_min_ = 0;
  float_t clip_value_max_ = 0;
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(ClipByValueOp);

  void InitForward() override {
    InitPredict();
    aux_.resize(X_->shape());
  }

  void InitPredict() override {
    OpUnaryElementWiseBase::InitForward();
    clip_value_min_ =
        (float_t)((const ClipByValueNode*)node_)->clip_value_min();
    clip_value_max_ =
        (float_t)((const ClipByValueNode*)node_)->clip_value_max();
  }

  void Forward() override {
    ClipByValue(*X_, Z_, clip_value_min_, clip_value_max_, &aux_);
  }

  void Predict() override {
    ClipByValue(*X_, Z_, clip_value_min_, clip_value_max_);
  }

  void Backward() override {
    if (gX_) {
      ClipByValueBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(ClipByValue);

/************************************************************************/
/* MatrixBandPartNode */
/************************************************************************/
namespace {

template <typename T>
struct MatrixBandPartAux {
  int pre_dim = 0;   // total dim of X without the last two axis
  int post_dim = 0;  // the last two axis dim
  Tensor<T> in_band;
};

template <typename T>
bool MatrixBandPartPrepare(const Shape& Xshape, int num_lower, int num_upper,
                           MatrixBandPartAux<T>* aux) noexcept {
  int rank = Xshape.rank();
  if (rank < 2) {
    DXERROR("Invalid X: rank of X %d must be greater than or equal to 2.",
            rank);
    return false;
  }

  int m = Xshape.dim(rank - 2);
  if (num_lower >= m) {
    DXERROR(
        "Invalid num_lower: num_lower %d must be less than the number of rows "
        "of the inner matrix %d.",
        num_lower, m);
    return false;
  }

  int n = Xshape.dim(rank - 1);
  if (num_upper >= n) {
    DXERROR(
        "Invalid num_upper: num_upper %d must be less than the number of cols "
        "of the inner matrix %d.",
        num_upper, n);
    return false;
  }

  aux->pre_dim = 1;
  for (int i = 0; i < rank - 2; ++i) {
    aux->pre_dim *= Xshape.dim(i);
  }

  aux->post_dim = m * n;

  aux->in_band.resize(m, n);
  T* _in_band = aux->in_band.data();
  int k = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      _in_band[k] = 0;
      if ((num_lower < 0 || (i - j) <= num_lower) &&
          (num_upper < 0 || (j - i) <= num_upper)) {
        _in_band[k] = 1;
      }
      ++k;
    }
  }
  return true;
}

template <typename T>
void MatrixBandPart(const Tensor<T>& X, Tensor<T>* Z,
                    const MatrixBandPartAux<T>& aux) noexcept {
  DXASSERT_SAME_SHAPE(X, *Z);
  const T* _X = X.data();
  T* _Z = Z->data();
  const T* _in_band = aux.in_band.data();
  for (int i = 0; i < aux.pre_dim; ++i) {
    LLMath<T>::mul(aux.post_dim, _X, _in_band, _Z);
    _X += aux.post_dim;
    _Z += aux.post_dim;
  }
}

template <typename T>
void MatrixBandPartBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                            const Tensor<T>& gZ, Tensor<T>* gX,
                            const MatrixBandPartAux<T>& aux) noexcept {
  DXASSERT_SAME_SHAPE(gZ, *gX);
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _in_band = aux.in_band.data();
  for (int i = 0; i < aux.pre_dim; ++i) {
    LLMath<T>::xypz(aux.post_dim, _gZ, _in_band, _gX);
    _gZ += aux.post_dim;
    _gX += aux.post_dim;
  }
}

}  // namespace

MatrixBandPartNode::MatrixBandPartNode(std::string name, GraphNode* X,
                                       int num_lower, int num_upper)
    : GraphNodeUnaryElementWiseBase(std::move(name), X),
      num_lower_(num_lower),
      num_upper_(num_upper) {}

class MatrixBandPartOp : public OpUnaryElementWiseBase {
 private:
  MatrixBandPartAux<float_t> aux_;

 public:
  DEFINE_OP_LIKE(MatrixBandPartOp);

  void InitForward() override {
    OpUnaryElementWiseBase::InitForward();
    int num_lower = ((const MatrixBandPartNode*)node_)->num_lower();
    int num_upper = ((const MatrixBandPartNode*)node_)->num_upper();
    DXCHECK_THROW(
        MatrixBandPartPrepare(X_->shape(), num_lower, num_upper, &aux_));
  }

  void Forward() override { MatrixBandPart(*X_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      MatrixBandPartBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(MatrixBandPart);

/************************************************************************/
/* IdentityNode */
/************************************************************************/
IdentityNode::IdentityNode(std::string name, GraphNode* X)
    : GraphNodeUnaryElementWiseBase(std::move(name), X) {}

class IdentityOp : public OpUnaryElementWiseBase {
 public:
  DEFINE_OP_LIKE(IdentityOp);

  void Forward() override { Z_->set_data(*X_); }

  void Backward() override {
    if (gZ_) {
      ll_tensor_t::add(*gZ_, *gX_, gX_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Identity);

}  // namespace deepx_core
