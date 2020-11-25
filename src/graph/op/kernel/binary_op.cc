// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#include "binary_meta.h"

namespace deepx_core {
namespace {

#define DEFINE_BINNARY_ELEMENT_WISE_OP(name)                                 \
  template <typename T>                                                      \
  void name(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z) noexcept { \
    DXASSERT_SAME_SHAPE(X, Y, *Z);                                           \
    Binary##name##Meta<T>::Forward(X.total_dim(), X.data(), Y.data(),        \
                                   Z->data());                               \
  }

#define DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(name)                         \
  template <typename T>                                                       \
  void name##Backward(const Tensor<T>& X, const Tensor<T>& Y,                 \
                      const Tensor<T>& Z, const Tensor<T>& gZ, Tensor<T>* gX, \
                      Tensor<T>* gY) noexcept {                               \
    T* _gX = nullptr;                                                         \
    T* _gY = nullptr;                                                         \
    DXASSERT_SAME_SHAPE(X, Y, Z, gZ);                                         \
    if (gX) {                                                                 \
      DXASSERT_SAME_SHAPE(X, *gX);                                            \
      _gX = gX->data();                                                       \
    }                                                                         \
    if (gY) {                                                                 \
      DXASSERT_SAME_SHAPE(X, *gY);                                            \
      _gY = gY->data();                                                       \
    }                                                                         \
    Binary##name##Meta<T>::Backward(X.total_dim(), X.data(), Y.data(),        \
                                    Z.data(), gZ.data(), _gX, _gY);           \
  }

DEFINE_BINNARY_ELEMENT_WISE_OP(Add)
DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(Add)
DEFINE_BINNARY_ELEMENT_WISE_OP(Sub)
DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(Sub)
DEFINE_BINNARY_ELEMENT_WISE_OP(Mul)
DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(Mul)
DEFINE_BINNARY_ELEMENT_WISE_OP(Div)
DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(Div)
DEFINE_BINNARY_ELEMENT_WISE_OP(Pow)
DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(Pow)
DEFINE_BINNARY_ELEMENT_WISE_OP(Max)
DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(Max)
DEFINE_BINNARY_ELEMENT_WISE_OP(Min)
DEFINE_BINNARY_ELEMENT_WISE_OP_BACKWARD(Min)
DEFINE_BINNARY_ELEMENT_WISE_OP(Equal)
DEFINE_BINNARY_ELEMENT_WISE_OP(Greater)
DEFINE_BINNARY_ELEMENT_WISE_OP(GreaterEqual)
DEFINE_BINNARY_ELEMENT_WISE_OP(Less)
DEFINE_BINNARY_ELEMENT_WISE_OP(LessEqual)

}  // namespace

#define DEFINE_BINNARY_ELEMENT_WISE_OP1(name)                          \
  name##Node::name##Node(std::string name, GraphNode* X, GraphNode* Y) \
      : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}       \
                                                                       \
  class name##Op : public OpBinaryElementWiseBase {                    \
   public:                                                             \
    DEFINE_OP_LIKE(name##Op);                                          \
    void Forward() override { name(*X_, *Y_, Z_); }                    \
    void Backward() override {                                         \
      if (gZ_) {                                                       \
        name##Backward(*X_, *Y_, *Z_, *gZ_, gX_, gY_);                 \
      }                                                                \
    }                                                                  \
  };                                                                   \
                                                                       \
  GRAPH_NODE_OP_REGISTER(name)

#define DEFINE_BINNARY_ELEMENT_WISE_OP2(name)                          \
  name##Node::name##Node(std::string name, GraphNode* X, GraphNode* Y) \
      : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}       \
                                                                       \
  class name##Op : public OpBinaryElementWiseBase {                    \
   public:                                                             \
    DEFINE_OP_LIKE(name##Op);                                          \
    void Forward() override { name(*X_, *Y_, Z_); }                    \
  };                                                                   \
                                                                       \
  GRAPH_NODE_OP_REGISTER(name)

DEFINE_BINNARY_ELEMENT_WISE_OP1(Add);
DEFINE_BINNARY_ELEMENT_WISE_OP1(Sub);
DEFINE_BINNARY_ELEMENT_WISE_OP1(Mul);
DEFINE_BINNARY_ELEMENT_WISE_OP1(Div);
DEFINE_BINNARY_ELEMENT_WISE_OP1(Pow);
DEFINE_BINNARY_ELEMENT_WISE_OP1(Max);
DEFINE_BINNARY_ELEMENT_WISE_OP1(Min);
DEFINE_BINNARY_ELEMENT_WISE_OP2(Equal);
DEFINE_BINNARY_ELEMENT_WISE_OP2(Greater);
DEFINE_BINNARY_ELEMENT_WISE_OP2(GreaterEqual);
DEFINE_BINNARY_ELEMENT_WISE_OP2(Less);
DEFINE_BINNARY_ELEMENT_WISE_OP2(LessEqual);

}  // namespace deepx_core
