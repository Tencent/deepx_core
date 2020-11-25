// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <typeindex>

namespace deepx_core {

class A {
 public:
  virtual ~A() = default;
  virtual std::type_index type_index() const noexcept { return typeid(A); }
};

class B : public A {
 public:
  std::type_index type_index() const noexcept override { return typeid(B); }
};

class C : public A {
 public:
  std::type_index type_index() const noexcept override { return typeid(C); }
};

class D : public B {
 public:
  std::type_index type_index() const noexcept override { return typeid(D); }
};

}  // namespace deepx_core
