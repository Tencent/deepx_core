// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/class_factory.h>
#include <gtest/gtest.h>
#include <memory>
#include <typeindex>

namespace deepx_core {

#define A_REGISTER(class_name, name) CLASS_FACTORY_REGISTER(A, class_name, name)
#define A_NEW(name) CLASS_FACTORY_NEW(A, name)

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

A_REGISTER(A, "A");
A_REGISTER(B, "B");
A_REGISTER(C, "C");
A_REGISTER(D, "D");

class ClassFactoryTest : public testing::Test {};

TEST_F(ClassFactoryTest, ClassFactory) {
  std::unique_ptr<A> a(A_NEW("A"));
  ASSERT_TRUE(a);
  EXPECT_EQ(a->type_index(), typeid(A));

  std::unique_ptr<A> b(A_NEW("B"));
  ASSERT_TRUE(b);
  EXPECT_EQ(b->type_index(), typeid(B));

  std::unique_ptr<A> c(A_NEW("C"));
  ASSERT_TRUE(c);
  EXPECT_EQ(c->type_index(), typeid(C));

  std::unique_ptr<A> d(A_NEW("D"));
  ASSERT_TRUE(d);
  EXPECT_EQ(d->type_index(), typeid(D));

  std::unique_ptr<A> e(A_NEW("E"));
  ASSERT_FALSE(e);
}

}  // namespace deepx_core
