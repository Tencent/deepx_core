// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "class_factory_test.h"
#include <deepx_core/common/class_factory.h>
#include <gtest/gtest.h>

namespace deepx_core {

class ClassFactoryTest : public testing::Test {};

TEST_F(ClassFactoryTest, ClassFactory) {
  auto a = class_factory_make_unique<A>("A");
  ASSERT_TRUE(a);
  EXPECT_EQ(a->type_index(), typeid(A));

  auto b = class_factory_make_unique<A>("B");
  ASSERT_TRUE(b);
  EXPECT_EQ(b->type_index(), typeid(B));

  auto c = class_factory_make_shared<A>("C");
  ASSERT_TRUE(c);
  EXPECT_EQ(c->type_index(), typeid(C));

  auto d = class_factory_make_shared<A>("D");
  ASSERT_TRUE(d);
  EXPECT_EQ(d->type_index(), typeid(D));

  auto e = class_factory_make_shared<A>("E");
  ASSERT_FALSE(e);
}

}  // namespace deepx_core
