// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/variable_scope.h>
#include <gtest/gtest.h>
#include <memory>

namespace deepx_core {

class VariableScopeTest : public testing::Test {};

TEST_F(VariableScopeTest, ClearVariable) {
  (void)GetVariable("X", Shape(1));
  ClearVariable();

  (void)GetVariable("X", Shape(2));
  ClearVariable();
}

TEST_F(VariableScopeTest, ReleaseVariable) {
  std::unique_ptr<VariableNode> X1(GetVariable("X", Shape(1)));
  ReleaseVariable();

  std::unique_ptr<VariableNode> X2(GetVariable("X", Shape(2)));
  ReleaseVariable();
}

TEST_F(VariableScopeTest, GetVariable) {
  // global scope
  auto* X1 = GetVariable("X", Shape(1));
  auto* X2 = GetVariable("X", Shape(1));
  EXPECT_EQ(X1, X2);  // X1, X2 share the same VariableNode
  EXPECT_EQ(X1->name(), "X");

  // couldn't change X's shape
  EXPECT_ANY_THROW(GetVariable("X", Shape(2)));

  {
    // scope a
    VariableScopeEnterer a("a");
    auto* a_X1 = GetVariable("X", Shape(2));
    auto* a_X2 = GetVariable("X", Shape(2));
    EXPECT_EQ(a_X1, a_X2);
    EXPECT_NE(a_X1, X1);
    EXPECT_EQ(a_X1->name(), "a/X");
    {
      // scope a/b
      VariableScopeEnterer b("b");
      auto* a_b_X1 = GetVariable("X", Shape(3));
      auto* a_b_X2 = GetVariable("X", Shape(3));
      EXPECT_EQ(a_b_X1, a_b_X2);
      EXPECT_NE(a_b_X1, a_X1);
      EXPECT_NE(a_b_X1, X1);
      EXPECT_EQ(a_b_X1->name(), "a/b/X");

      auto* a_b_Y = GetVariable("Y", Shape(3));
      EXPECT_EQ(a_b_Y->name(), "a/b/Y");
    }

    // scope a
    auto* a_Y = GetVariable("Y", Shape(2));
    EXPECT_EQ(a_Y->name(), "a/Y");
  }

  // global scope
  auto* Y = GetVariable("Y", Shape(1));
  EXPECT_EQ(Y->name(), "Y");

  ClearVariable();
}

}  // namespace deepx_core
