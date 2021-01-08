// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/any.h>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace deepx_core {

class AnyTest : public testing::Test {
 protected:
  using vi_t = std::vector<int>;
  static const vi_t VI234;
};

const AnyTest::vi_t AnyTest::VI234{2, 3, 4};

TEST_F(AnyTest, Copy) {
  Any a1(1);
  EXPECT_TRUE(a1.is<int>());
  EXPECT_EQ(a1.unsafe_to_ref<int>(), 1);

  Any a2(a1);
  EXPECT_TRUE(a2.is<int>());
  EXPECT_EQ(a2.unsafe_to_ref<int>(), 1);

  Any a3;
  EXPECT_TRUE(a3.is<void>());
  a3 = a2;
  EXPECT_TRUE(a3.is<int>());
  EXPECT_EQ(a3.unsafe_to_ref<int>(), 1);

  Any a4;
  EXPECT_TRUE(a4.is<void>());
  a1 = a4;
  EXPECT_TRUE(a1.is<void>());
}

TEST_F(AnyTest, Move) {
  Any a1(1);

  Any a2(std::move(a1));
  EXPECT_EQ(a2.to_ref<int>(), 1);

  Any a3;
  a3 = std::move(a2);
  EXPECT_EQ(a3.to_ref<int>(), 1);
}

TEST_F(AnyTest, Construct_any) {
  Any a1(1);
  EXPECT_EQ(a1.to_ref<int>(), 1);
  a1 = VI234;
  EXPECT_EQ(a1.to_ref<vi_t>(), VI234);

  Any a2(VI234);
  EXPECT_EQ(a2.to_ref<vi_t>(), VI234);
  a2 = 1;
  EXPECT_EQ(a2.to_ref<int>(), 1);
}

TEST_F(AnyTest, emplace_1) {
  Any a;
  vi_t vi(VI234);
  a.emplace(1);
  EXPECT_EQ(a.to_ref<int>(), 1);
  a.emplace(VI234);
  EXPECT_EQ(a.to_ref<vi_t>(), VI234);
  a.emplace(std::move(vi));
  EXPECT_EQ(a.to_ref<vi_t>(), VI234);
}

TEST_F(AnyTest, emplace_2) {
  Any a;
  a.emplace<vi_t>({2, 3, 4});
  EXPECT_EQ(a.to_ref<vi_t>(), VI234);
  a.emplace<vi_t>(3, 6);
  EXPECT_EQ(a.to_ref<vi_t>(), vi_t({6, 6, 6}));
}

TEST_F(AnyTest, reset) {
  Any a = 1;
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(a);
  EXPECT_TRUE(a.is<int>());
  a.reset();
  EXPECT_TRUE(a.empty());
  EXPECT_FALSE(a);
  EXPECT_TRUE(a.is<void>());
}

TEST_F(AnyTest, swap) {
  Any a1 = 1, a2 = 2;
  a1.swap(a2);
  EXPECT_EQ(a1.to_ref<int>(), 2);
  EXPECT_EQ(a2.to_ref<int>(), 1);
}

TEST_F(AnyTest, type_index) {
  Any a;
  EXPECT_EQ(a.type_index(), typeid(void));
  a = 1;
  EXPECT_EQ(a.type_index(), typeid(int));
  a = VI234;
  EXPECT_EQ(a.type_index(), typeid(vi_t));
}

TEST_F(AnyTest, is) {
  Any a1 = 1;
  Any a2 = VI234;
  EXPECT_TRUE(a1.is<int>());
  EXPECT_FALSE(a1.is<vi_t>());
  EXPECT_FALSE(a2.is<int>());
  EXPECT_TRUE(a2.is<vi_t>());
}

TEST_F(AnyTest, to_ref) {
  Any a1 = 1, a2;
  EXPECT_EQ(a1.to_ref<int>(), 1);
  EXPECT_ANY_THROW(a1.to_ref<int*>());
  auto& p1 = a1.to_ref<int>();
  a2 = &p1;
  auto* p2 = a2.to_ref<int*>();
  *p2 = 2;
  EXPECT_EQ(a1.to_ref<int>(), 2);
}

TEST_F(AnyTest, unsafe_to_ref) {
  Any a1 = 1, a2;
  EXPECT_EQ(a1.unsafe_to_ref<int>(), 1);
  auto& p1 = a1.unsafe_to_ref<int>();
  a2 = &p1;
  auto* p2 = a2.unsafe_to_ref<int*>();
  *p2 = 2;
  EXPECT_EQ(a1.unsafe_to_ref<int>(), 2);
}

}  // namespace deepx_core
