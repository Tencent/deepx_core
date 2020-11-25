// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/array_view.h>
#include <deepx_core/common/array_view_io.h>
#include <deepx_core/common/stream.h>
#include <gtest/gtest.h>
#include <array>
#include <string>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* ArrayView */
/************************************************************************/
class ArrayViewTest : public testing::Test {
 protected:
  using avi_t = ArrayView<int>;
  static int CAI234[3];
  static std::array<int, 3> AI234;
  static std::vector<int> VI234;
  static std::vector<int> VI567;
  static std::string S234;
};

int ArrayViewTest::CAI234[3] = {2, 3, 4};
std::array<int, 3> ArrayViewTest::AI234{2, 3, 4};
std::vector<int> ArrayViewTest::VI234{2, 3, 4};
std::vector<int> ArrayViewTest::VI567{5, 6, 7};
std::string ArrayViewTest::S234{"234"};

TEST_F(ArrayViewTest, Construct_c_array) {
  avi_t a(CAI234);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(a);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
}

TEST_F(ArrayViewTest, Construct_std_array) {
  avi_t a(AI234);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
}

TEST_F(ArrayViewTest, Construct_std_vector) {
  avi_t a(VI234);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
}

TEST_F(ArrayViewTest, Construct_std_string) {
  std::string s("WTF");
  ArrayView<char> a(s);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[0], 'W');
  EXPECT_EQ(a[1], 'T');
  EXPECT_EQ(a[2], 'F');
}

TEST_F(ArrayViewTest, iterator) {
  avi_t a(VI234);
  int sum = 0;
  for (int i : a) {
    sum += i;
  }
  EXPECT_EQ(sum, 9);

  const avi_t& ca = a;
  sum = 0;
  for (int i : ca) {
    sum += i;
  }
  EXPECT_EQ(sum, 9);
}

TEST_F(ArrayViewTest, ElementAccess) {
  avi_t a(VI234);
  EXPECT_EQ(a.at(0), 2);
  EXPECT_EQ(a.at(1), 3);
  EXPECT_EQ(a.at(2), 4);
  EXPECT_ANY_THROW(a.at(3));
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
  EXPECT_EQ(a.front(), 2);
  EXPECT_EQ(a.back(), 4);

  const avi_t& ca = a;
  EXPECT_EQ(ca.at(0), 2);
  EXPECT_EQ(ca.at(1), 3);
  EXPECT_EQ(ca.at(2), 4);
  EXPECT_ANY_THROW(ca.at(3));
  EXPECT_EQ(ca[0], 2);
  EXPECT_EQ(ca[1], 3);
  EXPECT_EQ(ca[2], 4);
  EXPECT_EQ(ca.front(), 2);
  EXPECT_EQ(ca.back(), 4);
}

TEST_F(ArrayViewTest, clear) {
  avi_t a(VI234);
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(a);
  a.clear();
  EXPECT_TRUE(a.empty());
  EXPECT_FALSE(a);
}

TEST_F(ArrayViewTest, remove_prefix) {
  avi_t a(VI234);
  a.remove_prefix(1);
  EXPECT_EQ(a.size(), 2u);
  EXPECT_EQ(a.front(), 3);
  a.remove_prefix(1);
  EXPECT_EQ(a.size(), 1u);
  EXPECT_EQ(a.front(), 4);
  a.remove_prefix(1);
  EXPECT_EQ(a.size(), 0u);
}

TEST_F(ArrayViewTest, remove_suffix) {
  avi_t a(VI234);
  a.remove_suffix(1);
  EXPECT_EQ(a.size(), 2u);
  EXPECT_EQ(a.back(), 3);
  a.remove_suffix(1);
  EXPECT_EQ(a.size(), 1u);
  EXPECT_EQ(a.back(), 2);
  a.remove_suffix(1);
  EXPECT_EQ(a.size(), 0u);
}

TEST_F(ArrayViewTest, swap) {
  avi_t a1(VI234);
  avi_t a2(VI567);
  a1.swap(a2);
  EXPECT_EQ(a1[0], 5);
  EXPECT_EQ(a1[1], 6);
  EXPECT_EQ(a1[2], 7);
  EXPECT_EQ(a2[0], 2);
  EXPECT_EQ(a2[1], 3);
  EXPECT_EQ(a2[2], 4);
}

TEST_F(ArrayViewTest, Compare) {
  avi_t a1(VI234);
  avi_t a2(VI567);
  EXPECT_TRUE(a1 != a2);
  EXPECT_FALSE(a1 == a2);
}

TEST_F(ArrayViewTest, Compare_std_nullptr_t) {
  avi_t a1(VI234);
  avi_t a2;
  EXPECT_TRUE(a1 != nullptr);
  EXPECT_FALSE(a1 == nullptr);
  EXPECT_FALSE(nullptr != a2);
  EXPECT_TRUE(nullptr == a2);
}

TEST_F(ArrayViewTest, Compare_std_string) {
  string_view sv(S234);
  EXPECT_TRUE(sv == S234);
  EXPECT_FALSE(sv != S234);
  EXPECT_TRUE(S234 == sv);
  EXPECT_FALSE(S234 != sv);
}

TEST_F(ArrayViewTest, WriteReadView) {
  avi_t a(VI234), read_a;

  OutputStringStream os;
  InputStringStream is;

  os << a;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_a);
  ASSERT_TRUE(is);

  EXPECT_EQ(a, read_a);
}

/************************************************************************/
/* ConstArrayView */
/************************************************************************/
class ConstArrayViewTest : public ArrayViewTest {
 protected:
  using cavi_t = ConstArrayView<int>;
};

TEST_F(ConstArrayViewTest, Construct_c_array) {
  cavi_t a(CAI234);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(a);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
}

TEST_F(ConstArrayViewTest, Construct_std_array) {
  cavi_t a(AI234);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
}

TEST_F(ConstArrayViewTest, Construct_std_vector) {
  cavi_t a(VI234);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
}

TEST_F(ConstArrayViewTest, Construct_std_string) {
  std::string s("WTF");
  ConstArrayView<char> a(s);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(a[0], 'W');
  EXPECT_EQ(a[1], 'T');
  EXPECT_EQ(a[2], 'F');
}

TEST_F(ConstArrayViewTest, iterator) {
  cavi_t a(VI234);
  int sum = 0;
  for (int i : a) {
    sum += i;
  }
  EXPECT_EQ(sum, 9);
}

TEST_F(ConstArrayViewTest, ElementAccess) {
  cavi_t a(VI234);
  EXPECT_EQ(a.at(0), 2);
  EXPECT_EQ(a.at(1), 3);
  EXPECT_EQ(a.at(2), 4);
  EXPECT_ANY_THROW(a.at(3));
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 3);
  EXPECT_EQ(a[2], 4);
  EXPECT_EQ(a.front(), 2);
  EXPECT_EQ(a.back(), 4);
}

TEST_F(ConstArrayViewTest, clear) {
  cavi_t a(VI234);
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(a);
  a.clear();
  EXPECT_TRUE(a.empty());
  EXPECT_FALSE(a);
}

TEST_F(ConstArrayViewTest, remove_prefix) {
  cavi_t a(VI234);
  a.remove_prefix(1);
  EXPECT_EQ(a.size(), 2u);
  EXPECT_EQ(a.front(), 3);
  a.remove_prefix(1);
  EXPECT_EQ(a.size(), 1u);
  EXPECT_EQ(a.front(), 4);
  a.remove_prefix(1);
  EXPECT_EQ(a.size(), 0u);
}

TEST_F(ConstArrayViewTest, remove_suffix) {
  cavi_t a(VI234);
  a.remove_suffix(1);
  EXPECT_EQ(a.size(), 2u);
  EXPECT_EQ(a.back(), 3);
  a.remove_suffix(1);
  EXPECT_EQ(a.size(), 1u);
  EXPECT_EQ(a.back(), 2);
  a.remove_suffix(1);
  EXPECT_EQ(a.size(), 0u);
}

TEST_F(ConstArrayViewTest, swap) {
  cavi_t a1(VI234);
  cavi_t a2(VI567);
  a1.swap(a2);
  EXPECT_EQ(a1[0], 5);
  EXPECT_EQ(a1[1], 6);
  EXPECT_EQ(a1[2], 7);
  EXPECT_EQ(a2[0], 2);
  EXPECT_EQ(a2[1], 3);
  EXPECT_EQ(a2[2], 4);
}

TEST_F(ConstArrayViewTest, Compare) {
  cavi_t a1(VI234);
  cavi_t a2(VI567);
  EXPECT_TRUE(a1 != a2);
  EXPECT_FALSE(a1 == a2);
}

TEST_F(ConstArrayViewTest, Compare_std_nullptr_t) {
  cavi_t a1(VI234);
  cavi_t a2;
  EXPECT_TRUE(a1 != nullptr);
  EXPECT_FALSE(a1 == nullptr);
  EXPECT_FALSE(nullptr != a2);
  EXPECT_TRUE(nullptr == a2);
}

TEST_F(ConstArrayViewTest, Compare_std_string) {
  const_string_view csv(S234);
  EXPECT_TRUE(csv == S234);
  EXPECT_FALSE(csv != S234);
  EXPECT_TRUE(S234 == csv);
  EXPECT_FALSE(S234 != csv);
}

TEST_F(ConstArrayViewTest, WriteReadView) {
  cavi_t a(VI234), read_a;

  OutputStringStream os;
  InputStringStream is;

  os << a;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_a);
  ASSERT_TRUE(is);

  EXPECT_EQ(a, read_a);
}

}  // namespace deepx_core
