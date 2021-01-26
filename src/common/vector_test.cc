// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/common/vector.h>
#include <deepx_core/common/vector_io.h>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace deepx_core {

class VectorTest : public testing::Test {
 protected:
  using vi_t = Vector<int>;
  using vs_t = Vector<std::string>;
  static const std::vector<int> VI234;
  static const std::vector<int> VI567;
  static const std::vector<std::string> VS234;
  static const std::vector<std::string> VS567;
};

const std::vector<int> VectorTest::VI234{2, 3, 4};
const std::vector<int> VectorTest::VI567{5, 6, 7};
const std::vector<std::string> VectorTest::VS234{"2", "3", "4"};
const std::vector<std::string> VectorTest::VS567{"5", "6", "7"};

TEST_F(VectorTest, Construct_n) {
  vi_t v(2);
  EXPECT_FALSE(v.empty());
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v, vi_t({0, 0}));
}

TEST_F(VectorTest, Construct_nv) {
  vi_t v(2, 1);
  EXPECT_EQ(v, vi_t({1, 1}));
}

TEST_F(VectorTest, Construct_ii) {
  vi_t v(VI234.begin(), VI234.end());
  EXPECT_EQ(v, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, Copy) {
  vi_t v1(VI234);
  vi_t v2(v1);
  vi_t v3;
  v3 = v2;
  EXPECT_FALSE(v1.is_view());
  EXPECT_FALSE(v2.is_view());
  EXPECT_FALSE(v3.is_view());
  EXPECT_EQ(v1, v2);
  EXPECT_EQ(v1, v3);
}

TEST_F(VectorTest, Copy_view) {
  vi_t v1;
  v1.view(VI234.data(), VI234.size());
  vi_t v2(v1);
  vi_t v3;
  v3 = v2;
  EXPECT_TRUE(v1.is_view());
  EXPECT_TRUE(v2.is_view());
  EXPECT_TRUE(v3.is_view());
  EXPECT_EQ(v1, v2);
  EXPECT_EQ(v1, v3);
}

TEST_F(VectorTest, Move) {
  vi_t v1(VI234);
  EXPECT_FALSE(v1.is_view());
  EXPECT_EQ(v1, vi_t({2, 3, 4}));

  vi_t v2(std::move(v1));
  EXPECT_FALSE(v2.is_view());
  EXPECT_EQ(v2, vi_t({2, 3, 4}));

  vi_t v3;
  v3 = std::move(v2);
  EXPECT_FALSE(v3.is_view());
  EXPECT_EQ(v3, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, Move_view) {
  vi_t v1;
  v1.view(VI234.data(), VI234.size());
  EXPECT_TRUE(v1.is_view());
  EXPECT_EQ(v1, vi_t({2, 3, 4}));

  vi_t v2(std::move(v1));
  EXPECT_TRUE(v2.is_view());
  EXPECT_EQ(v2, vi_t({2, 3, 4}));

  vi_t v3;
  v3 = std::move(v2);
  EXPECT_TRUE(v3.is_view());
  EXPECT_EQ(v3, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, Construct_std_il) {
  vi_t v1{2, 3, 4};
  EXPECT_EQ(v1, vi_t({2, 3, 4}));

  vi_t v2;
  v2 = {5, 6, 7};
  EXPECT_EQ(v2, vi_t({5, 6, 7}));
}

TEST_F(VectorTest, Construct_std_vector) {
  vi_t v1(VI234);
  EXPECT_EQ(v1, vi_t({2, 3, 4}));

  vi_t v2;
  v2 = VI567;
  EXPECT_EQ(v2, vi_t({5, 6, 7}));
}

TEST_F(VectorTest, Construct_rvalue_std_vector) {
  std::vector<int>&& RVALUE_VI234 = std::vector<int>(VI234);
  std::vector<int>&& RVALUE_VI567 = std::vector<int>(VI567);
  vi_t v1(std::move(RVALUE_VI234));
  EXPECT_EQ(v1, vi_t({2, 3, 4}));

  vi_t v2;
  v2 = std::move(RVALUE_VI567);
  EXPECT_EQ(v2, vi_t({5, 6, 7}));
}

TEST_F(VectorTest, assign_nv) {
  vi_t v;
  v.assign(2, 1);
  EXPECT_EQ(v, vi_t({1, 1}));
}

TEST_F(VectorTest, assign_ii) {
  vi_t v;
  v.assign(VI234.begin(), VI234.end());
  EXPECT_EQ(v, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, assign_std_il) {
  vi_t v;
  v.assign({2, 3, 4});
  EXPECT_EQ(v, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, iterator) {
  vi_t v(VI234);
  int sum = 0;
  for (int i : v) {
    sum += i;
  }
  EXPECT_EQ(sum, 9);

  const vi_t& cv = v;
  sum = 0;
  for (int i : cv) {
    sum += i;
  }
  EXPECT_EQ(sum, 9);
}

TEST_F(VectorTest, iterator_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  const vi_t& cv = v;

  int sum = 0;
  for (int i : v) {
    sum += i;
  }
  EXPECT_EQ(sum, 9);

  sum = 0;
  for (int i : cv) {
    sum += i;
  }
  EXPECT_EQ(sum, 9);
}

TEST_F(VectorTest, ElementAccess) {
  vi_t v(VI234);
  EXPECT_EQ(v.at(0), 2);
  EXPECT_EQ(v.at(1), 3);
  EXPECT_EQ(v.at(2), 4);
  EXPECT_ANY_THROW(v.at(3));
  EXPECT_EQ(v[0], 2);
  EXPECT_EQ(v[1], 3);
  EXPECT_EQ(v[2], 4);
  EXPECT_EQ(v.front(), 2);
  EXPECT_EQ(v.back(), 4);

  const vi_t& cv = v;
  EXPECT_EQ(cv.at(0), 2);
  EXPECT_EQ(cv.at(1), 3);
  EXPECT_EQ(cv.at(2), 4);
  EXPECT_ANY_THROW(cv.at(3));
  EXPECT_EQ(cv[0], 2);
  EXPECT_EQ(cv[1], 3);
  EXPECT_EQ(cv[2], 4);
  EXPECT_EQ(cv.front(), 2);
  EXPECT_EQ(cv.back(), 4);
}

TEST_F(VectorTest, ElementAccess_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_EQ(v.at(0), 2);
  EXPECT_EQ(v.at(1), 3);
  EXPECT_EQ(v.at(2), 4);
  EXPECT_ANY_THROW(v.at(3));
  EXPECT_EQ(v[0], 2);
  EXPECT_EQ(v[1], 3);
  EXPECT_EQ(v[2], 4);
  EXPECT_EQ(v.front(), 2);
  EXPECT_EQ(v.back(), 4);

  const vi_t& cv = v;
  EXPECT_EQ(cv.at(0), 2);
  EXPECT_EQ(cv.at(1), 3);
  EXPECT_EQ(cv.at(2), 4);
  EXPECT_ANY_THROW(cv.at(3));
  EXPECT_EQ(cv[0], 2);
  EXPECT_EQ(cv[1], 3);
  EXPECT_EQ(cv[2], 4);
  EXPECT_EQ(cv.front(), 2);
  EXPECT_EQ(cv.back(), 4);
}

TEST_F(VectorTest, reserve) {
  vi_t v;
  EXPECT_EQ(v.capacity(), 0u);
  v.reserve(3);
  EXPECT_EQ(v.capacity(), 3u);
}

TEST_F(VectorTest, reserve_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.reserve(3));
}

TEST_F(VectorTest, shrink_to_fit) {
  vi_t v(VI234);
  EXPECT_EQ(v.size(), 3u);
  EXPECT_GE(v.capacity(), 3u);
  v.pop_back();
  EXPECT_EQ(v.size(), 2u);
  EXPECT_GE(v.capacity(), 3u);
  v.shrink_to_fit();
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v.capacity(), 2u);
}

TEST_F(VectorTest, shrink_to_fit_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.shrink_to_fit());
}

TEST_F(VectorTest, get_view) {
  vi_t v1(VI234);
  vi_t v2 = v1.get_view();
  vi_t v3 = v2.get_view();
  EXPECT_FALSE(v1.is_view());
  EXPECT_TRUE(v2.is_view());
  EXPECT_TRUE(v3.is_view());
  EXPECT_EQ(v1, v2);
  EXPECT_EQ(v2, v3);
}

TEST_F(VectorTest, view) {
  vi_t v;
  EXPECT_FALSE(v.is_view());
  v.view(VI234.data(), VI234.size());
  EXPECT_TRUE(v.is_view());
  EXPECT_EQ(v, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, clear) {
  vi_t v(VI234);
  EXPECT_FALSE(v.empty());
  v.clear();
  EXPECT_TRUE(v.empty());
}

TEST_F(VectorTest, clear_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_FALSE(v.empty());
  v.clear();
  EXPECT_TRUE(v.empty());
}

TEST_F(VectorTest, insert_v) {
  int ONE = 1;
  int TWO = 2;
  vi_t v;
  auto it = v.insert(v.begin(), ONE);
  EXPECT_EQ(v, vi_t({1}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), TWO);
  EXPECT_EQ(v, vi_t({2, 1}));
  EXPECT_EQ(it, v.begin());
}

TEST_F(VectorTest, insert_v_view) {
  int ONE = 1;
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.insert(v.begin(), ONE));
}

TEST_F(VectorTest, insert_rvalue_v) {
  int&& RVALUE_ONE = 1;
  int&& RVALUE_TWO = 2;
  vi_t v;
  auto it = v.insert(v.begin(), RVALUE_ONE);
  EXPECT_EQ(v, vi_t({1}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), RVALUE_TWO);
  EXPECT_EQ(v, vi_t({2, 1}));
  EXPECT_EQ(it, v.begin());
}

TEST_F(VectorTest, insert_rvalue_v_view) {
  int&& RVALUE_ONE = 1;
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.insert(v.begin(), RVALUE_ONE));
}

TEST_F(VectorTest, insert_nv) {
  vi_t v;
  auto it = v.insert(v.begin(), 2, 1);
  EXPECT_EQ(v, vi_t({1, 1}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), 2, 2);
  EXPECT_EQ(v, vi_t({2, 2, 1, 1}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), 0, 3);
  EXPECT_EQ(it, v.begin());
}

TEST_F(VectorTest, insert_nv_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.insert(v.begin(), 2, 1));
}

TEST_F(VectorTest, insert_ii) {
  vi_t v;
  auto it = v.insert(v.begin(), VI234.begin(), VI234.end());
  EXPECT_EQ(v, vi_t({2, 3, 4}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), VI567.begin(), VI567.end());
  EXPECT_EQ(v, vi_t({5, 6, 7, 2, 3, 4}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), VI234.begin(), VI234.begin());
  EXPECT_EQ(it, v.begin());
}

TEST_F(VectorTest, insert_ii_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.insert(v.begin(), VI234.begin(), VI234.end()));
}

TEST_F(VectorTest, insert_std_il) {
  vi_t v;
  auto it = v.insert(v.begin(), {2, 3, 4});
  EXPECT_EQ(v, vi_t({2, 3, 4}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), {5, 6, 7});
  EXPECT_EQ(v, vi_t({5, 6, 7, 2, 3, 4}));
  EXPECT_EQ(it, v.begin());
  it = v.insert(v.begin(), {});
  EXPECT_EQ(it, v.begin());
}

TEST_F(VectorTest, insert_std_il_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.insert(v.begin(), {2, 3, 4}));
}

TEST_F(VectorTest, emplace) {
  vi_t v;
  auto it = v.emplace(v.begin(), 1);
  EXPECT_EQ(v, vi_t({1}));
  EXPECT_EQ(it, v.begin());
  it = v.emplace(v.begin(), 2);
  EXPECT_EQ(v, vi_t({2, 1}));
  EXPECT_EQ(it, v.begin());
}

TEST_F(VectorTest, emplace_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.emplace(v.begin(), 1));
}

TEST_F(VectorTest, erase) {
  vi_t v(VI234);
  auto it = v.erase(v.begin() + 1);
  EXPECT_EQ(v, vi_t({2, 4}));
  EXPECT_EQ(it, v.begin() + 1);
  it = v.erase(v.begin());
  EXPECT_EQ(v, vi_t({4}));
  EXPECT_EQ(it, v.begin());
  it = v.erase(v.begin());
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(it, v.end());
}

TEST_F(VectorTest, erase_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.erase(v.begin() + 1));
}

TEST_F(VectorTest, erase_range) {
  vi_t v(VI234);
  auto it = v.erase(v.begin(), v.begin() + 1);
  EXPECT_EQ(v, vi_t({3, 4}));
  EXPECT_EQ(it, v.begin());
  it = v.erase(v.begin(), v.end());
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(it, v.end());
}

TEST_F(VectorTest, erase_range_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.erase(v.begin(), v.begin() + 1));
}

TEST_F(VectorTest, emplace_back) {
  vi_t v;
  v.emplace_back(1);
  EXPECT_EQ(v, vi_t({1}));
  v.emplace_back(2);
  EXPECT_EQ(v, vi_t({1, 2}));
}

TEST_F(VectorTest, emplace_back_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.emplace_back(1));
}

TEST_F(VectorTest, pop_back) {
  vi_t v(VI234);
  v.pop_back();
  EXPECT_EQ(v, vi_t({2, 3}));
  v.pop_back();
  EXPECT_EQ(v, vi_t({2}));
  v.pop_back();
  EXPECT_TRUE(v.empty());
}

TEST_F(VectorTest, pop_back_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.pop_back());
}

TEST_F(VectorTest, resize_n) {
  vi_t v;
  v.resize(2);
  EXPECT_EQ(v, vi_t({0, 0}));
  v.resize(4);
  EXPECT_EQ(v, vi_t({0, 0, 0, 0}));
  v.resize(3);
  EXPECT_EQ(v, vi_t({0, 0, 0}));
  v.resize(0);
  EXPECT_TRUE(v.empty());
}

TEST_F(VectorTest, resize_n_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.resize(2));
}

TEST_F(VectorTest, resize_nv) {
  vi_t v;
  v.resize(2, 1);
  EXPECT_EQ(v, vi_t({1, 1}));
  v.resize(4, 2);
  EXPECT_EQ(v, vi_t({1, 1, 2, 2}));
  v.resize(3, 3);
  EXPECT_EQ(v, vi_t({1, 1, 2}));
  v.resize(0);
  EXPECT_TRUE(v.empty());
}

TEST_F(VectorTest, resize_nv_view) {
  vi_t v;
  v.view(VI234.data(), VI234.size());
  EXPECT_ANY_THROW(v.resize(2, 1));
}

TEST_F(VectorTest, swap) {
  vi_t v1(VI234);
  vi_t v2(VI567);
  v1.swap(v2);
  EXPECT_EQ(v1, vi_t({5, 6, 7}));
  EXPECT_EQ(v2, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, swap_view) {
  vi_t v1;
  vi_t v2;
  v1.view(VI234.data(), VI234.size());
  v2.view(VI567.data(), VI567.size());
  v1.swap(v2);
  EXPECT_EQ(v1, vi_t({5, 6, 7}));
  EXPECT_EQ(v2, vi_t({2, 3, 4}));
}

TEST_F(VectorTest, Compare) {
  vi_t v1(VI234);
  vi_t v2(VI567);
  EXPECT_TRUE(v1 != v2);
  EXPECT_FALSE(v1 == v2);
}

TEST_F(VectorTest, Compare_view) {
  vi_t v1;
  vi_t v2;
  v1.view(VI234.data(), VI234.size());
  v2.view(VI567.data(), VI567.size());
  EXPECT_TRUE(v1 != v2);
  EXPECT_FALSE(v1 == v2);
}

TEST_F(VectorTest, Compare_std_nullptr_t) {
  vi_t v1(VI234);
  vi_t v2;
  EXPECT_TRUE(v1 != nullptr);
  EXPECT_FALSE(v1 == nullptr);
  EXPECT_FALSE(nullptr != v2);
  EXPECT_TRUE(nullptr == v2);
}

TEST_F(VectorTest, Compare_std_nullptr_t_view) {
  vi_t v1;
  vi_t v2;
  v1.view(VI234.data(), VI234.size());
  EXPECT_TRUE(v1 != nullptr);
  EXPECT_FALSE(v1 == nullptr);
  EXPECT_FALSE(nullptr != v2);
  EXPECT_TRUE(nullptr == v2);
}

TEST_F(VectorTest, WriteRead_int) {
  vi_t v(VI234), read_v;

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_v;
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_FALSE(read_v.is_view());
}

TEST_F(VectorTest, WriteRead_int_view) {
  vi_t v, read_v;
  v.view(VI234.data(), VI234.size());
  read_v.view(VI567.data(), VI567.size());

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_v;
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_FALSE(read_v.is_view());
}

TEST_F(VectorTest, WriteReadView_int) {
  vi_t v(VI234), read_v;

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_v);
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_TRUE(read_v.is_view());
}

TEST_F(VectorTest, WriteReadView_int_view) {
  vi_t v, read_v;
  v.view(VI234.data(), VI234.size());
  read_v.view(VI567.data(), VI567.size());

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_v);
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_TRUE(read_v.is_view());
}

TEST_F(VectorTest, WriteRead_std_string) {
  vs_t v(VS234), read_v;

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_v;
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_FALSE(read_v.is_view());
}

TEST_F(VectorTest, WriteRead_std_string_view) {
  vs_t v, read_v;
  v.view(VS234.data(), VS234.size());
  read_v.view(VS567.data(), VS567.size());

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_v;
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_FALSE(read_v.is_view());
}

TEST_F(VectorTest, WriteReadView_std_string) {
  vs_t v(VS234), read_v;

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_v);
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_FALSE(read_v.is_view());
}

TEST_F(VectorTest, WriteReadView_std_string_view) {
  vs_t v, read_v;
  v.view(VS234.data(), VS234.size());
  read_v.view(VS567.data(), VS567.size());

  OutputStringStream os;
  InputStringStream is;

  os << v;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_v);
  ASSERT_TRUE(is);

  EXPECT_EQ(v, read_v);
  EXPECT_FALSE(read_v.is_view());
}

}  // namespace deepx_core
