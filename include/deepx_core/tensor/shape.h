// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <initializer_list>
#include <iostream>
#include <iterator>  // std::distance, std::reverse_iterator
#include <string>
#include <type_traits>  // std::enable_if, ...
#include <utility>
#include <vector>

namespace deepx_core {

constexpr int SHAPE_DIM_ANY = -1;
constexpr int SHAPE_MAX_RANK = 8;
constexpr int SHAPE_INVALID_AXIS = -1;

/************************************************************************/
/* Shape */
/************************************************************************/
class Shape {
 private:
  int rank_ = 0;
  int total_dim_ = 0;
  int dim_[SHAPE_MAX_RANK] = {0};

  friend OutputStream& operator<<(OutputStream& os, const Shape& shape);
  friend InputStream& operator>>(InputStream& is, Shape& shape);

 public:
  int rank() const noexcept { return rank_; }
  int total_dim() const noexcept { return total_dim_; }
  const int* dim() const noexcept { return &dim_[0]; }
  template <typename Int>
  int dim(Int i) const noexcept {
    return dim_[i];
  }
  template <typename Int>
  int operator[](Int i) const noexcept {
    return dim_[i];
  }
  int front() const noexcept { return dim_[0]; }
  int back() const noexcept { return dim_[rank_ - 1]; }
  bool is_rank(int rank) const noexcept { return rank_ == rank; }
  bool is_scalar() const noexcept { return rank_ == 1 && dim_[0] == 1; }
  bool empty() const noexcept { return rank_ == 0; }

 public:
  Shape() = default;

  Shape(std::initializer_list<int> il);
  Shape& operator=(std::initializer_list<int> il);

  explicit Shape(const std::vector<int>& dim);
  Shape& operator=(const std::vector<int>& dim);

  template <typename II, typename = typename std::enable_if<
                             !std::is_integral<II>::value>::type>
  Shape(II first, II last) {
    assign(first, last);
  }

  explicit Shape(int d0) noexcept {
    rank_ = 1;
    total_dim_ = d0;
    dim_[0] = d0;
  }

  Shape(int d0, int d1) noexcept {
    rank_ = 2;
    total_dim_ = d0 * d1;
    dim_[0] = d0;
    dim_[1] = d1;
  }

  Shape(int d0, int d1, int d2) noexcept {
    rank_ = 3;
    total_dim_ = d0 * d1 * d2;
    dim_[0] = d0;
    dim_[1] = d1;
    dim_[2] = d2;
  }

  template <typename... Args>
  explicit Shape(int dim, Args&&... args) noexcept {
    static_assert(sizeof...(args) < SHAPE_MAX_RANK, "Too large rank.");
    total_dim_ = dim;
    dim_[rank_++] = dim;
    Construct(std::forward<Args>(args)...);
  }

 private:
  void Construct(int dim) noexcept;

  template <typename... Args>
  void Construct(int dim, Args&&... args) noexcept {
    total_dim_ *= dim;
    dim_[rank_++] = dim;
    Construct(std::forward<Args>(args)...);
  }

 public:
  template <typename II, typename = typename std::enable_if<
                             !std::is_integral<II>::value>::type>
  void assign(II first, II last) {
    int rank = (int)std::distance(first, last);
    if (rank == 0) {
      clear();
      return;
    }

    if (rank > SHAPE_MAX_RANK) {
      DXTHROW_INVALID_ARGUMENT("Too large rank: %d.", rank);
    }

    rank_ = rank;
    total_dim_ = 1;
    auto it = begin();
    for (; first != last; ++first) {
      *it = (int)*first;
      total_dim_ *= *it;
      ++it;
    }
  }

  void clear() noexcept {
    rank_ = 0;
    total_dim_ = 0;
  }

 public:
  bool real_axis(int* axis) const noexcept;
  int real_axis(int axis) const noexcept;

 public:
  Shape& resize(const Shape& other) { return *this = other; }

  Shape& resize(Shape& other) {  // NOLINT
    return *this = other;
  }

  Shape& resize(std::initializer_list<int> il) { return *this = il; }

  Shape& resize(int d0) noexcept {
    rank_ = 1;
    total_dim_ = d0;
    dim_[0] = d0;
    return *this;
  }

  Shape& resize(int d0, int d1) noexcept {
    rank_ = 2;
    total_dim_ = d0 * d1;
    dim_[0] = d0;
    dim_[1] = d1;
    return *this;
  }

  Shape& resize(int d0, int d1, int d2) noexcept {
    rank_ = 3;
    total_dim_ = d0 * d1 * d2;
    dim_[0] = d0;
    dim_[1] = d1;
    dim_[2] = d2;
    return *this;
  }

  template <typename... Args>
  Shape& resize(Args&&... args) {
    return *this = Shape(std::forward<Args>(args)...);
  }

 private:
  bool do_reshape_nothrow(const Shape& other) noexcept;
  bool do_expand_dim_nothrow(int axis) noexcept;
  bool do_squeeze_nothrow(int axis) noexcept;

 public:
  Shape& reshape(const Shape& other);

  Shape& reshape(Shape& other) {  // NOLINT
    return reshape((const Shape&)other);
  }

  Shape& reshape(std::initializer_list<int> il) {
    const Shape other(il);
    return reshape(other);
  }

  template <typename... Args>
  Shape& reshape(Args&&... args) {
    const Shape other(std::forward<Args>(args)...);
    return reshape(other);
  }

  Shape& reshape_nothrow(const Shape& other) noexcept;

  Shape& reshape_nothrow(Shape& other) noexcept {  // NOLINT
    return reshape_nothrow((const Shape&)other);
  }

  Shape& reshape_nothrow(std::initializer_list<int> il) noexcept {
    const Shape other(il);
    return reshape_nothrow(other);
  }

  template <typename... Args>
  Shape& reshape_nothrow(Args&&... args) noexcept {
    const Shape other(std::forward<Args>(args)...);
    return reshape_nothrow(other);
  }

  Shape& expand_dim(int axis);
  Shape& expand_dim_nothrow(int axis) noexcept;

  Shape& squeeze(int axis);
  Shape& squeeze_nothrow(int axis) noexcept;

 public:
  bool same_shape(const Shape& other) const noexcept;

  bool same_shape(Shape& other) const noexcept {  // NOLINT
    return same_shape((const Shape&)other);
  }

  bool same_shape(std::initializer_list<int> il) const {
    const Shape other(il);
    return same_shape(other);
  }

  template <typename... Args>
  bool same_shape(Args&&... args) const noexcept {
    const Shape other(std::forward<Args>(args)...);
    return same_shape(other);
  }

 public:
  // iterator
  using iterator = int*;
  using const_iterator = const int*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  iterator begin() noexcept { return &dim_[0]; }
  const_iterator begin() const noexcept { return &dim_[0]; }
  const_iterator cbegin() const noexcept { return &dim_[0]; }
  iterator end() noexcept { return &dim_[0] + rank_; }
  const_iterator end() const noexcept { return &dim_[0] + rank_; }
  const_iterator cend() const noexcept { return &dim_[0] + rank_; }
  reverse_iterator rbegin() noexcept { return (reverse_iterator(end())); }
  const_reverse_iterator rbegin() const noexcept {
    return (const_reverse_iterator(end()));
  }
  const_reverse_iterator crbegin() const noexcept {
    return (const_reverse_iterator(end()));
  }
  reverse_iterator rend() noexcept { return (reverse_iterator(begin())); }
  const_reverse_iterator rend() const noexcept {
    return (const_reverse_iterator(begin()));
  }
  const_reverse_iterator crend() const noexcept {
    return (const_reverse_iterator(begin()));
  }

 public:
  // comparison
  bool operator==(const Shape& other) const noexcept {
    return same_shape(other);
  }

  bool operator!=(const Shape& other) const noexcept {
    return !same_shape(other);
  }
};

std::string to_string(const Shape& shape);

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  os << to_string(shape);
  return os;
}

}  // namespace deepx_core

#define _SHAPE_CONCAT(x, y) x y
#define _SHAPE_COUNT_ARGS_IMPL2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
                                count, ...)                              \
  count
#define _SHAPE_COUNT_ARGS_IMPL1(args) _SHAPE_COUNT_ARGS_IMPL2 args
#define _SHAPE_COUNT_ARGS(...) \
  _SHAPE_COUNT_ARGS_IMPL1((__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#define _DXASSERT_SAME_SHAPE2(A, B) \
  do {                              \
    DXASSERT((A).same_shape(B));    \
  } while (0)
#define _DXASSERT_SAME_SHAPE3(A, B, C) \
  do {                                 \
    DXASSERT((A).same_shape(B));       \
    DXASSERT((A).same_shape(C));       \
  } while (0)
#define _DXASSERT_SAME_SHAPE4(A, B, C, D) \
  do {                                    \
    DXASSERT((A).same_shape(B));          \
    DXASSERT((A).same_shape(C));          \
    DXASSERT((A).same_shape(D));          \
  } while (0)
#define _DXASSERT_SAME_SHAPE5(A, B, C, D, E) \
  do {                                       \
    DXASSERT((A).same_shape(B));             \
    DXASSERT((A).same_shape(C));             \
    DXASSERT((A).same_shape(D));             \
    DXASSERT((A).same_shape(E));             \
  } while (0)
#define _DXASSERT_SAME_SHAPE6(A, B, C, D, E, F) \
  do {                                          \
    DXASSERT((A).same_shape(B));                \
    DXASSERT((A).same_shape(C));                \
    DXASSERT((A).same_shape(D));                \
    DXASSERT((A).same_shape(E));                \
    DXASSERT((A).same_shape(F));                \
  } while (0)
#define _DXASSERT_SAME_SHAPE7(A, B, C, D, E, F, G) \
  do {                                             \
    DXASSERT((A).same_shape(B));                   \
    DXASSERT((A).same_shape(C));                   \
    DXASSERT((A).same_shape(D));                   \
    DXASSERT((A).same_shape(E));                   \
    DXASSERT((A).same_shape(F));                   \
    DXASSERT((A).same_shape(G));                   \
  } while (0)
#define _DXASSERT_SAME_SHAPE8(A, B, C, D, E, F, G, H) \
  do {                                                \
    DXASSERT((A).same_shape(B));                      \
    DXASSERT((A).same_shape(C));                      \
    DXASSERT((A).same_shape(D));                      \
    DXASSERT((A).same_shape(E));                      \
    DXASSERT((A).same_shape(F));                      \
    DXASSERT((A).same_shape(G));                      \
    DXASSERT((A).same_shape(H));                      \
  } while (0)
#define _DXASSERT_SAME_SHAPE9(A, B, C, D, E, F, G, H, I) \
  do {                                                   \
    DXASSERT((A).same_shape(B));                         \
    DXASSERT((A).same_shape(C));                         \
    DXASSERT((A).same_shape(D));                         \
    DXASSERT((A).same_shape(E));                         \
    DXASSERT((A).same_shape(F));                         \
    DXASSERT((A).same_shape(G));                         \
    DXASSERT((A).same_shape(H));                         \
    DXASSERT((A).same_shape(I));                         \
  } while (0)
#define _DXASSERT_SAME_SHAPE10(A, B, C, D, E, F, G, H, I, J) \
  do {                                                       \
    DXASSERT((A).same_shape(B));                             \
    DXASSERT((A).same_shape(C));                             \
    DXASSERT((A).same_shape(D));                             \
    DXASSERT((A).same_shape(E));                             \
    DXASSERT((A).same_shape(F));                             \
    DXASSERT((A).same_shape(G));                             \
    DXASSERT((A).same_shape(H));                             \
    DXASSERT((A).same_shape(I));                             \
    DXASSERT((A).same_shape(J));                             \
  } while (0)
#define _DXASSERT_SAME_SHAPE_IMPL2(count) _DXASSERT_SAME_SHAPE##count
#define _DXASSERT_SAME_SHAPE_IMPL1(count) _DXASSERT_SAME_SHAPE_IMPL2(count)
#define _DXASSERT_SAME_SHAPE(count) _DXASSERT_SAME_SHAPE_IMPL1(count)
#define DXASSERT_SAME_SHAPE(...)                                      \
  _SHAPE_CONCAT(_DXASSERT_SAME_SHAPE(_SHAPE_COUNT_ARGS(__VA_ARGS__)), \
                (__VA_ARGS__))

#define DXASSERT_SAME_RANK(X, Y)        \
  do {                                  \
    DXASSERT((X).rank() == (Y).rank()); \
  } while (0)

#define DXASSERT_RANK(X, rank)   \
  do {                           \
    DXASSERT((X).is_rank(rank)); \
  } while (0)

#define DXASSERT_RANK1(X)     \
  do {                        \
    DXASSERT((X).is_rank(1)); \
  } while (0)

#define DXASSERT_RANK2(X)     \
  do {                        \
    DXASSERT((X).is_rank(2)); \
  } while (0)

#define DXASSERT_RANK3(X)     \
  do {                        \
    DXASSERT((X).is_rank(3)); \
  } while (0)

#define DXASSERT_SAME_TOTAL_DIM(X, Y)             \
  do {                                            \
    DXASSERT((X).total_dim() == (Y).total_dim()); \
  } while (0)

#define DXASSERT_TOTAL_DIM(X, _total_dim)    \
  do {                                       \
    DXASSERT((X).total_dim() == _total_dim); \
  } while (0)
