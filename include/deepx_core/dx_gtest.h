// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <gtest/gtest.h>
#include <cmath>  // std::fabs

#define EXPECT_DOUBLE_NEAR_EPS(left, right, eps)                     \
  do {                                                               \
    auto __l = (double)(left);                                       \
    auto __r = (double)(right);                                      \
    auto __eps = (double)(eps);                                      \
    auto __a = std::fabs(__l) + std::fabs(__r);                      \
    auto __b = std::fabs(__l - __r) / ((__a < __eps) ? __eps : __a); \
    EXPECT_LE(__b, __eps) << __l << " != " << __r;                   \
  } while (0)

#define EXPECT_ARRAY_NEAR_EPS(left, right, size, eps)    \
  do {                                                   \
    const auto& __la = (left);                           \
    const auto& __ra = (right);                          \
    auto __size = (int)(size);                           \
    for (int __i = 0; __i < __size; ++__i) {             \
      EXPECT_DOUBLE_NEAR_EPS(__la[__i], __ra[__i], eps); \
    }                                                    \
  } while (0)

#define EXPECT_VECTOR_NEAR_EPS(left, right, eps)                 \
  do {                                                           \
    const auto& __lv = (left);                                   \
    const auto& __rv = (right);                                  \
    auto __lsize = __lv.size();                                  \
    auto __rsize = __rv.size();                                  \
    EXPECT_EQ(__lsize, __rsize) << __lsize << " != " << __rsize; \
    if (__lsize == __rsize) {                                    \
      EXPECT_ARRAY_NEAR_EPS(__lv, __rv, __lsize, eps);           \
    }                                                            \
  } while (0)

#define EXPECT_TSR_NEAR_EPS(left, right, eps)                           \
  do {                                                                  \
    const auto& _l = (left);                                            \
    const auto& _r = (right);                                           \
    EXPECT_EQ(_l.shape(), _r.shape())                                   \
        << to_string(_l.shape()) << " != " << to_string(_r.shape());    \
    if (_l.shape() != _r.shape()) {                                     \
      break;                                                            \
    }                                                                   \
                                                                        \
    if (!_l.empty()) {                                                  \
      EXPECT_ARRAY_NEAR_EPS(_l.data(), _r.data(), _l.total_dim(), eps); \
    }                                                                   \
  } while (0)

#define EXPECT_SRM_NEAR_EPS(left, right, eps)                        \
  do {                                                               \
    const auto& _l = (left);                                         \
    const auto& _r = (right);                                        \
    EXPECT_EQ(_l.col(), _r.col()) << _l.col() << " != " << _r.col(); \
    if (_l.col() != _r.col()) {                                      \
      break;                                                         \
    }                                                                \
                                                                     \
    auto _lsize = _l.size();                                         \
    auto _rsize = _r.size();                                         \
    EXPECT_EQ(_lsize, _rsize) << _lsize << " != " << _rsize;         \
    if (_lsize != _rsize) {                                          \
      break;                                                         \
    }                                                                \
                                                                     \
    for (const auto& _entry : _l) {                                  \
      auto _id = _entry.first;                                       \
      auto _it = _r.find(_id);                                       \
      EXPECT_TRUE(_it != _r.end());                                  \
      if (_it != _r.end()) {                                         \
        const auto* _lvalue = _entry.second;                         \
        const auto* _rvalue = _it->second;                           \
        EXPECT_ARRAY_NEAR_EPS(_lvalue, _rvalue, _l.col(), eps);      \
      }                                                              \
    }                                                                \
  } while (0)

#define EXPECT_DOUBLE_EPS (1e-3)
#define EXPECT_DOUBLE_NEAR(l, r) EXPECT_DOUBLE_NEAR_EPS(l, r, EXPECT_DOUBLE_EPS)
#define EXPECT_ARRAY_NEAR(l, r, size) \
  EXPECT_ARRAY_NEAR_EPS(l, r, size, EXPECT_DOUBLE_EPS)
#define EXPECT_VECTOR_NEAR(l, r) EXPECT_VECTOR_NEAR_EPS(l, r, EXPECT_DOUBLE_EPS)
#define EXPECT_TSR_NEAR(l, r) EXPECT_TSR_NEAR_EPS(l, r, EXPECT_DOUBLE_EPS)
#define EXPECT_SRM_NEAR(l, r) EXPECT_SRM_NEAR_EPS(l, r, EXPECT_DOUBLE_EPS)
