// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/hash.h>
#include <deepx_core/common/hash_map.h>
#include <deepx_core/common/hash_map_io.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <algorithm>  // std::equal
#include <cstring>    // std::memcpy
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace deepx_core {

template <typename T, typename I>
class SparseRowGrad;
template <typename T, typename I>
class SRGIterator;
template <typename T, typename I>
class SRGConstIterator;

/************************************************************************/
/* SRGIterator */
/************************************************************************/
template <typename T, typename I>
class SRGIterator {
 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using srg_t = SparseRowGrad<float_t, int_t>;
  using map_t = HashMap<int_t, std::size_t, MurmurHash<int_t>>;
  using raw_iterator_t = typename map_t::iterator;
  using value_type = std::pair<int_t, ptr_t>;
  using _iterator = SRGIterator;
  using _const_iterator = SRGConstIterator<float_t, int_t>;
  friend _const_iterator;

 private:
  srg_t* srg_ = nullptr;
  raw_iterator_t it_;
  value_type value_;

 private:
  inline void set_value() noexcept;

 public:
  SRGIterator() = default;
  inline SRGIterator(srg_t* srg, raw_iterator_t&& it) noexcept;
  inline bool operator==(const _iterator& right) const noexcept;
  inline bool operator!=(const _iterator& right) const noexcept;
  inline bool operator==(const _const_iterator& right) const noexcept;
  inline bool operator!=(const _const_iterator& right) const noexcept;
  inline _iterator& operator++() noexcept;
  inline _iterator operator++(int) noexcept;
  value_type& operator*() noexcept { return value_; }
  value_type* operator->() noexcept { return &value_; }
};

/************************************************************************/
/* SRGConstIterator */
/************************************************************************/
template <typename T, typename I>
class SRGConstIterator {
 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using srg_t = SparseRowGrad<float_t, int_t>;
  using map_t = HashMap<int_t, std::size_t, MurmurHash<int_t>>;
  using raw_iterator_t = typename map_t::iterator;
  using raw_const_iterator_t = typename map_t::const_iterator;
  using value_type = std::pair<int_t, cptr_t>;
  using _iterator = SRGIterator<float_t, int_t>;
  using _const_iterator = SRGConstIterator;
  friend _iterator;

 private:
  const srg_t* srg_ = nullptr;
  raw_const_iterator_t it_;
  value_type value_;

 private:
  inline void set_value() noexcept;

 public:
  SRGConstIterator() = default;
  inline SRGConstIterator(const srg_t* srg, raw_iterator_t&& it) noexcept;
  inline SRGConstIterator(const srg_t* srg, raw_const_iterator_t&& it) noexcept;
  inline SRGConstIterator(const _iterator& it) noexcept;  // NOLINT
  inline bool operator==(const _iterator& right) const noexcept;
  inline bool operator!=(const _iterator& right) const noexcept;
  inline bool operator==(const _const_iterator& right) const noexcept;
  inline bool operator!=(const _const_iterator& right) const noexcept;
  inline _const_iterator& operator++() noexcept;
  inline _const_iterator operator++(int) noexcept;
  const value_type& operator*() const noexcept { return value_; }
  const value_type* operator->() const noexcept { return &value_; }
};

/************************************************************************/
/* SparseRowGrad */
/************************************************************************/
template <typename T, typename I>
class SparseRowGrad {
 private:
  using map_t = HashMap<I, std::size_t, MurmurHash<I>>;

 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using key_type = typename map_t::key_type;
  using mapped_type = typename map_t::mapped_type;
  using value_type = typename map_t::value_type;

 private:
  int col_ = 0;
  std::vector<float_t> value_;
  map_t row_offset_map_;

  template <typename T2, typename I2>
  friend OutputStream& operator<<(OutputStream& os,
                                  const SparseRowGrad<T2, I2>& srg);
  template <typename T2, typename I2>
  friend InputStream& operator>>(InputStream& is, SparseRowGrad<T2, I2>& srg);

 public:
  void set_col(int col) noexcept { col_ = col; }
  int col() const noexcept { return col_; }

 public:
  SparseRowGrad() = default;
  SparseRowGrad(
      std::initializer_list<int_t> rows,
      std::initializer_list<std::initializer_list<float_t>> row_values);

 public:
  template <typename Int>
  void reserve(Int size);
  void clear() noexcept;
  void zeros() noexcept;
  std::size_t size() const noexcept { return row_offset_map_.size(); }
  bool empty() const noexcept { return row_offset_map_.empty(); }
  void assign(int_t row, cptr_t row_value);
  template <class Func>
  void remove_if(Func&& func);
  void remove_zeros();

 public:
  inline ptr_t get_row_no_init(int_t row);

 public:
  // iterator
  using iterator = SRGIterator<float_t, int_t>;
  using const_iterator = SRGConstIterator<float_t, int_t>;
  friend iterator;
  friend const_iterator;
  iterator find(int_t row) noexcept {
    return iterator(this, row_offset_map_.find(row));
  }
  const_iterator find(int_t row) const noexcept {
    return const_iterator(this, row_offset_map_.find(row));
  }
  iterator begin() noexcept { return iterator(this, row_offset_map_.begin()); }
  const_iterator begin() const noexcept {
    return const_iterator(this, row_offset_map_.begin());
  }
  const_iterator cbegin() const noexcept {
    return const_iterator(this, row_offset_map_.cbegin());
  }
  iterator end() noexcept { return iterator(this, row_offset_map_.end()); }
  const_iterator end() const noexcept {
    return const_iterator(this, row_offset_map_.end());
  }
  const_iterator cend() const noexcept {
    return const_iterator(this, row_offset_map_.cend());
  }

 public:
  // comparison
  bool operator==(const SparseRowGrad& right) const noexcept;
  bool operator!=(const SparseRowGrad& right) const noexcept {
    return !(operator==(right));
  }
};

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseRowGrad<T, I>& srg);

/************************************************************************/
/* SparseRowGrad */
/************************************************************************/
template <typename T, typename I>
OutputStream& operator<<(OutputStream& os, const SparseRowGrad<T, I>& srg) {
  os << srg.col_ << srg.value_ << srg.row_offset_map_;
  return os;
}

template <typename T, typename I>
InputStream& operator>>(InputStream& is, SparseRowGrad<T, I>& srg) {
  is >> srg.col_ >> srg.value_ >> srg.row_offset_map_;
  return is;
}

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseRowGrad<T, I>& srg) {
  os << "(0," << srg.col() << ")" << std::endl;
  for (const auto& entry : srg) {
    os << "row " << entry.first << ":";
    for (int i = 0; i < srg.col(); ++i) {
      os << " " << entry.second[i];
    }
    os << std::endl;
  }
  return os;
}

template <typename T, typename I>
SparseRowGrad<T, I>::SparseRowGrad(
    std::initializer_list<int_t> rows,
    std::initializer_list<std::initializer_list<float_t>> row_values) {
  if (rows.size() == 0) {
    DXTHROW_INVALID_ARGUMENT("Invalid rows.");
  }

  if (rows.size() != row_values.size()) {
    DXTHROW_INVALID_ARGUMENT("Invalid rows and row_values.");
  }

  int _col = (int)(*row_values.begin()).size();
  if (_col == 0) {
    DXTHROW_INVALID_ARGUMENT("Invalid row_values.");
  }

  for (const auto& row_value : row_values) {
    if ((int)row_value.size() != _col) {
      DXTHROW_INVALID_ARGUMENT("Invalid row_values.");
    }
  }

  set_col(_col);
  reserve(row_values.size());

  auto first_row = rows.begin();
  auto last_row = rows.end();
  auto first_value = row_values.begin();
  for (; first_row != last_row; ++first_row, ++first_value) {
    assign(*first_row, first_value->begin());
  }
}

template <typename T, typename I>
template <typename Int>
void SparseRowGrad<T, I>::reserve(Int size) {
  value_.reserve((std::size_t)(size * col_));
  row_offset_map_.reserve((std::size_t)size);
}

template <typename T, typename I>
void SparseRowGrad<T, I>::clear() noexcept {
  col_ = 0;
  value_.clear();
  row_offset_map_.clear();
}

template <typename T, typename I>
void SparseRowGrad<T, I>::zeros() noexcept {
  value_.clear();
  row_offset_map_.clear();
}

template <typename T, typename I>
void SparseRowGrad<T, I>::assign(int_t row, cptr_t row_value) {
  std::memcpy(get_row_no_init(row), row_value, col_ * sizeof(float_t));
}

template <typename T, typename I>
template <class Func>
void SparseRowGrad<T, I>::remove_if(Func&& func) {
  auto first = row_offset_map_.begin();
  auto last = row_offset_map_.end();
  for (; first != last;) {
    if (func(*first)) {
      first = row_offset_map_.erase(first);
    } else {
      ++first;
    }
  }
}

template <typename T, typename I>
void SparseRowGrad<T, I>::remove_zeros() {
  remove_if([this](const value_type& entry) {
    for (int i = 0; i < col(); ++i) {
      if (value_[entry.second + (std::size_t)i] != 0) {
        return false;
      }
    }
    return true;
  });
}

template <typename T, typename I>
inline auto SparseRowGrad<T, I>::get_row_no_init(int_t row) -> ptr_t {
  auto it = row_offset_map_.find(row);
  if (it != row_offset_map_.end()) {
    return &value_[it->second];
  }

  std::size_t offset = value_.size();
  value_.resize(offset + col_);
  row_offset_map_.emplace(row, offset);
  return &value_[offset];
}

template <typename T, typename I>
bool SparseRowGrad<T, I>::operator==(const SparseRowGrad<T, I>& right) const
    noexcept {
  if (col() != right.col()) {
    return false;
  }

  if (size() != right.size()) {
    return false;
  }

  for (const auto& entry : *this) {
    int_t row = entry.first;
    auto it = right.find(row);
    if (it == right.end()) {
      return false;
    }

    cptr_t left_row_value = entry.second;
    cptr_t right_row_value = it->second;
    if (!std::equal(left_row_value, left_row_value + col(), right_row_value)) {
      return false;
    }
  }
  return true;
}

/************************************************************************/
/* SRGIterator */
/************************************************************************/
template <typename T, typename I>
inline void SRGIterator<T, I>::set_value() noexcept {
  if (it_ != srg_->row_offset_map_.end()) {
    value_.first = it_->first;
    value_.second = &srg_->value_[it_->second];
  } else {
    // ZERO is only to eliminate clang-tidy warnings.
    static float_t ZERO = 0;
    value_.first = 0;
    value_.second = &ZERO;
  }
}

template <typename T, typename I>
inline SRGIterator<T, I>::SRGIterator(srg_t* srg, raw_iterator_t&& it) noexcept
    : srg_(srg), it_(it) {
  set_value();
}

template <typename T, typename I>
inline bool SRGIterator<T, I>::operator==(const _iterator& right) const
    noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRGIterator<T, I>::operator!=(const _iterator& right) const
    noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline bool SRGIterator<T, I>::operator==(const _const_iterator& right) const
    noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRGIterator<T, I>::operator!=(const _const_iterator& right) const
    noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline auto SRGIterator<T, I>::operator++() noexcept -> _iterator& {
  ++it_;
  set_value();
  return *this;
}

template <typename T, typename I>
inline auto SRGIterator<T, I>::operator++(int) noexcept -> _iterator {
  _iterator origin = *this;
  operator++();
  return origin;
}

/************************************************************************/
/* SRGConstIterator */
/************************************************************************/
template <typename T, typename I>
inline void SRGConstIterator<T, I>::set_value() noexcept {
  if (it_ != srg_->row_offset_map_.end()) {
    value_.first = it_->first;
    value_.second = &srg_->value_[it_->second];
  } else {
    // ZERO is only to eliminate clang-tidy warnings.
    static float_t ZERO = 0;
    value_.first = 0;
    value_.second = &ZERO;
  }
}

template <typename T, typename I>
inline SRGConstIterator<T, I>::SRGConstIterator(const srg_t* srg,
                                                raw_iterator_t&& it) noexcept
    : srg_(srg), it_(it) {
  set_value();
}

template <typename T, typename I>
inline SRGConstIterator<T, I>::SRGConstIterator(
    const srg_t* srg, raw_const_iterator_t&& it) noexcept
    : srg_(srg), it_(it) {
  set_value();
}

template <typename T, typename I>
inline SRGConstIterator<T, I>::SRGConstIterator(const _iterator& it) noexcept
    : srg_(it.srg_), it_(it.it_) {
  set_value();
}

template <typename T, typename I>
inline bool SRGConstIterator<T, I>::operator==(const _iterator& right) const
    noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRGConstIterator<T, I>::operator!=(const _iterator& right) const
    noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline bool SRGConstIterator<T, I>::operator==(
    const _const_iterator& right) const noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRGConstIterator<T, I>::operator!=(
    const _const_iterator& right) const noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline auto SRGConstIterator<T, I>::operator++() noexcept -> _const_iterator& {
  ++it_;
  set_value();
  return *this;
}

template <typename T, typename I>
inline auto SRGConstIterator<T, I>::operator++(int) noexcept
    -> _const_iterator {
  _const_iterator origin = *this;
  operator++();
  return origin;
}

}  // namespace deepx_core
