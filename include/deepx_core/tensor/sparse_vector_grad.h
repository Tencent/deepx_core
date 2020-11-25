// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/hash.h>
#include <deepx_core/common/hash_map.h>
#include <deepx_core/common/hash_map_io.h>
#include <deepx_core/common/stream.h>
#include <initializer_list>
#include <iostream>
#include <utility>

namespace deepx_core {

/************************************************************************/
/* SparseVectorGrad */
/************************************************************************/
template <typename T, typename I>
class SparseVectorGrad {
 private:
  using map_t = HashMap<I, T, MurmurHash<I>>;

 public:
  using float_t = T;
  using int_t = I;
  using key_type = typename map_t::key_type;
  using mapped_type = typename map_t::mapped_type;
  using value_type = typename map_t::value_type;

 private:
  map_t row_map_;

  template <typename T2, typename I2>
  friend OutputStream& operator<<(OutputStream& os,
                                  const SparseVectorGrad<T2, I2>& svg);
  template <typename T2, typename I2>
  friend InputStream& operator>>(InputStream& is,
                                 SparseVectorGrad<T2, I2>& svg);

 public:
  SparseVectorGrad() = default;
  SparseVectorGrad(std::initializer_list<std::pair<int_t, float_t>> il);

 public:
  template <typename Int>
  void reserve(Int size);
  void clear() noexcept { row_map_.clear(); }
  void zeros() noexcept { row_map_.clear(); }
  std::size_t size() const noexcept { return row_map_.size(); }
  bool empty() const noexcept { return row_map_.empty(); }
  void assign(int_t row, float_t row_value) { row_map_[row] = row_value; }
  template <class Func>
  void remove_if(Func&& func);
  void remove_zeros();

 public:
  float_t& get_scalar_no_init(int_t row) { return row_map_[row]; }

 public:
  // iterator
  using iterator = typename map_t::iterator;
  using const_iterator = typename map_t::const_iterator;
  iterator find(int_t row) noexcept { return row_map_.find(row); }
  const_iterator find(int_t row) const noexcept { return row_map_.find(row); }
  iterator begin() noexcept { return row_map_.begin(); }
  const_iterator begin() const noexcept { return row_map_.begin(); }
  const_iterator cbegin() const noexcept { return row_map_.cbegin(); }
  iterator end() noexcept { return row_map_.end(); }
  const_iterator end() const noexcept { return row_map_.end(); }
  const_iterator cend() const noexcept { return row_map_.cend(); }

 public:
  // comparison
  bool operator==(const SparseVectorGrad& right) const noexcept;
  bool operator!=(const SparseVectorGrad& right) const noexcept {
    return !(operator==(right));
  }
};

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseVectorGrad<T, I>& svg);

/************************************************************************/
/* SparseVectorGrad */
/************************************************************************/
template <typename T, typename I>
OutputStream& operator<<(OutputStream& os, const SparseVectorGrad<T, I>& svg) {
  os << svg.row_map_;
  return os;
}

template <typename T, typename I>
InputStream& operator>>(InputStream& is, SparseVectorGrad<T, I>& svg) {
  is >> svg.row_map_;
  return is;
}

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseVectorGrad<T, I>& svg) {
  os << "(0,1)" << std::endl;
  for (const auto& entry : svg) {
    os << "row " << entry.first << ":" << entry.second << std::endl;
  }
  return os;
}

template <typename T, typename I>
SparseVectorGrad<T, I>::SparseVectorGrad(
    std::initializer_list<std::pair<int_t, float_t>> il) {
  reserve(il.size());
  for (const auto& entry : il) {
    assign(entry.first, entry.second);
  }
}

template <typename T, typename I>
template <typename Int>
void SparseVectorGrad<T, I>::reserve(Int size) {
  row_map_.reserve((std::size_t)size);
}

template <typename T, typename I>
template <class Func>
void SparseVectorGrad<T, I>::remove_if(Func&& func) {
  auto first = row_map_.begin();
  auto last = row_map_.end();
  for (; first != last;) {
    if (func(*first)) {
      first = row_map_.erase(first);
    } else {
      ++first;
    }
  }
}

template <typename T, typename I>
void SparseVectorGrad<T, I>::remove_zeros() {
  remove_if([](const value_type& entry) { return entry.second == 0; });
}

template <typename T, typename I>
bool SparseVectorGrad<T, I>::operator==(const SparseVectorGrad& right) const
    noexcept {
  return row_map_ == right.row_map_;
}

}  // namespace deepx_core
