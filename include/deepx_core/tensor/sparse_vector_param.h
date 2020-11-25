// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/hash.h>
#include <deepx_core/common/hash_map.h>
#include <deepx_core/common/hash_map_io.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/tensor/tensor_type.h>
#include <initializer_list>
#include <iostream>
#include <random>
#include <utility>

namespace deepx_core {

/************************************************************************/
/* SparseVectorParam */
/************************************************************************/
template <typename T, typename I>
class SparseVectorParam {
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
  int initializer_type_ = TENSOR_INITIALIZER_TYPE_NONE;
  float_t initializer_param1_ = 0;
  float_t initializer_param2_ = 0;

  template <typename T2, typename I2>
  friend OutputStream& operator<<(OutputStream& os,
                                  const SparseVectorParam<T2, I2>& svp);
  template <typename T2, typename I2>
  friend InputStream& operator>>(InputStream& is,
                                 SparseVectorParam<T2, I2>& svp);

 public:
  void set_initializer(int initializer_type, float_t initializer_param1 = 0,
                       float_t initializer_param2 = 0);

 public:
  SparseVectorParam() = default;
  SparseVectorParam(std::initializer_list<std::pair<int_t, float_t>> il);

 public:
  template <typename Int>
  void reserve(Int size);
  void clear() noexcept;
  void zeros() noexcept { row_map_.clear(); }
  std::size_t size() const noexcept { return row_map_.size(); }
  bool empty() const noexcept { return row_map_.empty(); }
  void upsert(const SparseVectorParam& other);
  template <class Func>
  void upsert_if(const SparseVectorParam& other, Func&& func);
  void merge(const SparseVectorParam& other);
  void merge(SparseVectorParam&& other);
  template <class Func>
  void merge_if(const SparseVectorParam& other, Func&& func);
  template <class Func>
  void merge_if(SparseVectorParam&& other, Func&& func);
  void assign(int_t row, float_t row_value) { row_map_[row] = row_value; }
  template <class Func>
  void remove_if(Func&& func);
  void remove_zeros();

 public:
  template <class RandomEngine>
  inline float_t& get_scalar(RandomEngine&& engine, int_t row);
  inline float_t& get_scalar_no_init(int_t row);
  inline float_t get_scalar_no_init(int_t row) const noexcept;

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
  bool operator==(const SparseVectorParam& right) const noexcept;
  bool operator!=(const SparseVectorParam& right) const noexcept {
    return !(operator==(right));
  }
};

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseVectorParam<T, I>& svp);

/************************************************************************/
/* SparseVectorParam */
/************************************************************************/
template <typename T, typename I>
OutputStream& operator<<(OutputStream& os, const SparseVectorParam<T, I>& svp) {
  os << svp.row_map_ << svp.initializer_type_ << svp.initializer_param1_
     << svp.initializer_param2_;
  return os;
}

template <typename T, typename I>
InputStream& operator>>(InputStream& is, SparseVectorParam<T, I>& svp) {
  is >> svp.row_map_ >> svp.initializer_type_ >> svp.initializer_param1_ >>
      svp.initializer_param2_;
  return is;
}

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseVectorParam<T, I>& svp) {
  os << "(0,1)" << std::endl;
  for (const auto& entry : svp) {
    os << "row " << entry.first << ":" << entry.second << std::endl;
  }
  return os;
}

template <typename T, typename I>
void SparseVectorParam<T, I>::set_initializer(int initializer_type,
                                              float_t initializer_param1,
                                              float_t initializer_param2) {
  if (initializer_type != TENSOR_INITIALIZER_TYPE_NONE &&
      initializer_type != TENSOR_INITIALIZER_TYPE_ZEROS &&
      initializer_type != TENSOR_INITIALIZER_TYPE_ONES &&
      initializer_type != TENSOR_INITIALIZER_TYPE_CONSTANT &&
      initializer_type != TENSOR_INITIALIZER_TYPE_RAND &&
      initializer_type != TENSOR_INITIALIZER_TYPE_RANDN) {
    DXTHROW_INVALID_ARGUMENT("Invalid initializer_type: %d.", initializer_type);
  }

  if (initializer_type == TENSOR_INITIALIZER_TYPE_RAND) {
    if (initializer_param1 > initializer_param2) {
      DXTHROW_INVALID_ARGUMENT(
          "Invalid initializer_param1 and initializer_param2: %f, %f.",
          initializer_param1, initializer_param2);
    }
  }

  initializer_type_ = initializer_type;
  initializer_param1_ = initializer_param1;
  initializer_param2_ = initializer_param2;
}

template <typename T, typename I>
SparseVectorParam<T, I>::SparseVectorParam(
    std::initializer_list<std::pair<int_t, float_t>> il) {
  reserve(il.size());
  for (const auto& entry : il) {
    assign(entry.first, entry.second);
  }
}

template <typename T, typename I>
template <typename Int>
void SparseVectorParam<T, I>::reserve(Int size) {
  row_map_.reserve((std::size_t)size);
}

template <typename T, typename I>
void SparseVectorParam<T, I>::clear() noexcept {
  row_map_.clear();
  initializer_type_ = TENSOR_INITIALIZER_TYPE_NONE;
  initializer_param1_ = 0;
  initializer_param2_ = 0;
}

template <typename T, typename I>
void SparseVectorParam<T, I>::upsert(const SparseVectorParam& other) {
  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    assign(entry.first, entry.second);
  }
}

template <typename T, typename I>
template <class Func>
void SparseVectorParam<T, I>::upsert_if(const SparseVectorParam& other,
                                        Func&& func) {
  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    if (func(entry)) {
      assign(entry.first, entry.second);
    }
  }
}

template <typename T, typename I>
void SparseVectorParam<T, I>::merge(const SparseVectorParam& other) {
  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    row_map_.emplace(entry);
  }
}

template <typename T, typename I>
void SparseVectorParam<T, I>::merge(SparseVectorParam&& other) {
  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    row_map_.emplace(entry);
  }
  other.zeros();
}

template <typename T, typename I>
template <class Func>
void SparseVectorParam<T, I>::merge_if(const SparseVectorParam& other,
                                       Func&& func) {
  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    if (func(entry)) {
      row_map_.emplace(entry);
    }
  }
}

template <typename T, typename I>
template <class Func>
void SparseVectorParam<T, I>::merge_if(SparseVectorParam&& other, Func&& func) {
  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    if (func(entry)) {
      row_map_.emplace(entry);
    }
  }
  other.zeros();
}

template <typename T, typename I>
template <class Func>
void SparseVectorParam<T, I>::remove_if(Func&& func) {
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
void SparseVectorParam<T, I>::remove_zeros() {
  remove_if([](const value_type& entry) { return entry.second == 0; });
}

template <typename T, typename I>
template <class RandomEngine>
inline auto SparseVectorParam<T, I>::get_scalar(RandomEngine&& engine,
                                                int_t row) -> float_t& {
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return it->second;
  }

  auto& value = row_map_[row];
  switch (initializer_type_) {
    case TENSOR_INITIALIZER_TYPE_ONES: {
      value = 1;
    } break;
    case TENSOR_INITIALIZER_TYPE_CONSTANT: {
      value = initializer_param1_;
    } break;
    case TENSOR_INITIALIZER_TYPE_RAND: {
      std::uniform_real_distribution<float_t> dist(initializer_param1_,
                                                   initializer_param2_);
      value = dist(engine);
    } break;
    case TENSOR_INITIALIZER_TYPE_RANDN: {
      std::normal_distribution<float_t> dist(initializer_param1_,
                                             initializer_param2_);
      value = dist(engine);
    } break;
  }
  return value;
}

template <typename T, typename I>
inline auto SparseVectorParam<T, I>::get_scalar_no_init(int_t row) -> float_t& {
  return row_map_[row];
}

template <typename T, typename I>
inline auto SparseVectorParam<T, I>::get_scalar_no_init(int_t row) const
    noexcept -> float_t {
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return it->second;
  }
  return 0;
}

template <typename T, typename I>
bool SparseVectorParam<T, I>::operator==(const SparseVectorParam& right) const
    noexcept {
  if (initializer_type_ != right.initializer_type_ ||
      initializer_param1_ != right.initializer_param1_ ||
      initializer_param2_ != right.initializer_param2_) {
    return false;
  }

  if (size() != right.size()) {
    return false;
  }

  return row_map_ == right.row_map_;
}

}  // namespace deepx_core
