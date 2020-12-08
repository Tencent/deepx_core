// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/hash.h>
#include <deepx_core/common/hash_map.h>
#include <deepx_core/common/hash_map_io.h>
#include <deepx_core/common/read_write_lock.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/common/vector.h>
#include <deepx_core/common/vector_io.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/tensor/shape.h>
#include <deepx_core/tensor/tensor_type.h>
#include <cstring>  // memcpy
#include <initializer_list>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace deepx_core {

template <typename T, typename I>
class SparseRowMatrix;
template <typename T, typename I>
class SRMIterator;
template <typename T, typename I>
class SRMConstIterator;

/************************************************************************/
/* SRMIterator */
/************************************************************************/
template <typename T, typename I>
class SRMIterator {
 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using srm_t = SparseRowMatrix<float_t, int_t>;
  using map_t = HashMap<int_t, Vector<float_t>, MurmurHash<int_t>>;
  using raw_iterator_t = typename map_t::iterator;
  using value_type = std::pair<int_t, ptr_t>;
  using _iterator = SRMIterator;
  using _const_iterator = SRMConstIterator<float_t, int_t>;
  friend _const_iterator;

 private:
  srm_t* srm_ = nullptr;
  raw_iterator_t it_;
  value_type value_;

 private:
  inline void set_value() noexcept;

 public:
  SRMIterator() = default;
  inline SRMIterator(srm_t* srm, raw_iterator_t&& it) noexcept;
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
/* SRMConstIterator */
/************************************************************************/
template <typename T, typename I>
class SRMConstIterator {
 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using srm_t = SparseRowMatrix<float_t, int_t>;
  using map_t = HashMap<int_t, Vector<float_t>, MurmurHash<int_t>>;
  using raw_iterator_t = typename map_t::iterator;
  using raw_const_iterator_t = typename map_t::const_iterator;
  using value_type = std::pair<int_t, cptr_t>;
  using _iterator = SRMIterator<float_t, int_t>;
  using _const_iterator = SRMConstIterator;
  friend _iterator;

 private:
  const srm_t* srm_ = nullptr;
  raw_const_iterator_t it_;
  value_type value_;

 private:
  inline void set_value() noexcept;

 public:
  SRMConstIterator() = default;
  inline SRMConstIterator(const srm_t* srm, raw_iterator_t&& it) noexcept;
  inline SRMConstIterator(const srm_t* srm, raw_const_iterator_t&& it) noexcept;
  inline SRMConstIterator(const _iterator& it) noexcept;  // NOLINT
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
/* SparseRowMatrix */
/************************************************************************/
template <typename T, typename I>
class SparseRowMatrix {
 private:
  using map_t = HashMap<I, Vector<T>, MurmurHash<I>>;

 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using key_type = typename map_t::key_type;
  using mapped_type = typename map_t::mapped_type;
  using value_type = typename map_t::value_type;

 private:
  Shape shape_{0, 0};
  map_t row_map_;
  int initializer_type_ = TENSOR_INITIALIZER_TYPE_NONE;
  float_t initializer_param1_ = 0;
  float_t initializer_param2_ = 0;

  template <typename T2, typename I2>
  friend OutputStream& operator<<(OutputStream& os,
                                  const SparseRowMatrix<T2, I2>& srm);
  template <typename T2, typename I2>
  friend InputStream& operator>>(InputStream& is, SparseRowMatrix<T2, I2>& srm);
  template <typename T2, typename I2>
  friend InputStringStream& ReadView(InputStringStream& is,          // NOLINT
                                     SparseRowMatrix<T2, I2>& srm);  // NOLINT

  // backward compatibility
  template <typename T2, typename I2>
  friend InputStream& ReadSRP(InputStream& is,                // NOLINT
                              SparseRowMatrix<T2, I2>& srm);  // NOLINT

  // backward compatibility
  template <typename T2, typename I2>
  friend InputStringStream& ReadSRPView(
      InputStringStream& is,          // NOLINT
      SparseRowMatrix<T2, I2>& srm);  // NOLINT

  // backward compatibility
  template <typename T2, typename I2>
  friend InputStream& ReadSVP(InputStream& is,                // NOLINT
                              SparseRowMatrix<T2, I2>& srm);  // NOLINT

  // backward compatibility
  template <typename T2, typename I2>
  friend InputStringStream& ReadSVPView(
      InputStringStream& is,          // NOLINT
      SparseRowMatrix<T2, I2>& srm);  // NOLINT

 public:
  const Shape& shape() const noexcept { return shape_; }
  void set_col(int col) noexcept { shape_.resize(0, col); }
  int col() const noexcept { return shape_[1]; }

 public:
  void set_initializer(int initializer_type, float_t initializer_param1 = 0,
                       float_t initializer_param2 = 0);

 public:
  SparseRowMatrix() = default;
  SparseRowMatrix(
      std::initializer_list<int_t> rows,
      std::initializer_list<std::initializer_list<float_t>> row_values);

 public:
  template <typename Int>
  void reserve(Int size);
  void clear() noexcept;
  void zeros() noexcept { row_map_.clear(); }
  size_t size() const noexcept { return row_map_.size(); }
  bool empty() const noexcept { return row_map_.empty(); }
  void upsert(const SparseRowMatrix& other);
  template <class Func>
  void upsert_if(const SparseRowMatrix& other, Func&& func);
  void merge(const SparseRowMatrix& other);
  void merge(SparseRowMatrix&& other);
  template <class Func>
  void merge_if(const SparseRowMatrix& other, Func&& func);
  template <class Func>
  void merge_if(SparseRowMatrix&& other, Func&& func);
  void assign(int_t row, cptr_t row_value);
  void assign_view(int_t row, cptr_t row_value);
  template <class Func>
  void remove_if(Func&& func);
  void remove_zeros();

  void upsert(const SparseRowMatrix& other, ReadWriteLock* lock);
  template <class Func>
  void upsert_if(const SparseRowMatrix& other, Func&& func,
                 ReadWriteLock* lock);
  void assign(int_t row, cptr_t row_value, ReadWriteLock* lock);

 public:
  template <class RandomEngine>
  inline ptr_t get_row(RandomEngine&& engine, int_t row);
  inline ptr_t get_row_no_init(int_t row);
  inline cptr_t get_row_no_init(int_t row) const noexcept;
  template <class RandomEngine>
  inline float_t& get_scalar(RandomEngine&& engine, int_t row);
  inline float_t& get_scalar_no_init(int_t row);
  inline float_t get_scalar_no_init(int_t row) const noexcept;

  template <class RandomEngine>
  inline ptr_t get_row(RandomEngine&& engine, int_t row, ReadWriteLock* lock);
  inline ptr_t get_row_no_init(int_t row, ReadWriteLock* lock);
  inline cptr_t get_row_no_init(int_t row, ReadWriteLock* lock) const;
  template <class RandomEngine>
  inline float_t& get_scalar(RandomEngine&& engine, int_t row,
                             ReadWriteLock* lock);
  inline float_t& get_scalar_no_init(int_t row, ReadWriteLock* lock);
  inline float_t get_scalar_no_init(int_t row, ReadWriteLock* lock) const;

 public:
  // iterator
  using iterator = SRMIterator<float_t, int_t>;
  using const_iterator = SRMConstIterator<float_t, int_t>;
  friend iterator;
  friend const_iterator;
  iterator find(int_t row) noexcept {
    return iterator(this, row_map_.find(row));
  }
  const_iterator find(int_t row) const noexcept {
    return const_iterator(this, row_map_.find(row));
  }
  iterator begin() noexcept { return iterator(this, row_map_.begin()); }
  const_iterator begin() const noexcept {
    return const_iterator(this, row_map_.begin());
  }
  const_iterator cbegin() const noexcept {
    return const_iterator(this, row_map_.cbegin());
  }
  iterator end() noexcept { return iterator(this, row_map_.end()); }
  const_iterator end() const noexcept {
    return const_iterator(this, row_map_.end());
  }
  const_iterator cend() const noexcept {
    return const_iterator(this, row_map_.cend());
  }

 public:
  // comparison
  bool operator==(const SparseRowMatrix& right) const noexcept;
  bool operator!=(const SparseRowMatrix& right) const noexcept {
    return !(operator==(right));
  }
};

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseRowMatrix<T, I>& srm);

/************************************************************************/
/* SparseRowMatrix */
/************************************************************************/
template <typename T, typename I>
OutputStream& operator<<(OutputStream& os, const SparseRowMatrix<T, I>& srm) {
  int version = 0x0a0c72e7;  // magic number version
  os << version;
  os << srm.col() << srm.row_map_ << srm.initializer_type_
     << srm.initializer_param1_ << srm.initializer_param2_;
  return os;
}

template <typename T, typename I>
InputStream& operator>>(InputStream& is, SparseRowMatrix<T, I>& srm) {
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  if (version == 0x0a0c72e7) {  // magic number version
    int col;
    is >> version;
    is >> col >> srm.row_map_ >> srm.initializer_type_ >>
        srm.initializer_param1_ >> srm.initializer_param2_;
    if (is) {
      srm.set_col(col);
    }
  } else {
    // backward compatibility
    srm.clear();
    std::vector<T> value;
    HashMap<I, size_t, MurmurHash<I>> row_offset_map;
    is >> srm.shape_ >> value >> row_offset_map >> srm.initializer_type_ >>
        srm.initializer_param1_ >> srm.initializer_param2_;
    if (is) {
      for (const auto& entry : row_offset_map) {
        srm.assign(entry.first, &value[entry.second]);
      }
    }
  }
  return is;
}

template <typename T, typename I>
InputStringStream& ReadView(InputStringStream& is,         // NOLINT
                            SparseRowMatrix<T, I>& srm) {  // NOLINT
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  if (version == 0x0a0c72e7) {  // magic number version
    int col;
    ReadView(is, version);
    ReadView(is, col);
    ReadView(is, srm.row_map_);
    ReadView(is, srm.initializer_type_);
    ReadView(is, srm.initializer_param1_);
    ReadView(is, srm.initializer_param2_);
    if (is) {
      srm.set_col(col);
    }
  } else {
    // backward compatibility
    is.set_bad();
  }
  return is;
}

template <typename T, typename I>
InputStream& ReadSRP(InputStream& is,               // NOLINT
                     SparseRowMatrix<T, I>& srm) {  // NOLINT
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  if (version == 0x0a0c72e7) {  // magic number version
    int col;
    is >> version;
    is >> col >> srm.row_map_ >> srm.initializer_type_ >>
        srm.initializer_param1_ >> srm.initializer_param2_;
    if (is) {
      srm.set_col(col);
    }
  } else {
    // backward compatibility
    int col;
    is >> col >> srm.row_map_ >> srm.initializer_type_ >>
        srm.initializer_param1_ >> srm.initializer_param2_;
    if (is) {
      srm.set_col(col);
    }
  }
  return is;
}

template <typename T, typename I>
InputStringStream& ReadSRPView(InputStringStream& is,         // NOLINT
                               SparseRowMatrix<T, I>& srm) {  // NOLINT
  return ReadView(is, srm);
}

template <typename T, typename I>
InputStream& ReadSVP(InputStream& is,               // NOLINT
                     SparseRowMatrix<T, I>& srm) {  // NOLINT
  srm.clear();
  HashMap<I, T, MurmurHash<I>> row_map;
  is >> row_map >> srm.initializer_type_ >> srm.initializer_param1_ >>
      srm.initializer_param2_;
  if (is) {
    srm.set_col(1);
    for (const auto& entry : row_map) {
      srm.assign(entry.first, &entry.second);
    }
  }
  return is;
}

template <typename T, typename I>
InputStringStream& ReadSVPView(InputStringStream& is,         // NOLINT
                               SparseRowMatrix<T, I>& srm) {  // NOLINT
  ReadSVP(is, srm);
  return is;
}

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const SparseRowMatrix<T, I>& srm) {
  os << srm.shape() << std::endl;
  for (const auto& entry : srm) {
    os << "row " << entry.first << ":";
    for (int i = 0; i < srm.col(); ++i) {
      os << " " << entry.second[i];
    }
    os << std::endl;
  }
  return os;
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::set_initializer(int initializer_type,
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
SparseRowMatrix<T, I>::SparseRowMatrix(
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
void SparseRowMatrix<T, I>::reserve(Int size) {
  row_map_.reserve((size_t)size);
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::clear() noexcept {
  shape_.resize(0, 0);
  row_map_.clear();
  initializer_type_ = TENSOR_INITIALIZER_TYPE_NONE;
  initializer_param1_ = 0;
  initializer_param2_ = 0;
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::upsert(const SparseRowMatrix& other) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    assign(entry.first, entry.second.data());
  }
}

template <typename T, typename I>
template <class Func>
void SparseRowMatrix<T, I>::upsert_if(const SparseRowMatrix& other,
                                      Func&& func) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    if (func(entry)) {
      assign(entry.first, entry.second.data());
    }
  }
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::merge(const SparseRowMatrix& other) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    row_map_.emplace(entry);
  }
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::merge(SparseRowMatrix&& other) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (auto& entry : other.row_map_) {
    row_map_.emplace(entry.first, std::move(entry.second));
  }
  other.zeros();
}

template <typename T, typename I>
template <class Func>
void SparseRowMatrix<T, I>::merge_if(const SparseRowMatrix& other,
                                     Func&& func) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (const auto& entry : other.row_map_) {
    if (func(entry)) {
      row_map_.emplace(entry);
    }
  }
}

template <typename T, typename I>
template <class Func>
void SparseRowMatrix<T, I>::merge_if(SparseRowMatrix&& other, Func&& func) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  row_map_.reserve(row_map_.size() + other.row_map_.size());
  for (auto& entry : other.row_map_) {
    if (func(entry)) {
      row_map_.emplace(entry.first, std::move(entry.second));
    }
  }
  other.zeros();
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::assign(int_t row, cptr_t row_value) {
  auto& value = row_map_[row];
  value.resize(col());
  memcpy(&value[0], row_value, col() * sizeof(float_t));
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::assign_view(int_t row, cptr_t row_value) {
  auto& value = row_map_[row];
  value.view(row_value, col());
}

template <typename T, typename I>
template <class Func>
void SparseRowMatrix<T, I>::remove_if(Func&& func) {
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
void SparseRowMatrix<T, I>::remove_zeros() {
  remove_if([this](const value_type& entry) {
    for (int i = 0; i < col(); ++i) {
      if (entry.second[i] != 0) {
        return false;
      }
    }
    return true;
  });
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::upsert(const SparseRowMatrix& other,
                                   ReadWriteLock* lock) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  {
    WriteLockGuard guard(lock);
    row_map_.reserve(row_map_.size() + other.row_map_.size());
  }
  for (const auto& entry : other.row_map_) {
    assign(entry.first, entry.second.data(), lock);
  }
}

template <typename T, typename I>
template <class Func>
void SparseRowMatrix<T, I>::upsert_if(const SparseRowMatrix& other, Func&& func,
                                      ReadWriteLock* lock) {
  if (col() != other.col()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", col(), other.col());
  }

  {
    WriteLockGuard guard(lock);
    row_map_.reserve(row_map_.size() + other.row_map_.size());
  }
  for (const auto& entry : other.row_map_) {
    if (func(entry)) {
      assign(entry.first, entry.second.data(), lock);
    }
  }
}

template <typename T, typename I>
void SparseRowMatrix<T, I>::assign(int_t row, cptr_t row_value,
                                   ReadWriteLock* lock) {
  WriteLockGuard guard(lock);
  ptr_t value = get_row_no_init(row);
  memcpy(value, row_value, col() * sizeof(float_t));
}

template <typename T, typename I>
template <class RandomEngine>
inline auto SparseRowMatrix<T, I>::get_row(RandomEngine&& engine, int_t row)
    -> ptr_t {
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return &it->second[0];
  }

  auto& value = row_map_[row];
  value.resize(col());
  switch (initializer_type_) {
    case TENSOR_INITIALIZER_TYPE_ONES: {
      for (int i = 0; i < col(); ++i) {
        value[i] = 1;
      }
    } break;
    case TENSOR_INITIALIZER_TYPE_CONSTANT: {
      for (int i = 0; i < col(); ++i) {
        value[i] = initializer_param1_;
      }
    } break;
    case TENSOR_INITIALIZER_TYPE_RAND: {
      std::uniform_real_distribution<float_t> dist(initializer_param1_,
                                                   initializer_param2_);
      for (int i = 0; i < col(); ++i) {
        value[i] = dist(engine);
      }
    } break;
    case TENSOR_INITIALIZER_TYPE_RANDN: {
      std::normal_distribution<float_t> dist(initializer_param1_,
                                             initializer_param2_);
      for (int i = 0; i < col(); ++i) {
        value[i] = dist(engine);
      }
    } break;
  }
  return &value[0];
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_row_no_init(int_t row) -> ptr_t {
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return &it->second[0];
  }

  auto& value = row_map_[row];
  value.resize(col());
  return &value[0];
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_row_no_init(int_t row) const noexcept
    -> cptr_t {
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return &it->second[0];
  }
  return nullptr;
}

template <typename T, typename I>
template <class RandomEngine>
inline auto SparseRowMatrix<T, I>::get_scalar(RandomEngine&& engine, int_t row)
    -> float_t& {
  DXASSERT(col() == 1);
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return it->second[0];
  }

  auto& value = row_map_[row];
  value.resize(1);
  switch (initializer_type_) {
    case TENSOR_INITIALIZER_TYPE_ONES: {
      value[0] = 1;
    } break;
    case TENSOR_INITIALIZER_TYPE_CONSTANT: {
      value[0] = initializer_param1_;
    } break;
    case TENSOR_INITIALIZER_TYPE_RAND: {
      std::uniform_real_distribution<float_t> dist(initializer_param1_,
                                                   initializer_param2_);
      value[0] = dist(engine);
    } break;
    case TENSOR_INITIALIZER_TYPE_RANDN: {
      std::normal_distribution<float_t> dist(initializer_param1_,
                                             initializer_param2_);
      value[0] = dist(engine);
    } break;
  }
  return value[0];
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_scalar_no_init(int_t row) -> float_t& {
  DXASSERT(col() == 1);
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return it->second[0];
  }

  auto& value = row_map_[row];
  value.resize(1);
  return value[0];
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_scalar_no_init(int_t row) const noexcept
    -> float_t {
  DXASSERT(col() == 1);
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return it->second[0];
  }
  return 0;
}

template <typename T, typename I>
template <class RandomEngine>
inline auto SparseRowMatrix<T, I>::get_row(RandomEngine&& engine, int_t row,
                                           ReadWriteLock* lock) -> ptr_t {
  {
    ReadLockGuard guard(lock);
    auto it = row_map_.find(row);
    if (it != row_map_.end()) {
      return &it->second[0];
    }
  }

  {
    WriteLockGuard guard(lock);
    auto& value = row_map_[row];
    value.resize(col());
    switch (initializer_type_) {
      case TENSOR_INITIALIZER_TYPE_ONES: {
        for (int i = 0; i < col(); ++i) {
          value[i] = 1;
        }
      } break;
      case TENSOR_INITIALIZER_TYPE_CONSTANT: {
        for (int i = 0; i < col(); ++i) {
          value[i] = initializer_param1_;
        }
      } break;
      case TENSOR_INITIALIZER_TYPE_RAND: {
        std::uniform_real_distribution<float_t> dist(initializer_param1_,
                                                     initializer_param2_);
        for (int i = 0; i < col(); ++i) {
          value[i] = dist(engine);
        }
      } break;
      case TENSOR_INITIALIZER_TYPE_RANDN: {
        std::normal_distribution<float_t> dist(initializer_param1_,
                                               initializer_param2_);
        for (int i = 0; i < col(); ++i) {
          value[i] = dist(engine);
        }
      } break;
    }
    return &value[0];
  }
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_row_no_init(int_t row,
                                                   ReadWriteLock* lock)
    -> ptr_t {
  {
    ReadLockGuard guard(lock);
    auto it = row_map_.find(row);
    if (it != row_map_.end()) {
      return &it->second[0];
    }
  }

  {
    WriteLockGuard guard(lock);
    auto& value = row_map_[row];
    value.resize(col());
    return &value[0];
  }
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_row_no_init(int_t row,
                                                   ReadWriteLock* lock) const
    -> cptr_t {
  ReadLockGuard guard(lock);
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return &it->second[0];
  }
  return nullptr;
}

template <typename T, typename I>
template <class RandomEngine>
inline auto SparseRowMatrix<T, I>::get_scalar(RandomEngine&& engine, int_t row,
                                              ReadWriteLock* lock) -> float_t& {
  DXASSERT(col() == 1);
  {
    ReadLockGuard guard(lock);
    auto it = row_map_.find(row);
    if (it != row_map_.end()) {
      return it->second[0];
    }
  }

  {
    WriteLockGuard guard(lock);
    auto& value = row_map_[row];
    value.resize(1);
    switch (initializer_type_) {
      case TENSOR_INITIALIZER_TYPE_ONES: {
        value[0] = 1;
      } break;
      case TENSOR_INITIALIZER_TYPE_CONSTANT: {
        value[0] = initializer_param1_;
      } break;
      case TENSOR_INITIALIZER_TYPE_RAND: {
        std::uniform_real_distribution<float_t> dist(initializer_param1_,
                                                     initializer_param2_);
        value[0] = dist(engine);
      } break;
      case TENSOR_INITIALIZER_TYPE_RANDN: {
        std::normal_distribution<float_t> dist(initializer_param1_,
                                               initializer_param2_);
        value[0] = dist(engine);
      } break;
    }
    return value[0];
  }
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_scalar_no_init(int_t row,
                                                      ReadWriteLock* lock)
    -> float_t& {
  DXASSERT(col() == 1);
  {
    ReadLockGuard guard(lock);
    auto it = row_map_.find(row);
    if (it != row_map_.end()) {
      return it->second[0];
    }
  }

  {
    WriteLockGuard guard(lock);
    auto& value = row_map_[row];
    value.resize(1);
    return value[0];
  }
}

template <typename T, typename I>
inline auto SparseRowMatrix<T, I>::get_scalar_no_init(int_t row,
                                                      ReadWriteLock* lock) const
    -> float_t {
  DXASSERT(col() == 1);
  ReadLockGuard guard(lock);
  auto it = row_map_.find(row);
  if (it != row_map_.end()) {
    return it->second[0];
  }
  return 0;
}

template <typename T, typename I>
bool SparseRowMatrix<T, I>::operator==(const SparseRowMatrix& right) const
    noexcept {
  if (shape_ != right.shape_) {
    return false;
  }

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

/************************************************************************/
/* SRMIterator */
/************************************************************************/
template <typename T, typename I>
inline void SRMIterator<T, I>::set_value() noexcept {
  if (it_ != srm_->row_map_.end()) {
    value_.first = it_->first;
    value_.second = &it_->second[0];
  } else {
    // ZERO is only to eliminate clang-tidy warnings.
    static float_t ZERO = 0;
    value_.first = 0;
    value_.second = &ZERO;
  }
}

template <typename T, typename I>
inline SRMIterator<T, I>::SRMIterator(srm_t* srm, raw_iterator_t&& it) noexcept
    : srm_(srm), it_(it) {
  set_value();
}

template <typename T, typename I>
inline bool SRMIterator<T, I>::operator==(const _iterator& right) const
    noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRMIterator<T, I>::operator!=(const _iterator& right) const
    noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline bool SRMIterator<T, I>::operator==(const _const_iterator& right) const
    noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRMIterator<T, I>::operator!=(const _const_iterator& right) const
    noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline auto SRMIterator<T, I>::operator++() noexcept -> _iterator& {
  ++it_;
  set_value();
  return *this;
}

template <typename T, typename I>
inline auto SRMIterator<T, I>::operator++(int) noexcept -> _iterator {
  _iterator origin = *this;
  operator++();
  return origin;
}

/************************************************************************/
/* SRMConstIterator */
/************************************************************************/
template <typename T, typename I>
inline void SRMConstIterator<T, I>::set_value() noexcept {
  if (it_ != srm_->row_map_.end()) {
    value_.first = it_->first;
    value_.second = &it_->second[0];
  } else {
    // ZERO is only to eliminate clang-tidy warnings.
    static float_t ZERO = 0;
    value_.first = 0;
    value_.second = &ZERO;
  }
}

template <typename T, typename I>
inline SRMConstIterator<T, I>::SRMConstIterator(const srm_t* srm,
                                                raw_iterator_t&& it) noexcept
    : srm_(srm), it_(it) {
  set_value();
}

template <typename T, typename I>
inline SRMConstIterator<T, I>::SRMConstIterator(
    const srm_t* srm, raw_const_iterator_t&& it) noexcept
    : srm_(srm), it_(it) {
  set_value();
}

template <typename T, typename I>
inline SRMConstIterator<T, I>::SRMConstIterator(const _iterator& it) noexcept
    : srm_(it.srm_), it_(it.it_) {
  set_value();
}

template <typename T, typename I>
inline bool SRMConstIterator<T, I>::operator==(const _iterator& right) const
    noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRMConstIterator<T, I>::operator!=(const _iterator& right) const
    noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline bool SRMConstIterator<T, I>::operator==(
    const _const_iterator& right) const noexcept {
  return it_ == right.it_;
}

template <typename T, typename I>
inline bool SRMConstIterator<T, I>::operator!=(
    const _const_iterator& right) const noexcept {
  return !(operator==(right));
}

template <typename T, typename I>
inline auto SRMConstIterator<T, I>::operator++() noexcept -> _const_iterator& {
  ++it_;
  set_value();
  return *this;
}

template <typename T, typename I>
inline auto SRMConstIterator<T, I>::operator++(int) noexcept
    -> _const_iterator {
  _const_iterator origin = *this;
  operator++();
  return origin;
}

}  // namespace deepx_core
