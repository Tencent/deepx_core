// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <algorithm>  // std::is_sorted
#include <initializer_list>
#include <iostream>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* CSRMatrix */
/************************************************************************/
template <typename T, typename I>
class CSRMatrix {
 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;

 private:
  int row_ = 0;
  std::vector<int> row_offset_ = {0};
  std::vector<int_t> col_;
  std::vector<float_t> value_;

  template <typename T2, typename I2>
  friend OutputStream& operator<<(OutputStream& os,
                                  const CSRMatrix<T2, I2>& csr);
  template <typename T2, typename I2>
  friend InputStream& operator>>(InputStream& is, CSRMatrix<T2, I2>& csr);

 public:
  int row() const noexcept { return row_; }

  size_t row_offset_size() const noexcept { return row_offset_.size(); }
  template <typename Int>
  int row_offset(Int i) const noexcept {
    return row_offset_[(size_t)i];
  }
  const int* row_offset_begin() const noexcept { return row_offset_.data(); }
  const int* row_offset_end() const noexcept {
    return row_offset_.data() + row_offset_.size();
  }

  size_t col_size() const noexcept { return col_.size(); }
  template <typename Int>
  int_t col(Int i) const noexcept {
    return col_[(size_t)i];
  }
  const int_t* col_begin() const noexcept { return col_.data(); }
  const int_t* col_end() const noexcept { return col_.data() + col_.size(); }

  size_t value_size() const noexcept { return value_.size(); }
  template <typename Int>
  float_t value(Int i) const noexcept {
    return value_[(size_t)i];
  }
  cptr_t value_begin() const noexcept { return value_.data(); }
  cptr_t value_end() const noexcept { return value_.data() + value_.size(); }

 public:
  CSRMatrix() = default;
  CSRMatrix(std::initializer_list<int> row_offset,
            std::initializer_list<int_t> col,
            std::initializer_list<float_t> value);

  bool empty() const noexcept { return value_.empty(); }
  void clear() noexcept;
  template <typename Int>
  void reserve(Int size);
  inline void add_row();
  inline void emplace(int_t col, float_t value);
  inline void trim(size_t value_size);

 public:
  // comparison
  bool operator==(const CSRMatrix& right) const noexcept;
  bool operator!=(const CSRMatrix& right) const noexcept {
    return !(operator==(right));
  }
};

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const CSRMatrix<T, I>& csr);

// macros for iterator
#define CSR_FOR_EACH_ROW(csr, i) for (int i = 0; i < (csr).row(); ++i)
#define CSR_FOR_EACH_COL(csr, i) \
  for (int __k = (csr).row_offset(i); __k < (csr).row_offset(i + 1); ++__k)
#define CSR_COL(csr) ((csr).col(__k))
#define CSR_VALUE(csr) ((csr).value(__k))

/************************************************************************/
/* CSRMatrix */
/************************************************************************/
template <typename T, typename I>
OutputStream& operator<<(OutputStream& os, const CSRMatrix<T, I>& csr) {
  os << csr.row_ << csr.row_offset_ << csr.col_ << csr.value_;
  return os;
}

template <typename T, typename I>
InputStream& operator>>(InputStream& is, CSRMatrix<T, I>& csr) {
  is >> csr.row_ >> csr.row_offset_ >> csr.col_ >> csr.value_;
  return is;
}

template <typename T, typename I>
std::ostream& operator<<(std::ostream& os, const CSRMatrix<T, I>& csr) {
  CSR_FOR_EACH_ROW(csr, i) {
    os << "row " << i << ":";
    CSR_FOR_EACH_COL(csr, i) {
      os << " " << CSR_COL(csr) << "=" << CSR_VALUE(csr);
    }
    os << std::endl;
  }
  return os;
}

template <typename T, typename I>
CSRMatrix<T, I>::CSRMatrix(std::initializer_list<int> row_offset,
                           std::initializer_list<int_t> col,
                           std::initializer_list<float_t> value) {
  if (row_offset.size() <= 1) {
    DXTHROW_INVALID_ARGUMENT("Invalid row_offset.");
  }

  if (col.size() == 0) {
    DXTHROW_INVALID_ARGUMENT("Invalid col.");
  }

  if (col.size() != value.size()) {
    DXTHROW_INVALID_ARGUMENT("Invalid col and value.");
  }

  if (!std::is_sorted(row_offset.begin(), row_offset.end())) {
    DXTHROW_INVALID_ARGUMENT("Invalid row_offset.");
  }

  if (*row_offset.begin() != 0) {
    DXTHROW_INVALID_ARGUMENT("Invalid row_offset.");
  }

  if (*(row_offset.end() - 1) != (int)col.size()) {
    DXTHROW_INVALID_ARGUMENT("Invalid row_offset and col.");
  }

  row_ = (int)row_offset.size() - 1;
  row_offset_.assign(row_offset.begin(), row_offset.end());
  col_.assign(col.begin(), col.end());
  value_.assign(value.begin(), value.end());
}

template <typename T, typename I>
void CSRMatrix<T, I>::clear() noexcept {
  row_ = 0;
  row_offset_ = {0};
  col_.clear();
  value_.clear();
}

template <typename T, typename I>
template <typename Int>
void CSRMatrix<T, I>::reserve(Int size) {
  row_offset_.reserve((size_t)size + 1);
  col_.reserve((size_t)size * 512);    // magic number
  value_.reserve((size_t)size * 512);  // magic number
}

template <typename T, typename I>
inline void CSRMatrix<T, I>::add_row() {
  ++row_;
  row_offset_.emplace_back((int)col_.size());
}

template <typename T, typename I>
inline void CSRMatrix<T, I>::emplace(int_t col, float_t value) {
  col_.emplace_back(col);
  value_.emplace_back(value);
}

template <typename T, typename I>
inline void CSRMatrix<T, I>::trim(size_t value_size) {
  col_.erase(col_.begin() + value_size, col_.end());
  value_.erase(value_.begin() + value_size, value_.end());
}

template <typename T, typename I>
bool CSRMatrix<T, I>::operator==(const CSRMatrix& right) const noexcept {
  return row_ == right.row_ && row_offset_ == right.row_offset_ &&
         col_ == right.col_ && value_ == right.value_;
}

}  // namespace deepx_core
