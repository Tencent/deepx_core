// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/tensor/shape.h>
#include <deepx_core/tensor/tensor_type.h>
#include <algorithm>  // std::equal, ...
#include <cmath>      // std::sqrt
#include <cstddef>    // std::nullptr_t
#include <cstring>    // memset
#include <initializer_list>
#include <iostream>
#include <iterator>  // std::distance, std::reverse_iterator
#include <random>
#include <sstream>
#include <string>
#include <type_traits>  // std::is_floating_point, ...
#include <utility>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* Tensor */
/************************************************************************/
template <typename T>
class Tensor {
 public:
  static constexpr bool IS_FLOAT = std::is_floating_point<T>::value;
  static constexpr bool IS_INT = std::is_integral<T>::value;

 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  Shape shape_;
  std::vector<value_type> storage_;
  pointer data_ = nullptr;

  template <typename U>
  friend class Tensor;

  template <typename U>
  friend OutputStream& operator<<(OutputStream& os, const Tensor<U>& tsr);
  template <typename U>
  friend InputStream& operator>>(InputStream& is, Tensor<U>& tsr);
  template <typename U>
  friend InputStringStream& ReadView(InputStringStream& is,  // NOLINT
                                     Tensor<U>& tsr);        // NOLINT

 public:
  Tensor() = default;

  Tensor(const Tensor&);
  Tensor& operator=(const Tensor&);

  Tensor(Tensor&&) noexcept;
  Tensor& operator=(Tensor&&) noexcept;

  Tensor(std::initializer_list<value_type> il);
  Tensor& operator=(std::initializer_list<value_type> il);

  Tensor(std::initializer_list<std::initializer_list<value_type>> il);
  Tensor& operator=(
      std::initializer_list<std::initializer_list<value_type>> il) noexcept;

  explicit Tensor(const std::vector<value_type>& data);
  explicit Tensor(std::vector<value_type>&& data);
  explicit Tensor(const Shape& shape) { resize(shape); }

  template <typename II>
  Tensor(II first, II last);

 public:
  template <typename II>
  void assign(II first, II last);

  // Set data without changing the shape of current tensor.
  Tensor& set_data(std::initializer_list<value_type> il);
  Tensor& set_data(const std::vector<value_type>& data);
  template <typename Int>
  Tensor& set_data(const_pointer data, Int size);
  Tensor& set_data(const Tensor& other);
  template <typename II>
  Tensor& set_data(II first, II last);

 public:
  // element access
  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }
  template <typename Int>
  reference data(Int i) noexcept {
    return data_[i];
  }
  template <typename Int>
  value_type data(Int i) const noexcept {
    return data_[i];
  }
  reference front() noexcept { return data_[0]; }
  value_type front() const noexcept { return data_[0]; }
  reference back() noexcept { return data_[total_dim() - 1]; }
  value_type back() const noexcept { return data_[total_dim() - 1]; }

 public:
  // iterator
  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }
  iterator end() noexcept { return data_ + total_dim(); }
  const_iterator end() const noexcept { return data_ + total_dim(); }
  const_iterator cend() const noexcept { return data_ + total_dim(); }
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
  // view
  // Return if current tensor is a view.
  bool is_view() const noexcept { return data_ && data_ != storage_.data(); }

  // Return a view of current tensor.
  Tensor get_view() const noexcept;

  // Return a view of a slice of current tensor.
  template <typename Int>
  Tensor operator[](Int i);

  // Return a view of a slice of current tensor.
  template <typename Int>
  const Tensor operator[](Int i) const;

  // View 'data' with 'shape'.
  Tensor& view(const Shape& shape, pointer data) noexcept;

 public:
  const Shape& shape() const noexcept { return shape_; }
  // methods from Shape
  int rank() const noexcept { return shape_.rank(); }
  int total_dim() const noexcept { return shape_.total_dim(); }
  const int* dim() const noexcept { return shape_.dim(); }
  template <typename Int>
  int dim(Int i) const noexcept {
    return shape_[i];
  }
  bool is_rank(int rank) const noexcept { return shape_.is_rank(rank); }
  bool is_scalar() const noexcept { return shape_.is_scalar(); }
  bool empty() const noexcept { return shape_.empty(); }
  Tensor& resize(int d0);
  Tensor& resize(int d0, int d1);
  Tensor& resize(int d0, int d1, int d2);
  template <typename... Args>
  Tensor& resize(Args&&... args);
  template <typename... Args>
  Tensor& reshape(Args&&... args);
  Tensor& expand_dim(int axis);
  Tensor& squeeze(int axis);
  bool same_shape(const Tensor& other) const noexcept;
  bool same_shape(Tensor& other) const noexcept;  // NOLINT
  template <typename U>
  bool same_shape(const Tensor<U>& other) const noexcept;
  template <typename U>
  bool same_shape(Tensor<U>& other) const noexcept;  // NOLINT
  template <typename... Args>
  bool same_shape(Args&&... args) const noexcept;

 public:
  template <typename Int>
  void reserve(Int size);
  void clear() noexcept;
  void swap(Tensor& other) noexcept;

 public:
  // methods similar to numpy
  // Return sum of current tensor.
  value_type sum() const noexcept;

  // Return mean of current tensor.
  value_type mean() const noexcept;

  // Return sum of absolute value of current tensor.
  value_type asum() const noexcept;

  // Return mean of absolute value of current tensor.
  value_type amean() const noexcept;

  // Return standard deviation of current tensor.
  value_type std() const noexcept;

  // Return variance of current tensor.
  value_type var() const noexcept;

  // Fill current tensor with 0, 1, 2, ...
  Tensor& arange() noexcept;

  // Fill current tensor with 'c'.
  Tensor& constant(value_type c) noexcept;

  // Fill current tensor with 0.
  Tensor& zeros() noexcept;

  // Fill current tensor with 1.
  Tensor& ones() noexcept;

 public:
  // random
  // uniform distribution [min, max)
  template <class RandomEngine>
  Tensor& rand(RandomEngine&& engine, value_type _min,
               value_type _max) noexcept;

  // uniform distribution [0, 1)
  template <class RandomEngine>
  Tensor& rand(RandomEngine&& engine) noexcept;

  // normal distribution (mean, stddev)
  template <class RandomEngine>
  Tensor& randn(RandomEngine&& engine, value_type mean,
                value_type stddev) noexcept;

  // normal distribution (0, 1)
  template <class RandomEngine>
  Tensor& randn(RandomEngine&& engine) noexcept;

 private:
  // truncated normal distribution (mean, stddev)
  template <class RandomEngine>
  Tensor& randn_truncated(RandomEngine&& engine, value_type mean,
                          value_type stddev) noexcept;

  // truncated normal distribution (0, 1)
  template <class RandomEngine>
  Tensor& randn_truncated(RandomEngine&& engine) noexcept;

  // variance scaling initializer(uniform distribution)
  // 'mode' is 1: row
  // 'mode' is 2: col
  // 'mode' is 3: (row + col) / 2
  template <class RandomEngine>
  Tensor& rand_variance_scaling(RandomEngine&& engine, value_type scale,
                                int mode);

  // variance scaling initializer(normal distribution)
  // 'mode' is 1: row
  // 'mode' is 2: col
  // 'mode' is 3: (row + col) / 2
  template <class RandomEngine>
  Tensor& randn_variance_scaling(RandomEngine&& engine, value_type scale,
                                 int mode);

 public:
  // LeCun initializer
  // Reference: LeCun, Yann A., et al. "Efficient backprop." Neural networks:
  // Tricks of the trade. Springer, Berlin, Heidelberg, 2012. 9-48.
  template <class RandomEngine>
  Tensor& rand_lecun(RandomEngine&& engine);
  template <class RandomEngine>
  Tensor& randn_lecun(RandomEngine&& engine);

  // Xavier initializer (aka: Glorot)
  // Reference: Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty
  // of training deep feedforward neural networks." Proceedings of the
  // thirteenth international conference on artificial intelligence and
  // statistics. 2010.
  template <class RandomEngine>
  Tensor& rand_xavier(RandomEngine&& engine);
  template <class RandomEngine>
  Tensor& randn_xavier(RandomEngine&& engine);

  // He initializer
  // Reference: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
  // human-level performance on imagenet classification." Proceedings of the
  // IEEE international conference on computer vision. 2015.
  template <class RandomEngine>
  Tensor& rand_he(RandomEngine&& engine);
  template <class RandomEngine>
  Tensor& randn_he(RandomEngine&& engine);

  // uniform int distribution [min, max)
  template <class RandomEngine>
  Tensor& rand_int(RandomEngine&& engine, int _min, int _max) noexcept;

  // Random initialize current tensor according to
  // 'initializer_type', 'initializer_param1' and 'initializer_param2'.
  template <class RandomEngine>
  Tensor& rand_init(RandomEngine&& engine, int initializer_type,
                    value_type initializer_param1 = 0,
                    value_type initializer_param2 = 0);
};

template <typename T>
std::string to_string(const Tensor<T>& tsr, int summary = 3);

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tsr);

// comparison
template <typename T>
bool operator==(const Tensor<T>& left, const Tensor<T>& right) noexcept {
  return left.shape() == right.shape() &&
         std::equal(left.begin(), left.end(), right.begin());
}

template <typename T>
bool operator!=(const Tensor<T>& left, const Tensor<T>& right) noexcept {
  return !(left == right);
}

template <typename T>
bool operator==(const Tensor<T>& left, const std::vector<T>& right) noexcept {
  return left.total_dim() == (int)right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

template <typename T>
bool operator!=(const Tensor<T>& left, const std::vector<T>& right) noexcept {
  return !(left == right);
}

template <typename T>
bool operator==(const std::vector<T>& left, const Tensor<T>& right) noexcept {
  return right == left;
}

template <typename T>
bool operator!=(const std::vector<T>& left, const Tensor<T>& right) noexcept {
  return !(right == left);
}

template <typename T>
bool operator==(std::nullptr_t, Tensor<T> right) noexcept {
  return right.data() == nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, Tensor<T> right) noexcept {
  return right.data() != nullptr;
}

template <typename T>
bool operator==(Tensor<T> left, std::nullptr_t) noexcept {
  return left.data() == nullptr;
}

template <typename T>
bool operator!=(Tensor<T> left, std::nullptr_t) noexcept {
  return left.data() != nullptr;
}

/************************************************************************/
/* Tensor */
/************************************************************************/
template <typename T>
OutputStream& operator<<(OutputStream& os, const Tensor<T>& tsr) {
  os << tsr.shape_;
  os.Write(tsr.data_, sizeof(T) * tsr.total_dim());
  return os;
}

template <>
inline OutputStream& operator<<(OutputStream& os,
                                const Tensor<std::string>& tsr) {
  os << tsr.shape_;
  for (int i = 0; i < tsr.total_dim(); ++i) {
    os << tsr.data_[i];
    if (!os) {
      break;
    }
  }
  return os;
}

template <typename T>
InputStream& operator>>(InputStream& is, Tensor<T>& tsr) {
  is >> tsr.shape_;
  if (!is) {
    return is;
  }

  tsr.storage_.resize(tsr.total_dim());
  tsr.data_ = tsr.storage_.data();
  is.Read(tsr.data_, sizeof(T) * tsr.total_dim());
  return is;
}

template <>
inline InputStream& operator>>(InputStream& is, Tensor<std::string>& tsr) {
  is >> tsr.shape_;
  if (!is) {
    return is;
  }

  tsr.storage_.resize(tsr.total_dim());
  tsr.data_ = tsr.storage_.data();
  for (int i = 0; i < tsr.total_dim(); ++i) {
    is >> tsr.data_[i];
  }
  return is;
}

template <typename T>
InputStringStream& ReadView(InputStringStream& is, Tensor<T>& tsr) {  // NOLINT
  ReadView(is, tsr.shape_);
  if (!is) {
    return is;
  }

  tsr.storage_.clear();
  // The cast is ugly and unsafe.
  tsr.data_ = (T*)is.GetData();
  is.Skip(sizeof(T) * tsr.total_dim());
  return is;
}

template <>
inline InputStringStream& ReadView(InputStringStream& is,       // NOLINT
                                   Tensor<std::string>& tsr) {  // NOLINT
  // no actual view
  is >> tsr;
  return is;
}

namespace detail {

template <typename T>
class TensorWriteTextHelper {
 private:
  const Tensor<T>& tsr_;
  int summary_;
  size_t max_text_length_;

 private:
  class SliceHelper {
   private:
    int d0_;
    int begin1_, end1_, begin2_, end2_;

   public:
    int d0() const noexcept { return d0_; }
    int begin1() const noexcept { return begin1_; }
    int end1() const noexcept { return end1_; }
    int begin2() const noexcept { return begin2_; }
    int end2() const noexcept { return end2_; }

   public:
    SliceHelper(const Tensor<T>& slice, int summary);
    int has_summary() const noexcept { return end1_ != d0_; }
  };

 public:
  TensorWriteTextHelper(const Tensor<T>& tsr, int summary);

 private:
  static size_t GetTextLength(const T& t);
  void InitMaxTextLength(const Tensor<T>& slice);

 public:
  void WriteText(std::ostream& os);

 private:
  void WriteText(std::ostream& os, const T& t);
  void WriteText(std::ostream& os, const Tensor<T>& slice);
};

template <typename T>
TensorWriteTextHelper<T>::SliceHelper::SliceHelper(const Tensor<T>& slice,
                                                   int summary) {
  DXASSERT(slice.rank() > 0);
  d0_ = slice.dim(0);
  begin1_ = 0;
  end2_ = d0_;
  if (summary < 0 || summary * 2 >= d0_) {
    end1_ = d0_;
    begin2_ = d0_;
  } else {
    end1_ = summary;
    begin2_ = d0_ - summary;
  }
}

template <typename T>
TensorWriteTextHelper<T>::TensorWriteTextHelper(const Tensor<T>& tsr,
                                                int summary)
    : tsr_(tsr), summary_(summary), max_text_length_(0) {
  if (tsr_.rank() > 0) {
    InitMaxTextLength(tsr_);
  }
}

template <typename T>
size_t TensorWriteTextHelper<T>::GetTextLength(const T& t) {
  std::ostringstream os;
  os << t;
  return os.str().size();
}

template <>
inline size_t TensorWriteTextHelper<std::string>::GetTextLength(
    const std::string& s) {
  return s.size() + 2;  // 2 quotes
}

template <typename T>
void TensorWriteTextHelper<T>::InitMaxTextLength(const Tensor<T>& slice) {
  SliceHelper slice_helper(slice, summary_);
  if (slice.rank() == 1) {
    auto for_range = [this, &slice](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        max_text_length_ =
            std::max(max_text_length_, GetTextLength(slice.data(i)));
      }
    };
    for_range(slice_helper.begin1(), slice_helper.end1());
    for_range(slice_helper.begin2(), slice_helper.end2());
  } else {
    auto for_range = [this, &slice](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        InitMaxTextLength(slice[i]);
      }
    };
    for_range(slice_helper.begin1(), slice_helper.end1());
    for_range(slice_helper.begin2(), slice_helper.end2());
  }
}

template <typename T>
void TensorWriteTextHelper<T>::WriteText(std::ostream& os) {
  os << tsr_.shape() << std::endl;
  if (tsr_.rank() > 0) {
    WriteText(os, tsr_);
    os << std::endl;
  }
}

template <typename T>
void TensorWriteTextHelper<T>::WriteText(std::ostream& os, const T& t) {
  std::streamsize prev_width = os.width();
  os.width((std::streamsize)max_text_length_);
  os << t;
  os.width(prev_width);
}

template <>
inline void TensorWriteTextHelper<std::string>::WriteText(
    std::ostream& os, const std::string& s) {
  std::string quoted_s = "\"" + s + "\"";
  std::streamsize prev_width = os.width();
  os.width((std::streamsize)max_text_length_);
  os << quoted_s;
  os.width(prev_width);
}

template <typename T>
void TensorWriteTextHelper<T>::WriteText(std::ostream& os,
                                         const Tensor<T>& slice) {
  SliceHelper slice_helper(slice, summary_);
  if (slice.rank() == 1) {
    os << "[";
    auto for_range = [this, &os, &slice](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        if (i != 0) {
          os << " ";
        }
        WriteText(os, slice.data(i));
      }
    };
    for_range(slice_helper.begin1(), slice_helper.end1());
    if (slice_helper.has_summary()) {
      os << " ";
      os << "...";
    }
    for_range(slice_helper.begin2(), slice_helper.end2());
    os << "]";
  } else {
    std::string spaces(tsr_.rank() - slice.rank() + 1, ' ');
    os << "[";
    auto for_range = [this, &os, &slice, &slice_helper, &spaces](int begin,
                                                                 int end) {
      for (int i = begin; i < end; ++i) {
        if (i != 0) {
          os << spaces;
        }
        WriteText(os, slice[i]);
        if (i != slice_helper.d0() - 1) {
          os << std::endl;
        }
      }
    };
    for_range(slice_helper.begin1(), slice_helper.end1());
    if (slice_helper.has_summary()) {
      os << spaces << "..." << std::endl;
    }
    for_range(slice_helper.begin2(), slice_helper.end2());
    os << "]";
  }
}

}  // namespace detail

template <typename T>
std::string to_string(const Tensor<T>& tsr, int summary) {
  std::ostringstream os;
  detail::TensorWriteTextHelper<T> helper(tsr, summary);
  helper.WriteText(os);
  return os.str();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tsr) {
  detail::TensorWriteTextHelper<T> helper(tsr, 3);
  helper.WriteText(os);
  return os;
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other) {
  shape_ = other.shape_;
  if (!other.is_view()) {
    storage_ = other.storage_;
    data_ = storage_.data();
  } else {
    data_ = other.data_;
  }
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
  if (this != &other) {
    shape_ = other.shape_;
    if (!other.is_view()) {
      storage_ = other.storage_;
      data_ = storage_.data();
    } else {
      data_ = other.data_;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept {
  shape_ = std::move(other.shape_);
  storage_ = std::move(other.storage_);
  data_ = other.data_;
  other.shape_.clear();
  other.storage_.clear();
  other.data_ = nullptr;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    storage_ = std::move(other.storage_);
    data_ = other.data_;
    other.shape_.clear();
    other.storage_.clear();
    other.data_ = nullptr;
  }
  return *this;
}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<value_type> il) {
  int d0 = (int)il.size();
  if (d0 == 0) {
    return;
  }

  shape_.resize(d0);
  storage_.assign(il.begin(), il.end());
  data_ = storage_.data();
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(std::initializer_list<value_type> il) {
  return *this = Tensor{il};
}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<std::initializer_list<value_type>> il) {
  int d0 = (int)il.size();
  if (d0 == 0) {
    return;
  }

  int d1 = (int)(*il.begin()).size();
  for (const auto& _il : il) {
    if (d1 != (int)_il.size()) {
      DXTHROW_INVALID_ARGUMENT("Inconsistent col: %d vs %d.", d1,
                               (int)_il.size());
    }
  }

  if (d1 == 0) {
    return;
  }

  shape_.resize(d0, d1);
  storage_.resize(total_dim());
  data_ = storage_.data();
  pointer p = data_;
  for (const auto& _il : il) {
    for (value_type value : _il) {
      *p++ = value;
    }
  }
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(
    std::initializer_list<std::initializer_list<value_type>> il) noexcept {
  return *this = Tensor{il};
}

template <typename T>
Tensor<T>::Tensor(const std::vector<value_type>& data) {
  shape_.resize((int)data.size());
  storage_ = data;
  data_ = storage_.data();
}

template <typename T>
Tensor<T>::Tensor(std::vector<value_type>&& data) {
  shape_.resize((int)data.size());
  storage_ = std::move(data);
  data_ = storage_.data();
}

template <typename T>
template <typename II>
Tensor<T>::Tensor(II first, II last) {
  assign(first, last);
}

template <typename T>
template <typename II>
void Tensor<T>::assign(II first, II last) {
  int size = (int)std::distance(first, last);
  shape_.resize(size);
  storage_.assign(first, last);
  data_ = storage_.data();
}

template <typename T>
Tensor<T>& Tensor<T>::set_data(std::initializer_list<value_type> il) {
  if (total_dim() != (int)il.size()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent total dim: %d vs %d.", total_dim(),
                             (int)il.size());
  }
  std::copy(il.begin(), il.end(), data_);
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::set_data(const std::vector<value_type>& data) {
  if (total_dim() != (int)data.size()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent total dim: %d vs %d.", total_dim(),
                             (int)data.size());
  }
  std::copy(data.begin(), data.end(), data_);
  return *this;
}

template <typename T>
template <typename Int>
Tensor<T>& Tensor<T>::set_data(const_pointer data, Int size) {
  if (total_dim() != (int)size) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent total dim: %d vs %d.", total_dim(),
                             (int)size);
  }
  std::copy(data, data + size, data_);
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::set_data(const Tensor& other) {
  if (total_dim() != other.total_dim()) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent total dim: %d vs %d.", total_dim(),
                             other.total_dim());
  }
  std::copy(other.data_, other.data_ + total_dim(), data_);
  return *this;
}

template <typename T>
template <typename II>
Tensor<T>& Tensor<T>::set_data(II first, II last) {
  int size = (int)std::distance(first, last);
  if (total_dim() != size) {
    DXTHROW_INVALID_ARGUMENT("Inconsistent total dim: %d vs %d.", total_dim(),
                             size);
  }
  std::copy(first, last, data_);
  return *this;
}

template <typename T>
Tensor<T> Tensor<T>::get_view() const noexcept {
  Tensor view;
  view.shape_ = shape_;
  view.data_ = data_;
  return view;
}

template <typename T>
template <typename Int>
Tensor<T> Tensor<T>::operator[](Int i) {
  if ((int)i >= shape_.dim(0)) {
    DXTHROW_INVALID_ARGUMENT("Invalid i: %d.", (int)i);
  }
  Tensor slice;
  slice.shape_.assign(shape_.dim() + 1, shape_.dim() + shape_.rank());
  slice.data_ = data_ + shape_.total_dim() / shape_.dim(0) * i;
  return slice;
}

template <typename T>
template <typename Int>
const Tensor<T> Tensor<T>::operator[](Int i) const {
  if ((int)i >= shape_.dim(0)) {
    DXTHROW_INVALID_ARGUMENT("Invalid i: %d.", (int)i);
  }
  Tensor slice;
  slice.shape_.assign(shape_.dim() + 1, shape_.dim() + shape_.rank());
  slice.data_ = data_ + shape_.total_dim() / shape_.dim(0) * i;
  return slice;
}

template <typename T>
Tensor<T>& Tensor<T>::view(const Shape& shape, pointer data) noexcept {
  shape_ = shape;
  storage_.clear();
  data_ = data;
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::resize(int d0) {
  if (is_view()) {
    DXTHROW_RUNTIME_ERROR("Couldn't resize a tensor view.");
  }
  shape_.resize(d0);
  storage_.resize(total_dim());
  data_ = storage_.data();
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::resize(int d0, int d1) {
  if (is_view()) {
    DXTHROW_RUNTIME_ERROR("Couldn't resize a tensor view.");
  }
  shape_.resize(d0, d1);
  storage_.resize(total_dim());
  data_ = storage_.data();
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::resize(int d0, int d1, int d2) {
  if (is_view()) {
    DXTHROW_RUNTIME_ERROR("Couldn't resize a tensor view.");
  }
  shape_.resize(d0, d1, d2);
  storage_.resize(total_dim());
  data_ = storage_.data();
  return *this;
}

template <typename T>
template <typename... Args>
Tensor<T>& Tensor<T>::resize(Args&&... args) {
  if (is_view()) {
    DXTHROW_RUNTIME_ERROR("Couldn't resize a tensor view.");
  }
  shape_.resize(std::forward<Args>(args)...);
  storage_.resize(total_dim());
  data_ = storage_.data();
  return *this;
}

template <typename T>
template <typename... Args>
Tensor<T>& Tensor<T>::reshape(Args&&... args) {
  shape_.reshape(std::forward<Args>(args)...);
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::expand_dim(int axis) {
  shape_.expand_dim(axis);
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::squeeze(int axis) {
  shape_.squeeze(axis);
  return *this;
}

template <typename T>
bool Tensor<T>::same_shape(const Tensor& other) const noexcept {
  return shape_.same_shape(other.shape_);
}

template <typename T>
bool Tensor<T>::same_shape(Tensor& other) const noexcept {
  return shape_.same_shape(other.shape_);
}

template <typename T>
template <typename U>
bool Tensor<T>::same_shape(const Tensor<U>& other) const noexcept {
  return shape_.same_shape(other.shape_);
}

template <typename T>
template <typename U>
bool Tensor<T>::same_shape(Tensor<U>& other) const noexcept {
  return shape_.same_shape(other.shape_);
}

template <typename T>
template <typename... Args>
bool Tensor<T>::same_shape(Args&&... args) const noexcept {
  return shape_.same_shape(std::forward<Args>(args)...);
}

template <typename T>
template <typename Int>
void Tensor<T>::reserve(Int size) {
  if (is_view()) {
    DXTHROW_RUNTIME_ERROR("Couldn't reserve a tensor view.");
  }
  storage_.reserve((size_t)size);
}

template <typename T>
void Tensor<T>::clear() noexcept {
  shape_.clear();
  storage_.clear();
  data_ = nullptr;
}

template <typename T>
void Tensor<T>::swap(Tensor& other) noexcept {
  std::swap(shape_, other.shape_);
  storage_.swap(other.storage_);
  std::swap(data_, other.data_);
}

template <typename T>
auto Tensor<T>::sum() const noexcept -> value_type {
  static_assert(IS_FLOAT || IS_INT, "");
  value_type s = 0;
  for (int i = 0; i < total_dim(); ++i) {
    s += data_[i];
  }
  return s;
}

template <typename T>
auto Tensor<T>::mean() const noexcept -> value_type {
  static_assert(IS_FLOAT, "");
  if (empty()) {
    return 0;
  }
  return sum() / total_dim();
}

template <typename T>
auto Tensor<T>::asum() const noexcept -> value_type {
  static_assert(IS_FLOAT || IS_INT, "");
  value_type s = 0;
  for (int i = 0; i < total_dim(); ++i) {
    if (data_[i] >= 0) {
      s += data_[i];
    } else {
      s -= data_[i];
    }
  }
  return s;
}

template <typename T>
auto Tensor<T>::amean() const noexcept -> value_type {
  static_assert(IS_FLOAT, "");
  if (empty()) {
    return 0;
  }
  return asum() / total_dim();
}

template <typename T>
auto Tensor<T>::std() const noexcept -> value_type {
  static_assert(IS_FLOAT, "");
  return std::sqrt(var());
}

template <typename T>
auto Tensor<T>::var() const noexcept -> value_type {
  static_assert(IS_FLOAT, "");
  if (empty()) {
    return 0;
  }
  value_type m = mean();
  value_type s = 0;
  value_type tmp;
  for (int i = 0; i < total_dim(); ++i) {
    tmp = data_[i] - m;
    s += tmp * tmp;
  }
  return s / total_dim();
}

template <typename T>
Tensor<T>& Tensor<T>::arange() noexcept {
  static_assert(IS_FLOAT || IS_INT, "");
  for (int i = 0; i < total_dim(); ++i) {
    data_[i] = (value_type)i;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::constant(value_type c) noexcept {
  static_assert(IS_FLOAT || IS_INT, "");
  for (int i = 0; i < total_dim(); ++i) {
    data_[i] = c;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::zeros() noexcept {
  static_assert(IS_FLOAT || IS_INT, "");
  memset(data_, 0, total_dim() * sizeof(value_type));
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::ones() noexcept {
  static_assert(IS_FLOAT || IS_INT, "");
  return constant(1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand(RandomEngine&& engine, value_type _min,
                           value_type _max) noexcept {
  static_assert(IS_FLOAT, "");
  std::uniform_real_distribution<value_type> dist(_min, _max);
  for (int i = 0; i < total_dim(); ++i) {
    data_[i] = dist(engine);
  }
  return *this;
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand(RandomEngine&& engine) noexcept {
  static_assert(IS_FLOAT, "");
  return rand(engine, 0, 1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn(RandomEngine&& engine, value_type mean,
                            value_type stddev) noexcept {
  static_assert(IS_FLOAT, "");
  std::normal_distribution<value_type> dist(mean, stddev);
  for (int i = 0; i < total_dim(); ++i) {
    data_[i] = dist(engine);
  }
  return *this;
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn(RandomEngine&& engine) noexcept {
  static_assert(IS_FLOAT, "");
  return randn(engine, 0, 1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn_truncated(RandomEngine&& engine, value_type mean,
                                      value_type stddev) noexcept {
  static_assert(IS_FLOAT, "");
  std::normal_distribution<value_type> dist(mean, stddev);
  value_type upper = mean + 2 * stddev;
  value_type lower = mean - 2 * stddev;
  for (int i = 0; i < total_dim();) {
    data_[i] = dist(engine);
    if (data_[i] <= upper && data_[i] >= lower) {
      ++i;
    }
  }
  return *this;
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn_truncated(RandomEngine&& engine) noexcept {
  static_assert(IS_FLOAT, "");
  return randn_truncated(engine, 0, 1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand_variance_scaling(RandomEngine&& engine,
                                            value_type scale, int mode) {
  static_assert(IS_FLOAT, "");
  DXASSERT(is_rank(2));
  value_type n;
  switch (mode) {
    case 1:
      n = (value_type)dim(0);
      break;
    case 2:
      n = (value_type)dim(1);
      break;
    case 3:
    default:
      n = (value_type)(dim(0) + dim(1)) / 2;
      break;
  }
  value_type _max = std::sqrt(3 * scale / n);
  return rand(engine, -_max, _max);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn_variance_scaling(RandomEngine&& engine,
                                             value_type scale, int mode) {
  static_assert(IS_FLOAT, "");
  DXASSERT(is_rank(2));
  value_type n;
  switch (mode) {
    case 1:
      n = (value_type)dim(0);
      break;
    case 2:
      n = (value_type)dim(1);
      break;
    case 3:
    default:
      n = (value_type)(dim(0) + dim(1)) / 2;
      break;
  }
  value_type stddev = std::sqrt(scale / n);
  return randn_truncated(engine, 0, stddev);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand_lecun(RandomEngine&& engine) {
  static_assert(IS_FLOAT, "");
  return rand_variance_scaling(engine, 1, 1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn_lecun(RandomEngine&& engine) {
  static_assert(IS_FLOAT, "");
  return randn_variance_scaling(engine, 1, 1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand_xavier(RandomEngine&& engine) {
  static_assert(IS_FLOAT, "");
  return rand_variance_scaling(engine, 1, 3);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn_xavier(RandomEngine&& engine) {
  static_assert(IS_FLOAT, "");
  return randn_variance_scaling(engine, 1, 3);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand_he(RandomEngine&& engine) {
  static_assert(IS_FLOAT, "");
  return rand_variance_scaling(engine, 2, 1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::randn_he(RandomEngine&& engine) {
  static_assert(IS_FLOAT, "");
  return randn_variance_scaling(engine, 2, 1);
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand_int(RandomEngine&& engine, int _min,
                               int _max) noexcept {
  static_assert(IS_FLOAT || IS_INT, "");
  std::uniform_int_distribution<int> dist(_min, _max - 1);
  for (int i = 0; i < total_dim(); ++i) {
    data_[i] = (value_type)dist(engine);
  }
  return *this;
}

template <typename T>
template <class RandomEngine>
Tensor<T>& Tensor<T>::rand_init(RandomEngine&& engine, int initializer_type,
                                value_type initializer_param1,
                                value_type initializer_param2) {
  static_assert(IS_FLOAT, "");
  switch (initializer_type) {
    case TENSOR_INITIALIZER_TYPE_ZEROS:
      zeros();
      break;
    case TENSOR_INITIALIZER_TYPE_ONES:
      ones();
      break;
    case TENSOR_INITIALIZER_TYPE_CONSTANT:
      constant(initializer_param1);
      break;
    case TENSOR_INITIALIZER_TYPE_RAND:
      rand(engine, initializer_param1, initializer_param2);
      break;
    case TENSOR_INITIALIZER_TYPE_RANDN:
      randn(engine, initializer_param1, initializer_param2);
      break;
    case TENSOR_INITIALIZER_TYPE_RAND_LECUN:
      rand_lecun(engine);
      break;
    case TENSOR_INITIALIZER_TYPE_RANDN_LECUN:
      randn_lecun(engine);
      break;
    case TENSOR_INITIALIZER_TYPE_RAND_XAVIER:
      rand_xavier(engine);
      break;
    case TENSOR_INITIALIZER_TYPE_RANDN_XAVIER:
      randn_xavier(engine);
      break;
    case TENSOR_INITIALIZER_TYPE_RAND_HE:
      rand_he(engine);
      break;
    case TENSOR_INITIALIZER_TYPE_RANDN_HE:
      randn_he(engine);
      break;
    case TENSOR_INITIALIZER_TYPE_RAND_INT:
      rand_int(engine, (int)initializer_param1, (int)initializer_param2);
      break;
    case TENSOR_INITIALIZER_TYPE_ARANGE:
      arange();
      break;
  }
  return *this;
}

}  // namespace deepx_core
