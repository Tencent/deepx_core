// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <algorithm>  // std::equal
#include <array>
#include <cstddef>    // std::nullptr_t
#include <iterator>   // std::reverse_iterator
#include <stdexcept>  // std::out_of_range
#include <string>
#include <utility>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* ArrayView */
/************************************************************************/
template <typename T>
class ArrayView {
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = value_type*;
  using const_iterator = const value_type*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

 private:
  pointer data_ = nullptr;
  size_type size_ = 0;

 public:
  ArrayView() noexcept = default;

  template <size_type N>
  ArrayView(T (&a)[N]) noexcept  // NOLINT
      : data_(N > 0 ? &a[0] : nullptr), size_(N) {}

  ArrayView(T* a, size_type N) noexcept
      : data_(N > 0 ? a : nullptr), size_(N) {}

  template <size_type N>
  ArrayView(std::array<T, N>& a) noexcept  // NOLINT
      : data_(N > 0 ? a.data() : nullptr), size_(N) {}

  template <class Alloc>
  ArrayView(std::vector<T, Alloc>& v) noexcept  // NOLINT
      : data_(!v.empty() ? v.data() : nullptr), size_(v.size()) {}

  template <class Traits, class Alloc>
  ArrayView(std::basic_string<T, Traits, Alloc>& s) noexcept  // NOLINT
      : data_(!s.empty() ? &s[0] : nullptr), size_(s.size()) {}

 public:
  // iterator
  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return begin(); }
  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return end(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

 public:
  // element access
  template <typename Int>
  reference at(Int n) {
    if ((size_type)n >= size()) {
      throw std::out_of_range("at");
    }
    return data_[n];
  }

  template <typename Int>
  const_reference at(Int n) const {
    if ((size_type)n >= size()) {
      throw std::out_of_range("at");
    }
    return data_[n];
  }

  template <typename Int>
  reference operator[](Int n) noexcept {
    return data_[n];
  }

  template <typename Int>
  const_reference operator[](Int n) const noexcept {
    return data_[n];
  }

  reference front() noexcept { return data_[0]; }
  const_reference front() const noexcept { return data_[0]; }
  reference back() noexcept { return data_[size_ - 1]; }
  const_reference back() const noexcept { return data_[size_ - 1]; }
  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }

 public:
  // size
  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  explicit operator bool() const noexcept { return size_ > 0; }

 public:
  // modifier
  void clear() noexcept {
    data_ = nullptr;
    size_ = 0;
  }

  void remove_prefix(size_type n) noexcept {
    data_ += n;
    size_ -= n;
    if (size_ == 0) {
      data_ = nullptr;
    }
  }

  void remove_suffix(size_type n) noexcept {
    size_ -= n;
    if (size_ == 0) {
      data_ = nullptr;
    }
  }

  void swap(ArrayView& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }
};

// comparison
template <typename T>
bool operator==(const ArrayView<T>& left, const ArrayView<T>& right) noexcept {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

template <typename T>
bool operator!=(const ArrayView<T>& left, const ArrayView<T>& right) noexcept {
  return !(left == right);
}

template <typename T>
bool operator==(std::nullptr_t, ArrayView<T> right) noexcept {
  return right.data() == nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, ArrayView<T> right) noexcept {
  return right.data() != nullptr;
}

template <typename T>
bool operator==(ArrayView<T> left, std::nullptr_t) noexcept {
  return left.data() == nullptr;
}

template <typename T>
bool operator!=(ArrayView<T> left, std::nullptr_t) noexcept {
  return left.data() != nullptr;
}

using string_view = ArrayView<char>;

inline bool operator==(string_view left, const std::string& right) noexcept {
  return left.size() == right.size() && right.compare(left.data()) == 0;
}

inline bool operator!=(string_view left, const std::string& right) noexcept {
  return !(left == right);
}

inline bool operator==(const std::string& left, string_view right) noexcept {
  return left.size() == right.size() && left.compare(right.data()) == 0;
}

inline bool operator!=(const std::string& left, string_view right) noexcept {
  return !(left == right);
}

/************************************************************************/
/* ConstArrayView */
/************************************************************************/
template <typename T>
class ConstArrayView {
 public:
  using value_type = T;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  using const_iterator = const value_type*;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

 private:
  const_pointer data_ = nullptr;
  size_type size_ = 0;

 public:
  ConstArrayView() noexcept = default;

  template <size_type N>
  ConstArrayView(const T (&a)[N]) noexcept  // NOLINT
      : data_(N > 0 ? &a[0] : nullptr), size_(N) {}

  ConstArrayView(const T* a, size_type N) noexcept
      : data_(N > 0 ? a : nullptr), size_(N) {}

  template <size_type N>
  ConstArrayView(const std::array<T, N>& a) noexcept  // NOLINT
      : data_(N > 0 ? a.data() : nullptr), size_(N) {}

  template <class Alloc>
  ConstArrayView(const std::vector<T, Alloc>& v) noexcept  // NOLINT
      : data_(!v.empty() ? v.data() : nullptr), size_(v.size()) {}

  template <class Traits, class Alloc>
  ConstArrayView(  // NOLINT
      const std::basic_string<T, Traits, Alloc>& s) noexcept
      : data_(!s.empty() ? s.data() : nullptr), size_(s.size()) {}

  ConstArrayView(const ArrayView<T>& a) noexcept  // NOLINT
      : data_(a.data()), size_(a.size()) {}

 public:
  // iterator
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return end(); }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

 public:
  // element access
  template <typename Int>
  const_reference at(Int n) const {
    if ((size_type)n >= size()) {
      throw std::out_of_range("at");
    }
    return data_[n];
  }

  template <typename Int>
  const_reference operator[](Int n) const noexcept {
    return data_[n];
  }

  const_reference front() const noexcept { return data_[0]; }
  const_reference back() const noexcept { return data_[size_ - 1]; }
  const_pointer data() const noexcept { return data_; }

 public:
  // size
  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  explicit operator bool() const noexcept { return size_ > 0; }

 public:
  // modifier
  void clear() noexcept {
    data_ = nullptr;
    size_ = 0;
  }

  void remove_prefix(size_type n) noexcept {
    data_ += n;
    size_ -= n;
    if (size_ == 0) {
      data_ = nullptr;
    }
  }

  void remove_suffix(size_type n) noexcept {
    size_ -= n;
    if (size_ == 0) {
      data_ = nullptr;
    }
  }

  void swap(ConstArrayView& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }
};

// comparison
template <typename T>
bool operator==(const ConstArrayView<T>& left,
                const ConstArrayView<T>& right) noexcept {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

template <typename T>
bool operator!=(const ConstArrayView<T>& left,
                const ConstArrayView<T>& right) noexcept {
  return !(left == right);
}

template <typename T>
bool operator==(std::nullptr_t, ConstArrayView<T> right) noexcept {
  return right.data() == nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, ConstArrayView<T> right) noexcept {
  return right.data() != nullptr;
}

template <typename T>
bool operator==(ConstArrayView<T> left, std::nullptr_t) noexcept {
  return left.data() == nullptr;
}

template <typename T>
bool operator!=(ConstArrayView<T> left, std::nullptr_t) noexcept {
  return left.data() != nullptr;
}

using const_string_view = ConstArrayView<char>;

inline bool operator==(const_string_view left,
                       const std::string& right) noexcept {
  return left.size() == right.size() && right.compare(left.data()) == 0;
}

inline bool operator!=(const_string_view left,
                       const std::string& right) noexcept {
  return !(left == right);
}

inline bool operator==(const std::string& left,
                       const_string_view right) noexcept {
  return left.size() == right.size() && left.compare(right.data()) == 0;
}

inline bool operator!=(const std::string& left,
                       const_string_view right) noexcept {
  return !(left == right);
}

}  // namespace deepx_core
