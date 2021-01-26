// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <algorithm>  // std::equal
#include <cstddef>    // std::nullptr_t
#include <initializer_list>
#include <iterator>   // std::distance, std::reverse_iterator
#include <stdexcept>  // std::out_of_range, std::runtime_error
#include <utility>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* Vector */
/************************************************************************/
template <typename T>
class Vector {
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
  std::vector<value_type> storage_;
  pointer data_;
  size_type size_;

 public:
  Vector() noexcept;

  explicit Vector(size_type n);

  Vector(size_type n, const_reference value);

  template <typename II>
  Vector(II first, II last);

  Vector(const Vector& other);
  Vector& operator=(const Vector& other);

  Vector(Vector&& other) noexcept;
  Vector& operator=(Vector&& other) noexcept;

  Vector(std::initializer_list<value_type> il);
  Vector& operator=(std::initializer_list<value_type> il);

  explicit Vector(std::vector<T> other);
  Vector& operator=(const std::vector<T>& other);
  Vector& operator=(std::vector<T>&& other);

 public:
  void assign(size_type n, const_reference value);
  template <typename II>
  void assign(II first, II last);
  void assign(std::initializer_list<value_type> il);

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

 private:
  // raw iterator
  using raw_iterator = typename std::vector<value_type>::iterator;
  using raw_const_iterator = typename std::vector<value_type>::const_iterator;
  raw_iterator to_raw_iterator(const_iterator pos) noexcept {
    return storage_.begin() + (pos - data_);
  }
  iterator to_iterator(raw_iterator pos) noexcept {
    return data_ + std::distance(storage_.begin(), pos);
  }

 public:
  // size
  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  size_type max_size() const noexcept { return storage_.max_size(); }
  template <typename Int>
  void reserve(Int n);
  size_type capacity() const noexcept { return storage_.capacity(); }
  void shrink_to_fit();

 private:
  // view
  void set_view() noexcept;
  void set_view_non_empty() noexcept;

 public:
  // view
  bool is_view() const noexcept { return data_ && data_ != storage_.data(); }
  Vector get_view() const noexcept;
  Vector& view(pointer data, size_type n) noexcept;
  Vector& view(const_pointer data, size_type n) noexcept;

 public:
  // modifier
  void clear() noexcept;
  iterator insert(const_iterator pos, const_reference value);
  iterator insert(const_iterator pos, value_type&& value);
  iterator insert(const_iterator pos, size_type n, const_reference value);
  template <typename II>
  iterator insert(const_iterator pos, II first, II last);
  iterator insert(const_iterator pos, std::initializer_list<value_type> il);
  template <typename... Args>
  iterator emplace(const_iterator pos, Args&&... args);
  iterator erase(const_iterator pos);
  iterator erase(const_iterator first, const_iterator last);
  void push_back(const_reference value);
  void push_back(value_type&& value);
  template <typename... Args>
  void emplace_back(Args&&... args);
  void pop_back();
  void resize(size_type n);
  void resize(size_type n, const_reference value);
  void swap(Vector& other) noexcept;
};

// comparison
template <typename T>
bool operator==(const Vector<T>& left, const Vector<T>& right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

template <typename T>
bool operator!=(const Vector<T>& left, const Vector<T>& right) {
  return !(left == right);
}

template <typename T>
bool operator==(std::nullptr_t, Vector<T> right) noexcept {
  return right.data() == nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, Vector<T> right) noexcept {
  return right.data() != nullptr;
}

template <typename T>
bool operator==(Vector<T> left, std::nullptr_t) noexcept {
  return left.data() == nullptr;
}

template <typename T>
bool operator!=(Vector<T> left, std::nullptr_t) noexcept {
  return left.data() != nullptr;
}

/************************************************************************/
/* Vector */
/************************************************************************/
template <typename T>
Vector<T>::Vector() noexcept : storage_() {
  set_view();
}

template <typename T>
Vector<T>::Vector(size_type n) : storage_(n) {
  set_view();
}

template <typename T>
Vector<T>::Vector(size_type n, const_reference value) : storage_(n, value) {
  set_view();
}

template <typename T>
template <typename II>
Vector<T>::Vector(II first, II last) : storage_(first, last) {
  set_view();
}

template <typename T>
Vector<T>::Vector(const Vector& other) {
  if (!other.is_view()) {
    storage_ = other.storage_;
    data_ = storage_.data();
  } else {
    data_ = other.data_;
  }
  size_ = other.size_;
}

template <typename T>
Vector<T>& Vector<T>::operator=(const Vector& other) {
  if (this != &other) {
    if (!other.is_view()) {
      storage_ = other.storage_;
      data_ = storage_.data();
    } else {
      storage_.clear();
      data_ = other.data_;
    }
    size_ = other.size_;
  }
  return *this;
}

template <typename T>
Vector<T>::Vector(Vector&& other) noexcept {
  storage_ = std::move(other.storage_);
  data_ = other.data_;
  size_ = other.size_;
  other.storage_.clear();
  other.data_ = nullptr;
  other.size_ = 0;
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector&& other) noexcept {
  if (this != &other) {
    storage_ = std::move(other.storage_);
    data_ = other.data_;
    size_ = other.size_;
    other.storage_.clear();
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

template <typename T>
Vector<T>::Vector(std::initializer_list<value_type> il) : storage_(il) {
  set_view();
}

template <typename T>
Vector<T>& Vector<T>::operator=(std::initializer_list<value_type> il) {
  storage_ = il;
  set_view();
  return *this;
}

template <typename T>
Vector<T>::Vector(std::vector<T> other) : storage_(std::move(other)) {
  set_view();
}

template <typename T>
Vector<T>& Vector<T>::operator=(const std::vector<T>& other) {
  storage_ = other;
  set_view();
  return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(std::vector<T>&& other) {
  storage_ = other;
  set_view();
  other.clear();
  return *this;
}

template <typename T>
void Vector<T>::assign(size_type n, const_reference value) {
  storage_.assign(n, value);
  set_view();
}

template <typename T>
template <typename II>
void Vector<T>::assign(II first, II last) {
  storage_.assign(first, last);
  set_view();
}

template <typename T>
void Vector<T>::assign(std::initializer_list<value_type> il) {
  storage_.assign(il);
  set_view();
}

template <typename T>
template <typename Int>
void Vector<T>::reserve(Int n) {
  if (is_view()) {
    throw std::runtime_error("reserve: couldn't reserve a vector view.");
  }
  storage_.reserve((size_type)n);
}

template <typename T>
void Vector<T>::shrink_to_fit() {
  if (is_view()) {
    throw std::runtime_error(
        "shrink_to_fit: couldn't shrink_to_fit a vector view.");
  }
  storage_.shrink_to_fit();
}

template <typename T>
void Vector<T>::set_view() noexcept {
  if (storage_.empty()) {
    data_ = nullptr;
    size_ = 0;
  } else {
    data_ = storage_.data();
    size_ = storage_.size();
  }
}

template <typename T>
void Vector<T>::set_view_non_empty() noexcept {
  data_ = storage_.data();
  size_ = storage_.size();
}

template <typename T>
Vector<T> Vector<T>::get_view() const noexcept {
  Vector view;
  view.data_ = data_;
  view.size_ = size_;
  return view;
}

template <typename T>
Vector<T>& Vector<T>::view(pointer data, size_type n) noexcept {
  storage_.clear();
  data_ = data;
  size_ = n;
  return *this;
}

template <typename T>
Vector<T>& Vector<T>::view(const_pointer data, size_type n) noexcept {
  storage_.clear();
  // The cast is ugly and unsafe.
  data_ = (pointer)data;
  size_ = n;
  return *this;
}

template <typename T>
void Vector<T>::clear() noexcept {
  storage_.clear();
  data_ = nullptr;
  size_ = 0;
}

template <typename T>
auto Vector<T>::insert(const_iterator pos, const_reference value) -> iterator {
  if (is_view()) {
    throw std::runtime_error("insert: couldn't insert to a vector view.");
  }
  auto it = storage_.insert(to_raw_iterator(pos), value);
  set_view_non_empty();
  return to_iterator(it);
}

template <typename T>
auto Vector<T>::insert(const_iterator pos, value_type&& value) -> iterator {
  if (is_view()) {
    throw std::runtime_error("insert: couldn't insert to a vector view.");
  }
  auto it = storage_.insert(to_raw_iterator(pos), value);
  set_view_non_empty();
  return to_iterator(it);
}

template <typename T>
auto Vector<T>::insert(const_iterator pos, size_type n, const_reference value)
    -> iterator {
  if (is_view()) {
    throw std::runtime_error("insert: couldn't insert to a vector view.");
  }
  auto it = storage_.insert(to_raw_iterator(pos), n, value);
  set_view_non_empty();
  return to_iterator(it);
}

template <typename T>
template <typename II>
auto Vector<T>::insert(const_iterator pos, II first, II last) -> iterator {
  if (is_view()) {
    throw std::runtime_error("insert: couldn't insert to a vector view.");
  }
  size_type pos_offset = pos - data_;
  (void)storage_.insert(to_raw_iterator(pos), first, last);
  set_view_non_empty();
  return data_ + pos_offset;
}

template <typename T>
auto Vector<T>::insert(const_iterator pos, std::initializer_list<value_type> il)
    -> iterator {
  if (is_view()) {
    throw std::runtime_error("insert: couldn't insert to a vector view.");
  }
  size_type pos_offset = pos - data_;
  (void)storage_.insert(to_raw_iterator(pos), il);
  set_view_non_empty();
  return data_ + pos_offset;
}

template <typename T>
template <typename... Args>
auto Vector<T>::emplace(const_iterator pos, Args&&... args) -> iterator {
  if (is_view()) {
    throw std::runtime_error("emplace: couldn't emplace to a vector view.");
  }
  auto it = storage_.emplace(to_raw_iterator(pos), std::forward<Args>(args)...);
  set_view_non_empty();
  return to_iterator(it);
}

template <typename T>
auto Vector<T>::erase(const_iterator pos) -> iterator {
  if (is_view()) {
    throw std::runtime_error("erase: couldn't erase from a vector view.");
  }
  auto it = storage_.erase(to_raw_iterator(pos));
  set_view();
  return to_iterator(it);
}

template <typename T>
auto Vector<T>::erase(const_iterator first, const_iterator last) -> iterator {
  if (is_view()) {
    throw std::runtime_error("erase: couldn't erase from a vector view.");
  }
  auto it = storage_.erase(to_raw_iterator(first), to_raw_iterator(last));
  set_view();
  return to_iterator(it);
}

template <typename T>
template <typename... Args>
void Vector<T>::emplace_back(Args&&... args) {
  if (is_view()) {
    throw std::runtime_error(
        "emplace_back: couldn't emplace_back to a vector view.");
  }
  storage_.emplace_back(std::forward<Args>(args)...);
  set_view_non_empty();
}

template <typename T>
void Vector<T>::pop_back() {
  if (is_view()) {
    throw std::runtime_error("pop_back: couldn't pop_back from a vector view.");
  }
  storage_.pop_back();
  set_view();
}

template <typename T>
void Vector<T>::resize(size_type n) {
  if (is_view()) {
    throw std::runtime_error("resize: couldn't resize a vector view.");
  }
  storage_.resize(n);
  set_view();
}

template <typename T>
void Vector<T>::resize(size_type n, const_reference value) {
  if (is_view()) {
    throw std::runtime_error("resize: couldn't resize a vector view.");
  }
  storage_.resize(n, value);
  set_view();
}

template <typename T>
void Vector<T>::swap(Vector& other) noexcept {
  storage_.swap(other.storage_);
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
}

}  // namespace deepx_core
