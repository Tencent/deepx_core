// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <memory>
#include <type_traits>  // std::decay, ...
#include <typeindex>
#include <typeinfo>
#include <utility>

namespace deepx_core {

/************************************************************************/
/* Any */
/************************************************************************/
class Any {
 private:
  class Base {
   private:
    friend class Any;
    const std::type_index index_;

   public:
    explicit Base(const std::type_info& info) noexcept : index_(info) {}
    virtual ~Base() = default;
    virtual Base* clone() const = 0;
  };

  template <typename T>
  class Container : public Base {
   private:
    friend class Any;
    using value_type = typename std::decay<T>::type;
    value_type value_;

   public:
    template <typename U>
    explicit Container(U&& value)
        : Base(typeid(value_type)), value_(std::forward<U>(value)) {}

    template <typename... Args>
    explicit Container(Args&&... args)
        : Base(typeid(value_type)), value_(std::forward<Args>(args)...) {}

    Container* clone() const override {
      static_assert(std::is_copy_constructible<value_type>::value,
                    "value_type must be copy constructible.");
      return new Container(value_);
    }
  };

  std::unique_ptr<Base> p_;

 public:
  Any() = default;

  Any(const Any& other) {
    if (other.p_) {
      p_.reset(other.p_->clone());
    }
  }

  Any(Any& other) : Any((const Any&)other) {  // NOLINT
  }

  Any& operator=(const Any& other) {
    if (this != &other) {
      if (other.p_) {
        p_.reset(other.p_->clone());
      } else {
        p_.reset();
      }
    }
    return *this;
  }

  Any& operator=(Any& other) {  // NOLINT
    return operator=((const Any&)other);
  }

  Any(Any&& other) noexcept : p_(std::move(other.p_)) {}

  Any& operator=(Any&& other) noexcept {
    if (this != &other) {
      p_ = std::move(other.p_);
    }
    return *this;
  }

  template <typename T>
  Any(T&& value) {  // NOLINT
    emplace(std::forward<T>(value));
  }

  template <typename T>
  Any& operator=(T&& value) {
    emplace(std::forward<T>(value));
    return *this;
  }

 public:
  template <typename T>
  void emplace(T&& value) {
    p_.reset(new Container<T>(std::forward<T>(value)));
  }

  template <typename T, typename... Args>
  void emplace(Args&&... args) {
    p_.reset(new Container<T>(std::forward<Args>(args)...));
  }

  void reset() noexcept { p_.reset(); }

  void swap(Any& other) noexcept { p_.swap(other.p_); }

  bool empty() const noexcept { return !p_.operator bool(); }

  explicit operator bool() const noexcept { return p_.operator bool(); }

  std::type_index type_index() const noexcept {
    if (p_) {
      return p_->index_;
    }
    return typeid(void);
  }

 public:
  template <typename T>
  bool is() const noexcept {
    if (p_) {
      return p_->index_ == typeid(T);
    }
    return typeid(void) == typeid(T);
  }

  template <typename T>
  T& to_ref() {
    if (is<T>()) {
      return ((Container<T>*)p_.get())->value_;
    }
    throw std::bad_cast();
  }

  template <typename T>
  const T& to_ref() const {
    if (is<T>()) {
      return ((const Container<T>*)p_.get())->value_;
    }
    throw std::bad_cast();
  }

  template <typename T>
  T& unsafe_to_ref() noexcept {
    return ((Container<T>*)p_.get())->value_;
  }

  template <typename T>
  const T& unsafe_to_ref() const noexcept {
    return ((const Container<T>*)p_.get())->value_;
  }
};

}  // namespace deepx_core
