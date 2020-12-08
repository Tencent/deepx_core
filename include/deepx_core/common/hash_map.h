// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <functional>  // std::hash
#include <initializer_list>
#include <stdexcept>    // std::out_of_range
#include <type_traits>  // std::is_integral
#include <utility>
#include <vector>

namespace deepx_core {
namespace detail {

template <typename Key>
struct KeyHash {
  using key_type = Key;
  using size_type = size_t;

  size_type operator()(const key_type& k) const noexcept {
    return operator()(k, typename std::is_integral<key_type>::type());
  }

  size_type operator()(const key_type& k, std::true_type /*is_integral*/) const
      noexcept {
    // Dispatch integral types to their integral values.
    return (size_type)k;
  }

  size_type operator()(const key_type& k, std::false_type /*is_integral*/) const
      noexcept {
    // Dispatch non-integral types to std::hash.
    return std::hash<key_type>()(k);
  }
};

template <typename Key>
struct KeyEqual {
  using key_type = Key;

  bool operator()(const key_type& left, const key_type& right) const noexcept {
    return left == right;
  }
};

template <size_t>
struct RoundUpPow2;

template <>
struct RoundUpPow2<4> {
  template <typename T>
  T operator()(T n) const noexcept {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (T)(n + 1);
  }
};

template <>
struct RoundUpPow2<8> {
  template <typename T>
  T operator()(T n) const noexcept {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return (T)(n + 1);
  }
};

constexpr size_t HASH_MAP_INVALID_SLOT_INDEX = (size_t)-1;
constexpr size_t HASH_MAP_MIN_BUCKET_SIZE = 128;
constexpr double HASH_MAP_MAX_LOAD_FACTOR = 2.0;
constexpr double HASH_MAP_INV_MIN_LOAD_FACTOR = 1 / 1.5;

}  // namespace detail

template <typename Key, typename Value, class KeyHash = detail::KeyHash<Key>,
          class KeyEqual = detail::KeyEqual<Key>>
class HashMap;
template <class HashMap>
class HashMapIterator;
template <class HashMap>
class HashMapConstIterator;

/************************************************************************/
/* HashMapIterator */
/************************************************************************/
template <class HashMap>
class HashMapIterator {
 public:
  using hash_map_t = HashMap;
  using value_type = typename hash_map_t::value_type;
  using size_type = typename hash_map_t::size_type;
  using _iterator = HashMapIterator<hash_map_t>;
  using _const_iterator = HashMapConstIterator<hash_map_t>;
  friend hash_map_t;
  friend _const_iterator;

 private:
  hash_map_t* map_ = nullptr;
  size_type index_ = detail::HASH_MAP_INVALID_SLOT_INDEX;

 public:
  HashMapIterator() = default;

  HashMapIterator(hash_map_t* map, size_type index) noexcept
      : map_(map), index_(index) {}

  bool operator==(const _iterator& right) const noexcept {
    return map_ == right.map_ && index_ == right.index_;
  }

  bool operator!=(const _iterator& right) const noexcept {
    return !(operator==(right));
  }

  bool operator==(const _const_iterator& right) const noexcept;

  bool operator!=(const _const_iterator& right) const noexcept {
    return !(operator==(right));
  }

  _iterator& operator++() noexcept {
    index_ = map_->find_next_used_slot(index_ + 1);
    return *this;
  }

  _iterator operator++(int) noexcept {
    _iterator origin = *this;
    operator++();
    return origin;
  }

  // value_type.first is not const,
  // it is dangerous to modify it through an iterator.
  value_type& operator*() const noexcept { return map_->bucket_[index_]; }
  value_type* operator->() const noexcept { return &map_->bucket_[index_]; }
};

/************************************************************************/
/* HashMapConstIterator */
/************************************************************************/
template <class HashMap>
class HashMapConstIterator {
 public:
  using hash_map_t = HashMap;
  using value_type = typename hash_map_t::value_type;
  using size_type = typename hash_map_t::size_type;
  using _iterator = HashMapIterator<hash_map_t>;
  using _const_iterator = HashMapConstIterator<hash_map_t>;
  friend hash_map_t;
  friend _iterator;

 private:
  const hash_map_t* map_ = nullptr;
  size_type index_ = detail::HASH_MAP_INVALID_SLOT_INDEX;

 public:
  HashMapConstIterator() = default;

  HashMapConstIterator(const hash_map_t* map, size_type index) noexcept
      : map_(map), index_(index) {}

  HashMapConstIterator(const _iterator& it) noexcept  // NOLINT
      : map_(it.map_), index_(it.index_) {}

  bool operator==(const _const_iterator& right) const noexcept {
    return map_ == right.map_ && index_ == right.index_;
  }

  bool operator!=(const _const_iterator& right) const noexcept {
    return !(operator==(right));
  }

  bool operator==(const _iterator& right) const noexcept {
    return map_ == right.map_ && index_ == right.index_;
  }

  bool operator!=(const _iterator& right) const noexcept {
    return !(operator==(right));
  }

  _const_iterator& operator++() noexcept {
    index_ = map_->find_next_used_slot(index_ + 1);
    return *this;
  }

  _const_iterator operator++(int) noexcept {
    _const_iterator origin = *this;
    operator++();
    return origin;
  }

  const value_type& operator*() const noexcept { return map_->bucket_[index_]; }

  const value_type* operator->() const noexcept {
    return &map_->bucket_[index_];
  }
};

/************************************************************************/
/* HashMapIterator */
/************************************************************************/
template <class HashMap>
bool HashMapIterator<HashMap>::operator==(
    const HashMapConstIterator<HashMap>& right) const noexcept {
  return map_ == right.map_ && index_ == right.index_;
}

/************************************************************************/
/* HashMap */
/************************************************************************/
template <typename Key, typename Value, class KeyHash, class KeyEqual>
class HashMap {
 public:
  using key_type = Key;
  using mapped_type = Value;
  using value_type = std::pair<key_type, mapped_type>;
  using size_type = size_t;

 private:
  enum META_FLAG {
    META_FLAG_EMPTY = 0,
    META_FLAG_USED = 1,
    META_FLAG_DELETED = 2,
  };
  using meta_t = std::vector<char>;
  using bucket_t = std::vector<value_type>;
  using key_hash_t = KeyHash;
  using key_equal_t = KeyEqual;
  meta_t meta_;
  bucket_t bucket_;
  key_hash_t key_hash_;
  key_equal_t key_equal_;
  size_type rehash_threshold_ = 0;
  size_type size_ = 0;

 private:
  static size_type find_slot(const meta_t& meta, const bucket_t& bucket,
                             const key_hash_t& key_hash,
                             const key_equal_t& key_equal,
                             key_type k) noexcept {
    if (bucket.empty()) {
      return detail::HASH_MAP_INVALID_SLOT_INDEX;
    }

    size_type hash_mask = bucket.size() - 1;
    size_type hash_value = key_hash(k);
    size_type index;
    size_type first_deleted_index = detail::HASH_MAP_INVALID_SLOT_INDEX;
    int deleted_mode = 0;
    for (;;) {
      index = hash_value & hash_mask;
      switch (meta[index]) {
        case META_FLAG_USED:
          if (key_equal(bucket[index].first, k)) {
            // used & found
            return index;
          }
          break;
        case META_FLAG_EMPTY:
          // empty
          return deleted_mode ? first_deleted_index : index;
        case META_FLAG_DELETED:
          if (!deleted_mode) {
            first_deleted_index = index;
            deleted_mode = 1;
          }
          break;
      }

      // linear probe
      ++hash_value;
    }
  }

  static size_type find_used_slot(const meta_t& meta, const bucket_t& bucket,
                                  const key_hash_t& key_hash,
                                  const key_equal_t& key_equal,
                                  key_type k) noexcept {
    if (bucket.empty()) {
      return detail::HASH_MAP_INVALID_SLOT_INDEX;
    }

    size_type hash_mask = bucket.size() - 1;
    size_type hash_value = key_hash(k);
    size_type index;
    for (;;) {
      index = hash_value & hash_mask;
      switch (meta[index]) {
        case META_FLAG_USED:
          if (key_equal(bucket[index].first, k)) {
            // used & found
            return index;
          }
          break;
        case META_FLAG_EMPTY:
          // empty
          return detail::HASH_MAP_INVALID_SLOT_INDEX;
      }

      // linear probe
      ++hash_value;
    }
  }

  static size_type find_next_used_slot(const meta_t& meta,
                                       const bucket_t& bucket,
                                       size_type index) noexcept {
    size_type bucket_size = bucket.size();
    if (index >= bucket_size) {
      return detail::HASH_MAP_INVALID_SLOT_INDEX;
    }
    while (meta[index] != META_FLAG_USED) {
      if (++index >= bucket_size) {
        return detail::HASH_MAP_INVALID_SLOT_INDEX;
      }
    }
    return index;
  }

  static size_type next_size(size_type size) noexcept {
    size = detail::RoundUpPow2<sizeof(size_type)>()(
        (size_type)(size * detail::HASH_MAP_MAX_LOAD_FACTOR));
    if (size < detail::HASH_MAP_MIN_BUCKET_SIZE) {
      size = detail::HASH_MAP_MIN_BUCKET_SIZE;
    }
    return size;
  }

  void resize_bucket(size_type size) {
    size = next_size(size);
    meta_.resize(size);
    bucket_.resize(size);
    rehash_threshold_ =
        (size_type)(bucket_.size() * detail::HASH_MAP_INV_MIN_LOAD_FACTOR);
  }

  void do_rehash(size_type new_size) {
    meta_t new_meta(new_size);
    bucket_t new_bucket(new_size);

    for (size_type i = 0; i < bucket_.size(); ++i) {
      if (meta_[i] == META_FLAG_USED) {
        value_type& kv = bucket_[i];
        size_type index =
            find_slot(new_meta, new_bucket, key_hash_, key_equal_, kv.first);
        new_meta[index] = META_FLAG_USED;
        new_bucket[index] = std::move(kv);
      }
    }

    new_meta.swap(meta_);
    new_bucket.swap(bucket_);
    rehash_threshold_ =
        (size_type)(bucket_.size() * detail::HASH_MAP_INV_MIN_LOAD_FACTOR);
  }

  void rehash_for_emplace() {
    if (size_ >= rehash_threshold_) {
      size_type new_size = next_size(size_ + 1);
      do_rehash(new_size);
    }
  }

  size_type find_slot(key_type k) const noexcept {
    return find_slot(meta_, bucket_, key_hash_, key_equal_, k);
  }

  template <typename... Args>
  size_type find_slot(key_type k, Args&&...) const noexcept {
    return find_slot(k);
  }

  template <typename... Args>
  size_type find_slot(const value_type& kv, Args&&...) const noexcept {
    return find_slot(kv.first);
  }

  size_type find_used_slot(key_type k) const noexcept {
    return find_used_slot(meta_, bucket_, key_hash_, key_equal_, k);
  }

  size_type find_next_used_slot(size_type index) const noexcept {
    return find_next_used_slot(meta_, bucket_, index);
  }

 public:
  HashMap() = default;

  explicit HashMap(size_type initial_size,
                   const key_hash_t& key_hash = key_hash_t(),
                   const key_equal_t& key_equal = key_equal_t())
      : key_hash_(key_hash), key_equal_(key_equal) {
    resize_bucket(initial_size);
  }

  template <typename II>
  HashMap(II first, II last, size_type initial_size,
          const key_hash_t& key_hash = key_hash_t(),
          const key_equal_t& key_equal = key_equal_t())
      : key_hash_(key_hash), key_equal_(key_equal) {
    resize_bucket(initial_size);
    for (; first != last; ++first) {
      emplace(*first);
    }
  }

  HashMap(const HashMap&) = default;
  HashMap& operator=(const HashMap&) = default;

  HashMap(HashMap&& other) noexcept {
    meta_ = std::move(other.meta_);
    bucket_ = std::move(other.bucket_);
    key_hash_ = std::move(other.key_hash_);
    key_equal_ = std::move(other.key_equal_);
    rehash_threshold_ = other.rehash_threshold_;
    size_ = other.size_;
    other.meta_.clear();
    other.bucket_.clear();
    other.rehash_threshold_ = 0;
    other.size_ = 0;
  }

  HashMap& operator=(HashMap&& other) noexcept {
    if (this != &other) {
      meta_ = std::move(other.meta_);
      bucket_ = std::move(other.bucket_);
      key_hash_ = std::move(other.key_hash_);
      key_equal_ = std::move(other.key_equal_);
      rehash_threshold_ = other.rehash_threshold_;
      size_ = other.size_;
      other.meta_.clear();
      other.bucket_.clear();
      other.rehash_threshold_ = 0;
      other.size_ = 0;
    }
    return *this;
  }

  HashMap(std::initializer_list<value_type> il) {
    resize_bucket(il.size());
    for (const value_type& kv : il) {
      emplace(kv);
    }
  }

  HashMap& operator=(std::initializer_list<value_type> il) {
    clear();
    resize_bucket(il.size());
    for (const value_type& kv : il) {
      emplace(kv);
    }
    return *this;
  }

 public:
  // iterator
  using iterator = HashMapIterator<HashMap>;
  using const_iterator = HashMapConstIterator<HashMap>;
  friend iterator;
  friend const_iterator;

  iterator begin() noexcept { return iterator(this, find_next_used_slot(0)); }
  const_iterator begin() const noexcept {
    return const_iterator(this, find_next_used_slot(0));
  }
  const_iterator cbegin() const noexcept {
    return const_iterator(this, find_next_used_slot(0));
  }
  iterator end() noexcept {
    return iterator(this, detail::HASH_MAP_INVALID_SLOT_INDEX);
  }
  const_iterator end() const noexcept {
    return const_iterator(this, detail::HASH_MAP_INVALID_SLOT_INDEX);
  }
  const_iterator cend() const noexcept {
    return const_iterator(this, detail::HASH_MAP_INVALID_SLOT_INDEX);
  }

 public:
  // comparison
  bool operator==(const HashMap& right) const noexcept {
    if (size() != right.size()) {
      return false;
    }

    const_iterator first = cbegin(), last = cend(), it;
    for (; first != last; ++first) {
      it = right.find(first->first);
      if ((it == right.end()) || !(it->second == first->second)) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const HashMap& right) const noexcept {
    return !(operator==(right));
  }

 public:
  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  size_type bucket_size() const noexcept { return bucket_.size(); }

 public:
  mapped_type& operator[](key_type k) {
    rehash_for_emplace();
    size_type index = find_slot(k);
    char& flag = meta_[index];
    value_type& kv = bucket_[index];
    if (flag != META_FLAG_USED) {
      kv.first = k;
      kv.second = mapped_type{};
      ++size_;
      flag = META_FLAG_USED;
    }
    return kv.second;
  }

  mapped_type& at(key_type k) {
    size_type index = find_used_slot(k);
    if (index == detail::HASH_MAP_INVALID_SLOT_INDEX) {
      throw std::out_of_range("at");
    }
    return bucket_[index].second;
  }

  const mapped_type& at(key_type k) const {
    size_type index = find_used_slot(k);
    if (index == detail::HASH_MAP_INVALID_SLOT_INDEX) {
      throw std::out_of_range("at");
    }
    return bucket_[index].second;
  }

 public:
  iterator find(key_type k) noexcept {
    return iterator(this, find_used_slot(k));
  }

  const_iterator find(key_type k) const noexcept {
    return const_iterator(this, find_used_slot(k));
  }

  size_type count(key_type k) const noexcept {
    size_type index = find_used_slot(k);
    if (index == detail::HASH_MAP_INVALID_SLOT_INDEX) {
      return 0;
    }
    return 1;
  }

  std::pair<iterator, bool> insert(const value_type& kv) { return emplace(kv); }

  template <typename II>
  void insert(II first, II last) {
    for (; first != last; ++first) {
      emplace(*first);
    }
  }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    rehash_for_emplace();
    size_type index = find_slot(std::forward<Args>(args)...);
    char& flag = meta_[index];
    if (flag == META_FLAG_USED) {
      return std::make_pair(iterator(this, index), false);
    } else {
      bucket_[index] = value_type{std::forward<Args>(args)...};
      ++size_;
      flag = META_FLAG_USED;
      return std::make_pair(iterator(this, index), true);
    }
  }

 private:
  // Nothing is actually cleared.
  static void clear_value(value_type* /*kv*/) {}
  static void clear_bucket(bucket_t* /*bucket*/) {}

 public:
  iterator erase(const_iterator pos) {
    size_type index = pos.index_;
    char& flag = meta_[index];
    if (flag == META_FLAG_USED) {
      clear_value(&bucket_[index]);
      --size_;
      flag = META_FLAG_DELETED;
    }
    return iterator(this, find_next_used_slot(index + 1));
  }

  void clear() {
    meta_.assign(meta_.size(), META_FLAG_EMPTY);
    clear_bucket(&bucket_);
    rehash_threshold_ = 0;
    size_ = 0;
  }

  template <typename Int>
  void rehash(Int _new_size) {
    size_type new_size = next_size((size_type)_new_size);
    if (new_size > bucket_.size()) {
      do_rehash(new_size);
    }
  }

  template <typename Int>
  void reserve(Int size) {
    rehash(size);
  }

  void swap(HashMap& other) noexcept {
    meta_.swap(other.meta_);
    bucket_.swap(other.bucket_);
    std::swap(key_hash_, other.key_hash_);
    std::swap(key_equal_, other.key_equal_);
    std::swap(rehash_threshold_, other.rehash_threshold_);
    std::swap(size_, other.size_);
  }
};

}  // namespace deepx_core
