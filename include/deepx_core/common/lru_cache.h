// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <cstddef>  // offsetof
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* LRUCache */
/************************************************************************/
template <typename Key, typename Value, class KeyHash = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>>
class LRUCache {
 public:
  using key_type = Key;
  using mapped_type = Value;
  using size_type = size_t;

  struct ListNode {
    ListNode* next;  // NOLINT
    ListNode* prev;  // NOLINT
  };

  class Node {
   private:
    friend class LRUCache;
    const key_type key_;
    mapped_type value_;   // NOLINT
    size_type key_hash_;  // NOLINT
    int in_cache_;        // NOLINT
    int ref_;             // NOLINT
    ListNode hash_node_;  // NOLINT
    ListNode lru_node_;   // NOLINT

   public:
    explicit Node(const key_type& key) : key_(key) {}
    const key_type& key() const noexcept { return key_; };
    mapped_type* mutable_value() noexcept { return &value_; };
    const mapped_type& value() const noexcept { return value_; };
  };

  using node_type = Node;
  using node_pointer =
      std::unique_ptr<node_type, std::function<void(node_type*)>>;

 private:
  using key_hash_t = KeyHash;
  using key_equal_t = KeyEqual;
  key_hash_t key_hash_;
  key_equal_t key_equal_;
  size_type size_ = 0;
  size_type capacity_ = 0;
  std::vector<ListNode> hash_bucket_;
  size_type hash_bucket_mask_ = 0;
  ListNode lru_head_;

 public:
  LRUCache() { list_init_head(&lru_head_); }
  ~LRUCache() { clear(); }
  LRUCache(const LRUCache&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;

  // Initialize the LRU cache.
  //
  // It can hold 'capacity' elements at most.
  // Extra elements will be evicted according to the LRU policy.
  void init(size_type capacity) {
    clear();

    capacity_ = 4;  // magic number
    while (capacity_ < capacity) {
      capacity_ *= 2;
    }

    hash_bucket_.resize(capacity_);
    for (size_type i = 0; i < capacity_; ++i) {
      list_init_head(&hash_bucket_[i]);
    }

    hash_bucket_mask_ = hash_bucket_.size() - 1;
  }

  // Clear all elements.
  void clear() noexcept {
    ListNode* lru_node = lru_head_.next;
    ListNode* lru_next;
    for (; lru_node != &lru_head_;) {
      lru_next = lru_node->next;
      release(lru_node_to_node(lru_node));
      lru_node = lru_next;
    }

    size_ = 0;
    capacity_ = 0;
    hash_bucket_.clear();
    list_init_head(&lru_head_);
  }

  // Return the number of elements.
  size_type size() const noexcept { return size_; }

  // Return the capacity.
  size_type capacity() const noexcept { return capacity_; }

  // Return if the number of elements is zero.
  bool empty() const noexcept { return size_ == 0; }

  // Get or insert node by 'key'.
  //
  // Return the node.
  node_pointer get_or_insert(const key_type& key) {
    size_type key_hash = key_hash_(key);
    ListNode* hash_node;
    if (hash_get(key, key_hash, &hash_node)) {
      node_type* node = hash_node_to_node(hash_node);
      ref(node);
      return create_node_pointer(node);
    }

    node_pointer new_node = create_node_pointer(key);
    new_node->value_ = mapped_type();
    new_node->key_hash_ = key_hash;
    new_node->in_cache_ = 1;
    new_node->ref_ = 2;
    hash_append(hash_node, new_node.get());
    lru_append(&lru_head_, new_node.get());
    if (++size_ > capacity_) {
      evict();
    }
    return new_node;
  }

  // Insert 'key' and 'value', existing element will be updated.
  //
  // Return the node.
  // NOTE: the semantic is different from STL's insert.
  template <typename... Args>
  node_pointer insert(const key_type& key, const mapped_type& value) {
    size_type key_hash = key_hash_(key);
    ListNode* hash_node;
    if (hash_get(key, key_hash, &hash_node)) {
      node_type* node = hash_node_to_node(hash_node);
      node->value_ = value;
      ref(node);
      return create_node_pointer(node);
    }

    node_pointer new_node = create_node_pointer(key);
    new_node->value_ = value;
    new_node->key_hash_ = key_hash;
    new_node->in_cache_ = 1;
    new_node->ref_ = 2;
    hash_append(hash_node, new_node.get());
    lru_append(&lru_head_, new_node.get());
    if (++size_ > capacity_) {
      evict();
    }
    return new_node;
  }

  // Insert 'key' and value, existing element will be updated.
  //
  // Return the node.
  // NOTE: the semantic is different from STL's emplace.
  template <typename... Args>
  node_pointer emplace(const key_type& key, Args&&... value_args) {
    size_type key_hash = key_hash_(key);
    ListNode* hash_node;
    if (hash_get(key, key_hash, &hash_node)) {
      node_type* node = hash_node_to_node(hash_node);
      node->value_ = mapped_type(std::forward<Args>(value_args)...);
      ref(node);
      return create_node_pointer(node);
    }

    node_pointer new_node = create_node_pointer(key);
    new_node->value_ = mapped_type(std::forward<Args>(value_args)...);
    new_node->key_hash_ = key_hash;
    new_node->in_cache_ = 1;
    new_node->ref_ = 2;
    hash_append(hash_node, new_node.get());
    lru_append(&lru_head_, new_node.get());
    if (++size_ > capacity_) {
      evict();
    }
    return new_node;
  }

  // Get node by 'key'.
  //
  // Return the node if 'key' exists.
  // Return nullptr if 'key' does not exist.
  node_pointer get(const key_type& key) noexcept {
    size_type key_hash = key_hash_(key);
    ListNode* hash_node;
    if (hash_get(key, key_hash, &hash_node)) {
      node_type* node = hash_node_to_node(hash_node);
      ref(node);
      return create_node_pointer(node);
    }
    return create_node_pointer(nullptr);
  }

  // Erase by 'key'.
  //
  // Return 1 if 'key' exists and the associated element is removed.
  // Return 0 if 'key' does not exist.
  size_type erase(const key_type& key) noexcept {
    size_type key_hash = key_hash_(key);
    ListNode* hash_node;
    if (hash_get(key, key_hash, &hash_node)) {
      node_type* node = hash_node_to_node(hash_node);
      release(node);
      return 1;
    }
    return 0;
  }

 private:
  static node_type* hash_node_to_node(ListNode* node) noexcept {
    union {
      void* pv;
      char* pc;
    } u;
    u.pv = node;
    u.pc -= offsetof(node_type, hash_node_);
    return (node_type*)u.pv;
  }

  static node_type* lru_node_to_node(ListNode* node) noexcept {
    union {
      void* pv;
      char* pc;
    } u;
    u.pv = node;
    u.pc -= offsetof(node_type, lru_node_);
    return (node_type*)u.pv;
  }

  static void list_init_head(ListNode* head) noexcept {
    head->next = head;
    head->prev = head;
  }

  static void list_remove(ListNode* node) noexcept {
    node->next->prev = node->prev;
    node->prev->next = node->next;
  }

  static void list_append(ListNode* head, ListNode* node) noexcept {
    node->next = head;
    node->prev = head->prev;
    node->prev->next = node;
    node->next->prev = node;
  }

  static void hash_remove(node_type* node) noexcept {
    list_remove(&node->hash_node_);
  }

  static void hash_append(ListNode* head, node_type* node) noexcept {
    list_append(head, &node->hash_node_);
  }

  static void lru_remove(node_type* node) noexcept {
    list_remove(&node->lru_node_);
  }

  static void lru_append(ListNode* head, node_type* node) noexcept {
    list_append(head, &node->lru_node_);
  }

  void release(node_type* node) noexcept {
    if (node->in_cache_) {
      if (--node->ref_ == 0) {
        hash_remove(node);
        lru_remove(node);
        delete node;
        --size_;
      }
    } else {
      if (--node->ref_ == 0) {
        delete node;
      }
    }
  }

  void evict() noexcept {
    ListNode* lru_node = lru_head_.next;
    node_type* node = lru_node_to_node(lru_node);
    node->in_cache_ = 0;
    hash_remove(node);
    lru_remove(node);
    if (--node->ref_ == 0) {
      delete node;
    }
    --size_;
  }

  node_pointer create_node_pointer(node_type* node) noexcept {
    auto deleter = [this](node_type* _node) {
      if (_node) {
        release(_node);
      }
    };
    return node_pointer(node, deleter);
  }

  node_pointer create_node_pointer(const key_type& key) {
    node_type* node = new node_type(key);
    return create_node_pointer(node);
  }

  void ref(node_type* node) noexcept {
    ++node->ref_;
    lru_remove(node);
    lru_append(&lru_head_, node);
  }

  bool hash_get(const key_type& key, size_type key_hash,
                ListNode** pp) noexcept {
    ListNode* hash_head = &hash_bucket_[key_hash & hash_bucket_mask_];
    ListNode* hash_node = hash_head->next;
    for (; hash_node != hash_head; hash_node = hash_node->next) {
      node_type* node = hash_node_to_node(hash_node);
      if (key_hash == node->key_hash_ && key_equal_(key, node->key_)) {
        *pp = hash_node;
        return true;
      }
    }

    *pp = hash_head;
    return false;
  }
};

}  // namespace deepx_core
