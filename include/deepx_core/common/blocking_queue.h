// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <condition_variable>
#include <list>
#include <mutex>
#include <utility>
#if !defined NDEBUG
#include <stdexcept>  // std::runtime_error
#endif

namespace deepx_core {

/************************************************************************/
/* BlockingQueue */
/************************************************************************/
template <typename T>
class BlockingQueue {
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_reference = const value_type&;

 private:
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  int started_ = 0;
  std::list<value_type> queue_;

 public:
  ~BlockingQueue() { stop(); }

  size_t size() const {
    std::unique_lock<std::mutex> guard(mutex_);
    return queue_.size();
  }

  bool empty() const {
    std::unique_lock<std::mutex> guard(mutex_);
    return queue_.empty();
  }

  // Start the blocking queue.
  void start() {
    std::unique_lock<std::mutex> guard(mutex_);
    if (!started_) {
      started_ = 1;
    }
  }

  // Stop the blocking queue.
  void stop() {
    std::unique_lock<std::mutex> guard(mutex_);
    if (started_) {
      started_ = 0;
      cond_.notify_all();
    }
  }

  // Push an element constructed by 'args' to the blocking queue.
  //
  // The blocking queue must be started.
  template <typename... Args>
  void push(Args&&... args) {
    std::unique_lock<std::mutex> guard(mutex_);
#if !defined NDEBUG
    if (!started_) {
      throw std::runtime_error("push: the blocking queue is not started.");
    }
#endif
    queue_.emplace_back(std::forward<Args>(args)...);
    cond_.notify_one();
  }

  // Pop an element from the blocking queue in blocking mode.
  //
  // Return true, 'v' is got.
  // Return false, the blocking queue is stopped and empty.
  bool pop(pointer v) {
    std::unique_lock<std::mutex> guard(mutex_);
    while (queue_.empty()) {
      if (!started_) {
        return false;
      }
      cond_.wait(guard);
    }

    if (!queue_.empty()) {
      *v = std::move(queue_.front());
      queue_.pop_front();
      return true;
    }
    return false;
  }
};

}  // namespace deepx_core
