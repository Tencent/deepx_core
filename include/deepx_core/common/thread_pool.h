// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <condition_variable>
#include <forward_list>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#if !defined NDEBUG
#include <stdexcept>  // std::runtime_error
#endif

namespace deepx_core {

/************************************************************************/
/* ThreadPool */
/************************************************************************/
class ThreadPool {
 public:
  using function_t = std::function<void()>;
  struct wait_token_t {
    std::mutex mutex;
    std::condition_variable cond;
    int remain = 0;
  };

 private:
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  int started_ = 0;
  std::forward_list<function_t> tasks_;
  std::vector<std::thread> threads_;

 private:
  int started() const noexcept {
    std::unique_lock<std::mutex> guard(mutex_);
    return started_;
  }

 private:
  void worker_thread();

 public:
  ~ThreadPool();

  void start(int thread);
  void stop() noexcept;

  template <class Func>
  void emplace(Func&& func) {
#if !defined NDEBUG
    if (!started()) {
      throw std::runtime_error("emplace: the thread pool is not started.");
    }
#endif

    {
      std::unique_lock<std::mutex> guard(mutex_);
      tasks_.emplace_front(std::forward<function_t>(func));
    }
    cond_.notify_one();
  }

  template <class Func>
  void emplace_wait(Func&& func, wait_token_t* token) {
#if !defined NDEBUG
    if (!started()) {
      throw std::runtime_error("emplace_wait: the thread pool is not started.");
    }
#endif

    token->remain = 1;
    emplace([&func, token] {
      func();
      std::unique_lock<std::mutex> guard(token->mutex);
      if (--token->remain == 0) {
        token->cond.notify_all();
      }
    });

    std::unique_lock<std::mutex> guard(token->mutex);
    while (token->remain) {
      token->cond.wait(guard);
    }
  }

  void batch_emplace_wait(const std::vector<function_t>& funcs,
                          wait_token_t* token) {
#if !defined NDEBUG
    if (!started()) {
      throw std::runtime_error(
          "batch_emplace_wait: the thread pool is not started.");
    }
#endif

    token->remain = (int)funcs.size();
    for (const function_t& func : funcs) {
      emplace([&func, token] {
        func();
        std::unique_lock<std::mutex> guard(token->mutex);
        if (--token->remain == 0) {
          token->cond.notify_all();
        }
      });
    }

    std::unique_lock<std::mutex> guard(token->mutex);
    while (token->remain) {
      token->cond.wait(guard);
    }
  }
};

}  // namespace deepx_core
