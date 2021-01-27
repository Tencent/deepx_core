// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/thread_pool.h>
#include <utility>
#if !defined NDEBUG
#include <stdexcept>  // std::runtime_error
#endif

namespace deepx_core {

int ThreadPool::started() const {
  std::unique_lock<std::mutex> guard(mutex_);
  return started_;
}

void ThreadPool::worker_thread() {
  function_t func;
  for (;;) {
    std::unique_lock<std::mutex> guard(mutex_);
    while (started_ && tasks_.empty()) {
      cond_.wait(guard);
    }

    if (!started_) {
      break;
    }

    func = std::move(tasks_.front());
    tasks_.pop_front();
    guard.unlock();
    func();
  }

  std::unique_lock<std::mutex> guard(mutex_);
  while (!tasks_.empty()) {
    func = std::move(tasks_.front());
    tasks_.pop_front();
    guard.unlock();
    func();
    guard.lock();
  }
}

ThreadPool::~ThreadPool() { stop(); }

void ThreadPool::start(int n) {
  std::unique_lock<std::mutex> guard(mutex_);
  if (!started_) {
    started_ = 1;
    for (int i = 0; i < n; ++i) {
      threads_.emplace_back([this]() { worker_thread(); });
    }
  }
}

void ThreadPool::stop() {
  {
    std::unique_lock<std::mutex> guard(mutex_);
    if (started_) {
      started_ = 0;
    }
    cond_.notify_all();
  }

  if (!threads_.empty()) {
    for (std::thread& thread : threads_) {
      thread.join();
    }
    threads_.clear();
  }
}

void ThreadPool::post(function_t func) {
#if !defined NDEBUG
  if (!started()) {
    throw std::runtime_error("post: the thread pool is not started.");
  }
#endif

  {
    std::unique_lock<std::mutex> guard(mutex_);
    tasks_.emplace_front(std::move(func));
  }
  cond_.notify_one();
}

void ThreadPool::run(const function_t& func, wait_token_t* token) {
#if !defined NDEBUG
  if (!started()) {
    throw std::runtime_error("run: the thread pool is not started.");
  }
#endif

  token->remain = 1;
  post([&func, token] {
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

void ThreadPool::run(const std::vector<function_t>& funcs,
                     wait_token_t* token) {
#if !defined NDEBUG
  if (!started()) {
    throw std::runtime_error("run: the thread pool is not started.");
  }
#endif

  token->remain = (int)funcs.size();
  for (const function_t& func : funcs) {
    post([&func, token] {
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

}  // namespace deepx_core
