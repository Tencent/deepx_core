// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/thread_pool.h>

namespace deepx_core {

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

void ThreadPool::start(int thread) {
  std::unique_lock<std::mutex> guard(mutex_);
  if (!started_) {
    started_ = 1;
    for (int i = 0; i < thread; ++i) {
      threads_.emplace_back([this]() { worker_thread(); });
    }
  }
}

void ThreadPool::stop() noexcept {
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

}  // namespace deepx_core
