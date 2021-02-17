// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <condition_variable>
#include <forward_list>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* ThreadPool */
/************************************************************************/
class ThreadPool {
 public:
  struct WaitToken {
    std::mutex mutex;
    std::condition_variable cond;
    int remain = 0;
  };
  using function_t = std::function<void()>;
  using wait_token_t = WaitToken;

 private:
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  int started_ = 0;
  std::forward_list<function_t> tasks_;
  std::vector<std::thread> threads_;

 private:
  int started() const;
  void worker_thread();

 public:
  ~ThreadPool();

  // Start 'n' worker threads.
  void start(int n);

  // Run all remaining function objects and stop all worker threads.
  void stop();

  // Post 'func' and return immediately.
  // 'func' will be run in a worker thread.
  //
  // The thread pool must be started.
  void post(function_t func);

  // Run 'func' in a worker thread and wait for the completion.
  //
  // The thread pool must be started.
  void run(const function_t& func, wait_token_t* token);

  // Run 'funcs' in worker threads and wait for the completion.
  //
  // The thread pool must be started.
  void run(const std::vector<function_t>& funcs, wait_token_t* token);
};

}  // namespace deepx_core
