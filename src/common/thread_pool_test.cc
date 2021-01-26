// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/thread_pool.h>
#include <gtest/gtest.h>
#include <atomic>

namespace deepx_core {

class ThreadPoolTest : public testing::Test {
 protected:
  const int N = 1000;
};

TEST_F(ThreadPoolTest, post) {
  ThreadPool thread_pool;
  std::atomic<int> sum(0);
  thread_pool.start(4);
  for (int i = 0; i < N; ++i) {
    thread_pool.post([&sum, i]() { sum += i; });
  }
  thread_pool.stop();
  EXPECT_EQ(sum, N * (N - 1) / 2);
}

TEST_F(ThreadPoolTest, run_1) {
  ThreadPool thread_pool;
  ThreadPool::wait_token_t token;
  int sum = 0;
  thread_pool.start(4);
  for (int i = 0; i < N; ++i) {
    thread_pool.run([&sum, i]() { sum += i; }, &token);
  }
  thread_pool.stop();
  EXPECT_EQ(sum, N * (N - 1) / 2);
}

TEST_F(ThreadPoolTest, run_2) {
  ThreadPool thread_pool;
  ThreadPool::wait_token_t token;
  std::atomic<int> sum(0);
  thread_pool.start(4);
  for (int i = 0; i < N; ++i) {
    thread_pool.run({[&sum, i]() { sum += i; }, [&sum, i]() { sum += i; }},
                    &token);
  }
  thread_pool.stop();
  EXPECT_EQ(sum, N * (N - 1));
}

}  // namespace deepx_core
