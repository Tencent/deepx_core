// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/blocking_queue.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace deepx_core {

class BlockingQueueTest : public testing::Test {
 protected:
  using bqi_t = BlockingQueue<int>;
  const int N = 100;
  const int M = 10000;
};

TEST_F(BlockingQueueTest, ProducerConsumer) {
  bqi_t consumer_queue, producer_queue;
  consumer_queue.start();
  producer_queue.start();
  for (int i = 0; i < N; ++i) {
    consumer_queue.push(0);
  }

  auto producer = [this, &consumer_queue, &producer_queue]() {
    int item;
    for (int i = 0; i < M; ++i) {
      (void)consumer_queue.pop(&item);
      item = i;
      producer_queue.push(item);
    }
    producer_queue.stop();
  };

  int sum = 0;
  auto consumer = [&sum, &consumer_queue, &producer_queue]() {
    int item;
    for (;;) {
      if (producer_queue.pop(&item)) {
        sum += item;
        item = -1;
        consumer_queue.push(item);
      } else {
        consumer_queue.stop();
        break;
      }
    }
  };

  std::vector<std::thread> threads;
  threads.emplace_back(producer);
  threads.emplace_back(consumer);
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(sum, M * (M - 1) / 2);
}

}  // namespace deepx_core
