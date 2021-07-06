// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/memcpy.h>
#include "benchmark.h"

void Test(int n) {
  std::default_random_engine engine;
  std::vector<char> x(n);
  std::vector<char> y1(n);
  std::vector<char> y2(n);
  rand(engine, &x);

  memcpy(y1.data(), x.data(), n);

  sage2_memcpy(y2.data(), x.data(), n);
  CHECK_EQUAL(y1, y2);
}

template <class Func>
double Benchmark(Func&& func, int n, int m) {
  std::default_random_engine engine;
  constexpr size_t size = 128 * 1024 * 1024;
  size_t offset = 0;
  std::vector<char> x(size + n, 1);
  std::vector<char> y(size + n, 2);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(y.data() + offset, x.data() + offset, n);
    offset += n;
    offset &= (size - 1);
  }
  EndTimer();
  return 1e-9 * m * n / GetSeconds();
}

void Benchmark(int n) {
  int m = 200000000 / n;
  double gbps[2] = {0};
  gbps[0] = Benchmark(memcpy, n, m);
  gbps[1] = Benchmark(sage2_memcpy, n, m);
  PrintContent(n, gbps);
}

int main(int argc, char** argv) {
  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    for (int i = 1; i < 1000; ++i) {
      Test(i);
    }
  }

  if (action & 2) {
    PrintHeader1(2, "memcpy");
    PrintHeader2(2, "GB/s");
    PrintHeader3("libc", "opt");
    for (int n : GetLargeN()) {
      Benchmark(n);
    }
  }
  return 0;
}
