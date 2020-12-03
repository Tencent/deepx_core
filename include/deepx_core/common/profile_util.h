// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <chrono>

namespace deepx_core {

/************************************************************************/
/* TimerGuard */
/************************************************************************/
template <typename Duration>
class TimerGuard {
 private:
  double& accumulator_;
  std::chrono::time_point<std::chrono::steady_clock> begin_;

 public:
  explicit TimerGuard(double& accumulator)  // NOLINT
      : accumulator_(accumulator) {
    begin_ = std::chrono::steady_clock::now();
  }

  ~TimerGuard() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now - begin_;
    accumulator_ += std::chrono::duration_cast<Duration>(duration).count();
  }
};

using SecondTimerGuard = TimerGuard<std::chrono::seconds>;
using MillisecondTimerGuard = TimerGuard<std::chrono::milliseconds>;
using MicrosecondTimerGuard = TimerGuard<std::chrono::microseconds>;
using NanosecondTimerGuard = TimerGuard<std::chrono::nanoseconds>;

}  // namespace deepx_core
