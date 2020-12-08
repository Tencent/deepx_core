// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <chrono>
#include <ctime>

namespace deepx_core {

LogTime GetLogTime() noexcept {
  auto now = std::chrono::system_clock::now();
  time_t time = std::chrono::system_clock::to_time_t(now);
  const auto* tm = localtime(&time);
  auto microsecond = std::chrono::duration_cast<std::chrono::microseconds>(
      now.time_since_epoch());
  LogTime log_time;
  log_time.year = tm->tm_year + 1900;
  log_time.month = tm->tm_mon + 1;
  log_time.day = tm->tm_mday;
  log_time.hour = tm->tm_hour;
  log_time.minute = tm->tm_min;
  log_time.second = tm->tm_sec;
  log_time.microsecond = (int)(microsecond.count() % 1000000);
  return log_time;
}

}  // namespace deepx_core
