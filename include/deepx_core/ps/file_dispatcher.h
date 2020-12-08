// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <ctime>
#include <list>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace deepx_core {

/************************************************************************/
/* FileDispatcher */
/************************************************************************/
class FileDispatcher {
 private:
  int reverse_ = 0;
  int shuffle_ = 0;
  int timeout_ = 0;
  std::mutex mutex_;
  std::default_random_engine engine_;
  std::vector<std::string> files_;
  std::list<std::string> to_dispatch_;
  std::vector<std::string> finished_;
  std::unordered_map<std::string, time_t> dispatch_time_;

 public:
  void set_reverse(int reverse) noexcept { reverse_ = reverse; }
  int reverse() const noexcept { return reverse_; }
  void set_shuffle(int shuffle) noexcept { shuffle_ = shuffle; }
  int shuffle() const noexcept { return shuffle_; }
  void set_timeout(int timeout) noexcept { timeout_ = timeout; }
  int timeout() const noexcept { return timeout_; }

 public:
  void PreTrain(const std::vector<std::string>& files);
  void PreEpoch();
  bool WorkerDispatchFile(std::string* file);
  bool WorkerFinishFile(const std::string& file);
  void WorkerFailureFile(const std::string& file);
};

}  // namespace deepx_core
