// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/ps/file_dispatcher.h>
#include <algorithm>  // std::reverse, std::shuffle

namespace deepx_core {

void FileDispatcher::PreTrain(const std::vector<std::string>& files) {
  std::unique_lock<std::mutex> guard(mutex_);
  DXASSERT(!files.empty());
  files_ = files;
  if (reverse_) {
    std::reverse(files_.begin(), files_.end());
  }
}

void FileDispatcher::PreEpoch() {
  std::unique_lock<std::mutex> guard(mutex_);
  DXASSERT(!files_.empty());
  DXASSERT(to_dispatch_.empty());
  DXASSERT(dispatch_time_.empty());
  if (shuffle_) {
    std::shuffle(files_.begin(), files_.end(), engine_);
  }
  to_dispatch_.assign(files_.begin(), files_.end());
  finished_.clear();
}

bool FileDispatcher::WorkerDispatchFile(std::string* file) {
  std::unique_lock<std::mutex> guard(mutex_);
  for (;;) {
    if (!to_dispatch_.empty()) {
      *file = to_dispatch_.front();
      to_dispatch_.pop_front();
      dispatch_time_[*file] = time(nullptr);
      DXINFO("File is dispatched: %s.", file->c_str());
      return true;
    }

    if (timeout_ > 0) {
      time_t now = time(nullptr);
      for (const auto& entry : dispatch_time_) {
        if (now - entry.second > (time_t)timeout_) {
          const std::string& _file = entry.first;
          to_dispatch_.emplace_back(_file);
          dispatch_time_.erase(_file);
          DXERROR("File timed out: %s.", _file.c_str());
          break;
        }
      }
    }

    if (to_dispatch_.empty()) {
      DXINFO("No file to dispatch.");
      return false;
    }
  }
}

bool FileDispatcher::WorkerFinishFile(const std::string& file) {
  std::unique_lock<std::mutex> guard(mutex_);
  DXINFO("File is finished: %s.", file.c_str());
  finished_.emplace_back(file);
  dispatch_time_.erase(file);
  if (finished_.size() >= files_.size() && to_dispatch_.empty() &&
      dispatch_time_.empty()) {
    DXINFO("Epoch is finished.");
    return true;
  }
  return false;
}

void FileDispatcher::WorkerFailureFile(const std::string& file) {
  std::unique_lock<std::mutex> guard(mutex_);
  DXERROR("File is failed: %s.", file.c_str());
  if (dispatch_time_.count(file) > 0) {
    to_dispatch_.emplace_back(file);
    dispatch_time_.erase(file);
  }
}

}  // namespace deepx_core
