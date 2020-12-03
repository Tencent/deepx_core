// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <condition_variable>
#include <mutex>

namespace deepx_core {

/************************************************************************/
/* ReadWriteLock */
/************************************************************************/
class ReadWriteLock {
 private:
  // > 0, 'status_' readers
  // -1, 1 writer
  // 0, 0 reader and 0 writer
  int status_ = 0;
  int waiting_writers_ = 0;
  std::mutex mutex_;
  std::condition_variable read_cond_;
  std::condition_variable write_cond_;

 public:
  ReadWriteLock() = default;
  ReadWriteLock(const ReadWriteLock&) = delete;
  ReadWriteLock& operator=(const ReadWriteLock&) = delete;

  void lock_read() {
    std::unique_lock<std::mutex> guard(mutex_);
    read_cond_.wait(guard,
                    [this]() { return waiting_writers_ == 0 && status_ >= 0; });
    status_ += 1;
  }

  void lock_write() {
    std::unique_lock<std::mutex> guard(mutex_);
    waiting_writers_ += 1;
    write_cond_.wait(guard, [this]() { return status_ == 0; });
    waiting_writers_ -= 1;
    status_ = -1;
  }

  void unlock() {
    std::unique_lock<std::mutex> guard(mutex_);
    if (status_ == -1) {
      status_ = 0;
    } else {
      status_ -= 1;
    }
    if (waiting_writers_ > 0) {
      if (status_ == 0) {
        write_cond_.notify_one();
      }
    } else {
      read_cond_.notify_all();
    }
  }
};

/************************************************************************/
/* ReadLockGuard */
/************************************************************************/
class ReadLockGuard {
 private:
  ReadWriteLock* const lock_;

 public:
  explicit ReadLockGuard(ReadWriteLock* lock) : lock_(lock) {
    lock_->lock_read();
  }
  explicit ReadLockGuard(ReadWriteLock& lock) : lock_(&lock) {  // NOLINT
    lock_->lock_read();
  }
  ~ReadLockGuard() { lock_->unlock(); }
  ReadLockGuard(const ReadLockGuard&) = delete;
  ReadLockGuard& operator=(const ReadLockGuard&) = delete;
};

/************************************************************************/
/* WriteLockGuard */
/************************************************************************/
class WriteLockGuard {
 private:
  ReadWriteLock* const lock_;

 public:
  explicit WriteLockGuard(ReadWriteLock* lock) : lock_(lock) {
    lock_->lock_write();
  }
  explicit WriteLockGuard(ReadWriteLock& lock) : lock_(&lock) {  // NOLINT
    lock_->lock_write();
  }
  ~WriteLockGuard() { lock_->unlock(); }
  WriteLockGuard(const WriteLockGuard&) = delete;
  WriteLockGuard& operator=(const WriteLockGuard&) = delete;
};

}  // namespace deepx_core
