// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/array_view.h>
#include <deepx_core/common/stream.h>
#include <string>

namespace deepx_core {

/************************************************************************/
/* DIST_MESSAGE_TYPE */
/************************************************************************/
enum DIST_MESSAGE_TYPE {
  DIST_MESSAGE_TYPE_NONE = 0,
  DIST_MESSAGE_TYPE_ECHO_REQUEST = 1,
  DIST_MESSAGE_TYPE_ECHO_RESPONSE = 2,
  DIST_MESSAGE_TYPE_HEART_BEAT_NOTIFY = 3,

  // 'WK' means worker.
  // 'PS' means param server.
  // 'CS' means coord server.
  //
  // WK sends this message to CS to get a file.
  DIST_MESSAGE_TYPE_FILE_REQUEST = 11,
  DIST_MESSAGE_TYPE_FILE_RESPONSE = 12,
  // WK sends this message to CS after the WK has processed the file.
  DIST_MESSAGE_TYPE_FILE_FINISH_NOTIFY = 13,
  // WK sends this message to PS to pull model param.
  DIST_MESSAGE_TYPE_PULL_REQUEST = 14,
  DIST_MESSAGE_TYPE_PULL_RESPONSE = 15,
  // WK sends this message to PS to push model grad.
  DIST_MESSAGE_TYPE_PUSH_NOTIFY = 16,
  // CS sends this message to PS to save model.
  DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST = 17,
  DIST_MESSAGE_TYPE_MODEL_SAVE_RESPONSE = 18,
  // CS sends this message to PS to notify the termination of training.
  DIST_MESSAGE_TYPE_TERMINATION_NOTIFY = 19,

  DIST_MESSAGE_TYPE_USER_REQUEST = 31,
  DIST_MESSAGE_TYPE_USER_RESPONSE = 32,
  DIST_MESSAGE_TYPE_USER_NOTIFY = 33,
};

/************************************************************************/
/* DistMessage */
/************************************************************************/
class DistMessage {
 public:
  struct EchoRequest {
    std::string buf;
  };
  struct EchoResponse {
    std::string buf;
  };
  struct FileResponse {
    int epoch = 0;
    std::string file;
  };
  struct FileFinishNotify {
    std::string file;
    double loss = 0;
    double loss_weight = 0;
  };
  struct PullRequest {
    std::string buf;
  };
  struct PullResponse {
    std::string buf;
  };
  struct PushNotify {
    std::string buf;
  };
  struct ModelSaveRequest {
    int epoch = 0;
    std::string timestamp;
    int feature_kv_protocol_version = 0;
  };
  struct UserRequest {
    std::string buf;
  };
  struct UserResponse {
    std::string buf;
  };
  struct UserNotify {
    std::string buf;
  };

 private:
  int type_ = DIST_MESSAGE_TYPE_NONE;
  EchoRequest echo_request_;
  EchoResponse echo_response_;
  FileResponse file_response_;
  FileFinishNotify file_finish_notify_;
  PullRequest pull_request_;
  PullResponse pull_response_;
  PushNotify push_notify_;
  ModelSaveRequest model_save_request_;
  UserRequest user_request_;
  UserResponse user_response_;
  UserNotify user_notify_;

 public:
  void set_type(int type) noexcept { type_ = type; }
  int type() const noexcept { return type_; }

  EchoRequest* mutable_echo_request() noexcept { return &echo_request_; }
  const EchoRequest& echo_request() const noexcept { return echo_request_; }

  EchoResponse* mutable_echo_response() noexcept { return &echo_response_; }
  const EchoResponse& echo_response() const noexcept { return echo_response_; }

  FileResponse* mutable_file_response() noexcept { return &file_response_; }
  const FileResponse& file_response() const noexcept { return file_response_; }

  FileFinishNotify* mutable_file_finish_notify() noexcept {
    return &file_finish_notify_;
  }
  const FileFinishNotify& file_finish_notify() const noexcept {
    return file_finish_notify_;
  }

  PullRequest* mutable_pull_request() noexcept { return &pull_request_; }
  const PullRequest& pull_request() const noexcept { return pull_request_; }

  PullResponse* mutable_pull_response() noexcept { return &pull_response_; }
  const PullResponse& pull_response() const noexcept { return pull_response_; }

  PushNotify* mutable_push_notify() noexcept { return &push_notify_; }
  const PushNotify& push_notify() const noexcept { return push_notify_; }

  ModelSaveRequest* mutable_model_save_request() noexcept {
    return &model_save_request_;
  }
  const ModelSaveRequest& model_save_request() const noexcept {
    return model_save_request_;
  }

  UserRequest* mutable_user_request() noexcept { return &user_request_; }
  const UserRequest& user_request() const noexcept { return user_request_; }

  UserResponse* mutable_user_response() noexcept { return &user_response_; }
  const UserResponse& user_response() const noexcept { return user_response_; }

  UserNotify* mutable_user_notify() noexcept { return &user_notify_; }
  const UserNotify& user_notify() const noexcept { return user_notify_; }

 public:
  bool HasResponse() const noexcept;
  static bool HasResponse(int type) noexcept;
};

OutputStringStream& operator<<(OutputStringStream& os,
                               const DistMessage& message);

/************************************************************************/
/* DistMessageView */
/************************************************************************/
class DistMessageView {
 public:
  struct EchoRequest {
    const_string_view buf;
  };
  struct EchoResponse {
    const_string_view buf;
  };
  struct FileResponse {
    int epoch = 0;
    std::string file;
  };
  struct FileFinishNotify {
    std::string file;
    double loss = 0;
    double loss_weight = 0;
  };
  struct PullRequest {
    const_string_view buf;
  };
  struct PullResponse {
    const_string_view buf;
  };
  struct PushNotify {
    const_string_view buf;
  };
  struct ModelSaveRequest {
    int epoch = 0;
    std::string timestamp;
    int feature_kv_protocol_version = 0;
  };
  struct UserRequest {
    const_string_view buf;
  };
  struct UserResponse {
    const_string_view buf;
  };
  struct UserNotify {
    const_string_view buf;
  };

 private:
  int type_ = DIST_MESSAGE_TYPE_NONE;
  EchoRequest echo_request_;
  EchoResponse echo_response_;
  FileResponse file_response_;
  FileFinishNotify file_finish_notify_;
  PullRequest pull_request_;
  PullResponse pull_response_;
  PushNotify push_notify_;
  ModelSaveRequest model_save_request_;
  UserRequest user_request_;
  UserResponse user_response_;
  UserNotify user_notify_;

 public:
  void set_type(int type) noexcept { type_ = type; }
  int type() const noexcept { return type_; }

  EchoRequest* mutable_echo_request() noexcept { return &echo_request_; }
  const EchoRequest& echo_request() const noexcept { return echo_request_; }

  EchoResponse* mutable_echo_response() noexcept { return &echo_response_; }
  const EchoResponse& echo_response() const noexcept { return echo_response_; }

  FileResponse* mutable_file_response() noexcept { return &file_response_; }
  const FileResponse& file_response() const noexcept { return file_response_; }

  FileFinishNotify* mutable_file_finish_notify() noexcept {
    return &file_finish_notify_;
  }
  const FileFinishNotify& file_finish_notify() const noexcept {
    return file_finish_notify_;
  }

  PullRequest* mutable_pull_request() noexcept { return &pull_request_; }
  const PullRequest& pull_request() const noexcept { return pull_request_; }

  PullResponse* mutable_pull_response() noexcept { return &pull_response_; }
  const PullResponse& pull_response() const noexcept { return pull_response_; }

  PushNotify* mutable_push_notify() noexcept { return &push_notify_; }
  const PushNotify& push_notify() const noexcept { return push_notify_; }

  ModelSaveRequest* mutable_model_save_request() noexcept {
    return &model_save_request_;
  }
  const ModelSaveRequest& model_save_request() const noexcept {
    return model_save_request_;
  }

  UserRequest* mutable_user_request() noexcept { return &user_request_; }
  const UserRequest& user_request() const noexcept { return user_request_; }

  UserResponse* mutable_user_response() noexcept { return &user_response_; }
  const UserResponse& user_response() const noexcept { return user_response_; }

  UserNotify* mutable_user_notify() noexcept { return &user_notify_; }
  const UserNotify& user_notify() const noexcept { return user_notify_; }

 public:
  bool HasResponse() const noexcept;
  static bool HasResponse(int type) noexcept;
};

InputStringStream& ReadView(InputStringStream& is,      // NOLINT
                            DistMessageView& message);  // NOLINT

}  // namespace deepx_core
