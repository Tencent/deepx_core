// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/array_view_io.h>
#include <deepx_core/ps/dist_message.h>

namespace deepx_core {

/************************************************************************/
/* DistMessage */
/************************************************************************/
bool DistMessage::HasResponse() const noexcept { return HasResponse(type_); }

bool DistMessage::HasResponse(int type) noexcept {
  switch (type) {
    case DIST_MESSAGE_TYPE_ECHO_REQUEST:
    case DIST_MESSAGE_TYPE_FILE_REQUEST:
    case DIST_MESSAGE_TYPE_PULL_REQUEST:
    case DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST:
    case DIST_MESSAGE_TYPE_USER_REQUEST:
      return true;
    default:
      return false;
  }
}

OutputStringStream& operator<<(OutputStringStream& os,
                               const DistMessage& message) {
  os.BeginMessage();
  os << message.type();
  switch (message.type()) {
    case DIST_MESSAGE_TYPE_ECHO_REQUEST:
      os << message.echo_request().buf;
      break;
    case DIST_MESSAGE_TYPE_ECHO_RESPONSE:
      os << message.echo_response().buf;
      break;
    case DIST_MESSAGE_TYPE_HEART_BEAT_NOTIFY:
      break;
    case DIST_MESSAGE_TYPE_FILE_REQUEST:
      break;
    case DIST_MESSAGE_TYPE_FILE_RESPONSE:
      os << message.file_response().epoch;
      os << message.file_response().file;
      break;
    case DIST_MESSAGE_TYPE_FILE_FINISH_NOTIFY:
      os << message.file_finish_notify().file;
      os << message.file_finish_notify().loss;
      os << message.file_finish_notify().loss_weight;
      break;
    case DIST_MESSAGE_TYPE_PULL_REQUEST:
      os << message.pull_request().buf;
      break;
    case DIST_MESSAGE_TYPE_PULL_RESPONSE:
      os << message.pull_response().buf;
      break;
    case DIST_MESSAGE_TYPE_PUSH_NOTIFY:
      os << message.push_notify().buf;
      break;
    case DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST:
      os << message.model_save_request().epoch;
      os << message.model_save_request().timestamp;
      os << message.model_save_request().feature_kv_protocol_version;
      break;
    case DIST_MESSAGE_TYPE_MODEL_SAVE_RESPONSE:
      break;
    case DIST_MESSAGE_TYPE_TERMINATION_NOTIFY:
      break;
    case DIST_MESSAGE_TYPE_USER_REQUEST:
      os << message.user_request().buf;
      break;
    case DIST_MESSAGE_TYPE_USER_RESPONSE:
      os << message.user_response().buf;
      break;
    case DIST_MESSAGE_TYPE_USER_NOTIFY:
      os << message.user_notify().buf;
      break;
  }
  os.EndMessage();
  return os;
}

/************************************************************************/
/* DistMessageView */
/************************************************************************/
bool DistMessageView::HasResponse() const noexcept {
  return HasResponse(type_);
}

bool DistMessageView::HasResponse(int type) noexcept {
  switch (type) {
    case DIST_MESSAGE_TYPE_ECHO_REQUEST:
    case DIST_MESSAGE_TYPE_FILE_REQUEST:
    case DIST_MESSAGE_TYPE_PULL_REQUEST:
    case DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST:
    case DIST_MESSAGE_TYPE_USER_REQUEST:
      return true;
    default:
      return false;
  }
}

InputStringStream& ReadView(InputStringStream& is, DistMessageView& message) {
  int place_holder, type;
  ReadView(is, place_holder);
  ReadView(is, type);
  if (!is) {
    return is;
  }

  message.set_type(type);
  switch (type) {
    case DIST_MESSAGE_TYPE_ECHO_REQUEST:
      ReadView(is, message.mutable_echo_request()->buf);
      break;
    case DIST_MESSAGE_TYPE_ECHO_RESPONSE:
      ReadView(is, message.mutable_echo_response()->buf);
      break;
    case DIST_MESSAGE_TYPE_HEART_BEAT_NOTIFY:
      break;
    case DIST_MESSAGE_TYPE_FILE_REQUEST:
      break;
    case DIST_MESSAGE_TYPE_FILE_RESPONSE:
      ReadView(is, message.mutable_file_response()->epoch);
      ReadView(is, message.mutable_file_response()->file);
      break;
    case DIST_MESSAGE_TYPE_FILE_FINISH_NOTIFY:
      ReadView(is, message.mutable_file_finish_notify()->file);
      ReadView(is, message.mutable_file_finish_notify()->loss);
      ReadView(is, message.mutable_file_finish_notify()->loss_weight);
      break;
    case DIST_MESSAGE_TYPE_PULL_REQUEST:
      ReadView(is, message.mutable_pull_request()->buf);
      break;
    case DIST_MESSAGE_TYPE_PULL_RESPONSE:
      ReadView(is, message.mutable_pull_response()->buf);
      break;
    case DIST_MESSAGE_TYPE_PUSH_NOTIFY:
      ReadView(is, message.mutable_push_notify()->buf);
      break;
    case DIST_MESSAGE_TYPE_MODEL_SAVE_REQUEST:
      ReadView(is, message.mutable_model_save_request()->epoch);
      ReadView(is, message.mutable_model_save_request()->timestamp);
      ReadView(
          is,
          message.mutable_model_save_request()->feature_kv_protocol_version);
      break;
    case DIST_MESSAGE_TYPE_MODEL_SAVE_RESPONSE:
      break;
    case DIST_MESSAGE_TYPE_TERMINATION_NOTIFY:
      break;
    case DIST_MESSAGE_TYPE_USER_REQUEST:
      ReadView(is, message.mutable_user_request()->buf);
      break;
    case DIST_MESSAGE_TYPE_USER_RESPONSE:
      ReadView(is, message.mutable_user_response()->buf);
      break;
    case DIST_MESSAGE_TYPE_USER_NOTIFY:
      ReadView(is, message.mutable_user_notify()->buf);
      break;
  }
  return is;
}

}  // namespace deepx_core
