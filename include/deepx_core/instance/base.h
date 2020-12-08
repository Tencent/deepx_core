// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// include all headers needed by instance reader helpers
#include <deepx_core/common/fast_strtox.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/tensor/csr_matrix.h>
#include <deepx_core/tensor/tensor.h>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace deepx_core {

const std::string UUID_TAG = "uuid:";

/************************************************************************/
/* InstanceReaderHelper */
/************************************************************************/
template <typename T, typename I>
class InstanceReaderHelper {
 public:
  using float_t = T;
  using int_t = I;
  using tsr_t = Tensor<float_t>;
  using csr_t = CSRMatrix<float_t, int_t>;
  using tsrs_t = Tensor<std::string>;
  static constexpr float_t MAX_INSTANCE_LABEL = 10000;
  static constexpr float_t MAX_INSTANCE_WEIGHT = 10000;
  static constexpr float_t MAX_FEATURE_VALUE = 100;
  static constexpr int MAX_X_SIZE = 128;
  static constexpr int MAX_LABEL_SIZE = 32;

 protected:
  const std::string& line_;
  const char* s_ = nullptr;
  const char* line_end_ = nullptr;
  float_t label_ = 0;
  float_t weight_ = 0;
  std::string uuid_;

 public:
  explicit InstanceReaderHelper(const std::string& line) noexcept
      : line_(line), s_(line.data()), line_end_(s_ + line.size()) {}

 protected:
  static bool IsSpace(char c) noexcept { return c == ' ' || c == '\t'; }

  void SkipSpace() noexcept {
    while (s_ < line_end_ && IsSpace(*s_)) {
      ++s_;
    }
  }

  void SkipNonSpace() noexcept {
    while (s_ < line_end_ && !IsSpace(*s_)) {
      ++s_;
    }
  }

 protected:
  bool ParseLabelWeight() noexcept;
  bool ParseUUID() noexcept;
  bool ParseFeatures(csr_t* X);
  bool ParseFeatures(csr_t* X, char sep);
};

template <typename T, typename I>
bool InstanceReaderHelper<T, I>::ParseLabelWeight() noexcept {
  char* end;

  SkipSpace();
  if (s_ >= line_end_) {
    DXERROR("Missing label: %s.", line_.c_str());
    return false;
  }

  // label
  label_ = (float_t)fast_strtod(s_, &end);
  if (s_ == end) {
    DXERROR("Invalid label: %s.", line_.c_str());
    return false;
  }

  if (label_ > MAX_INSTANCE_LABEL || label_ < -MAX_INSTANCE_LABEL) {
    DXERROR("Too large or small label: %s.", line_.c_str());
    return false;
  }

  if (*end != ':') {
    if (end < line_end_ && !IsSpace(*end)) {
      DXERROR("Invalid character after label: %s.", line_.c_str());
      return false;
    }

    // no weight
    weight_ = 1;
  } else {
    // weight
    s_ = end + 1;

    weight_ = (float_t)fast_strtod(s_, &end);
    if (s_ == end) {
      DXERROR("Invalid weight: %s.", line_.c_str());
      return false;
    }

    if (weight_ <= 0) {
      DXERROR("Non-positive weight: %s.", line_.c_str());
      return false;
    }

    if (weight_ > MAX_INSTANCE_WEIGHT) {
      DXERROR("Too large weight: %s.", line_.c_str());
      return false;
    }
  }

  s_ = end + 1;
  return true;
}

template <typename T, typename I>
bool InstanceReaderHelper<T, I>::ParseUUID() noexcept {
  uuid_.clear();

  SkipSpace();
  if (s_ >= line_end_) {
    return true;
  }

  if (strncmp(s_, UUID_TAG.c_str(), UUID_TAG.size()) != 0) {
    return true;
  }
  s_ += UUID_TAG.size();

  // skip to the end of uuid
  const char* begin = s_;
  SkipNonSpace();
  if (s_ == begin) {
    DXERROR("Invalid uuid: %s.", line_.c_str());
    return false;
  }

  uuid_.assign(begin, s_);
  return true;
}

template <typename T, typename I>
bool InstanceReaderHelper<T, I>::ParseFeatures(csr_t* X) {
  return ParseFeatures(X, ' ');
}

template <typename T, typename I>
bool InstanceReaderHelper<T, I>::ParseFeatures(csr_t* X, char sep) {
  char* end;
  int_t feature_id;
  float_t feature_value;

  for (;;) {
    if (s_ >= line_end_) {
      break;
    }

    // feature id
    feature_id = fast_strtoi<int_t>(s_, &end);
    if (s_ == end) {
      DXERROR("Invalid feature id: %s.", line_.c_str());
      return false;
    }

    if (*end != ':') {
      if (sep != ' ') {
        if (end < line_end_ && !IsSpace(*end) && *end != sep) {
          DXERROR("Invalid character after feature id: %s.", line_.c_str());
          return false;
        }
      } else {
        if (end < line_end_ && !IsSpace(*end)) {
          DXERROR("Invalid character after feature id: %s.", line_.c_str());
          return false;
        }
      }

      // no feature value
      feature_value = 1;
    } else {
      // feature value
      s_ = end + 1;

      feature_value = (float_t)fast_strtod(s_, &end);
      if (s_ == end) {
        DXERROR("Invalid feature value: %s.", line_.c_str());
        return false;
      }

      if (feature_value > MAX_FEATURE_VALUE ||
          feature_value < -MAX_FEATURE_VALUE) {
        DXERROR("Too large or small feature value: %s.", line_.c_str());
        return false;
      }
    }

    s_ = end + 1;
    X->emplace(feature_id, feature_value);

    if (sep != ' ') {
      if (*end == sep) {
        break;
      }

      SkipSpace();
      if (s_ >= line_end_) {
        break;
      }

      if (*s_ == sep) {
        s_ = s_ + 1;
        break;
      }
    }
  }
  return true;
}

}  // namespace deepx_core
