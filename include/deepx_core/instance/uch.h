// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/instance/base.h>

namespace deepx_core {

/************************************************************************/
/* UCHInstanceReaderHelper */
/************************************************************************/
template <typename T, typename I>
class UCHInstanceReaderHelper : public InstanceReaderHelper<T, I> {
 public:
  using base_t = InstanceReaderHelper<T, I>;
  using float_t = typename base_t::float_t;
  using int_t = typename base_t::int_t;
  using tsr_t = typename base_t::tsr_t;
  using csr_t = typename base_t::csr_t;
  using tsrs_t = typename base_t::tsrs_t;
  using base_t::MAX_FEATURE_VALUE;
  using base_t::MAX_INSTANCE_LABEL;
  using base_t::MAX_INSTANCE_WEIGHT;
  using base_t::MAX_LABEL_SIZE;
  using base_t::MAX_X_SIZE;

 protected:
  using base_t::label_;
  using base_t::line_;
  using base_t::line_end_;
  using base_t::s_;
  using base_t::uuid_;
  using base_t::weight_;

 protected:
  using base_t::ParseFeatures;
  using base_t::ParseLabelWeight;
  using base_t::ParseUUID;
  using base_t::SkipSpace;

 public:
  explicit UCHInstanceReaderHelper(const std::string& line) noexcept
      : base_t(line) {}

 protected:
  bool ParseXUserFeatures(csr_t* X);
  bool ParseXCandFeatures(csr_t* X);
  bool ParseXHistFeatures(csr_t* X);

 public:
  bool Parse(csr_t* X_user, csr_t* X_cand, std::vector<csr_t*>* X_hist,
             tsr_t* X_hist_size, tsr_t* Y, tsr_t* W, tsrs_t* uuid);
  void ParseNoFailure(csr_t* X_user, csr_t* X_cand, std::vector<csr_t*>* X_hist,
                      tsr_t* X_hist_size, tsr_t* Y, tsr_t* W, tsrs_t* uuid);
};

template <typename T, typename I>
bool UCHInstanceReaderHelper<T, I>::ParseXUserFeatures(csr_t* X) {
  SkipSpace();
  if (s_ >= line_end_) {
    DXERROR("Missing user feature: %s.", line_.c_str());
    return false;
  }
  return ParseFeatures(X, '|');
}

template <typename T, typename I>
bool UCHInstanceReaderHelper<T, I>::ParseXCandFeatures(csr_t* X) {
  SkipSpace();
  if (s_ >= line_end_) {
    DXERROR("Missing candidate feature: %s.", line_.c_str());
    return false;
  }
  return ParseFeatures(X, '|');
}

template <typename T, typename I>
bool UCHInstanceReaderHelper<T, I>::ParseXHistFeatures(csr_t* X) {
  SkipSpace();
  if (s_ >= line_end_) {
    return false;
  }
  return ParseFeatures(X, '|');
}

template <typename T, typename I>
bool UCHInstanceReaderHelper<T, I>::Parse(csr_t* X_user, csr_t* X_cand,
                                          std::vector<csr_t*>* X_hist,
                                          tsr_t* X_hist_size, tsr_t* Y,
                                          tsr_t* W, tsrs_t* uuid) {
  int _X_hist_size = (int)X_hist->size();
  int batch = Y->dim(0);
  int label_size = Y->dim(1);
  DXCHECK_THROW(_X_hist_size <= MAX_X_SIZE);
  DXCHECK_THROW(label_size <= MAX_LABEL_SIZE);
  size_t X_user_value_size = X_user->value_size();
  size_t X_cand_value_size = X_cand->value_size();
  size_t X_hist_value_size[MAX_X_SIZE];
  float_t* _Y;
  float_t* _W = nullptr;
  for (int i = 0; i < _X_hist_size; ++i) {
    X_hist_value_size[i] = ((*X_hist)[i])->value_size();
  }
  X_hist_size->resize(batch + 1);
  Y->resize(batch + 1, label_size);
  _Y = &Y->data(batch * label_size);
  if (W) {
    W->resize(batch + 1, label_size);
    _W = &W->data(batch * label_size);
  }
  if (uuid) {
    uuid->resize(batch + 1);
  }

  do {
    int i;
    for (i = 0; i < label_size; ++i) {
      if (!(ParseLabelWeight())) {
        break;
      }
      _Y[i] = label_;
      if (_W) {
        _W[i] = weight_;
      }
    }
    if (i != label_size) {
      break;
    }

    if (!ParseUUID()) {
      break;
    }
    if (uuid) {
      uuid->data(batch) = std::move(uuid_);
    }

    if (!ParseXUserFeatures(X_user)) {
      break;
    }
    if (!ParseXCandFeatures(X_cand)) {
      break;
    }
    for (i = 0; i < _X_hist_size; ++i) {
      if (!ParseXHistFeatures((*X_hist)[i])) {
        break;
      }
    }
    X_hist_size->data(batch) = (float_t)i;

    X_user->add_row();
    X_cand->add_row();
    for (i = 0; i < _X_hist_size; ++i) {
      ((*X_hist)[i])->add_row();
    }
    return true;
  } while (0);  // NOLINT

  X_user->trim(X_user_value_size);
  X_cand->trim(X_cand_value_size);
  for (int i = 0; i < _X_hist_size; ++i) {
    ((*X_hist)[i])->trim(X_hist_value_size[i]);
  }
  X_hist_size->resize(batch);
  Y->resize(batch, label_size);
  if (W) {
    W->resize(batch, label_size);
  }
  if (uuid) {
    uuid->resize(batch);
  }
  return false;
}

template <typename T, typename I>
void UCHInstanceReaderHelper<T, I>::ParseNoFailure(csr_t* X_user, csr_t* X_cand,
                                                   std::vector<csr_t*>* X_hist,
                                                   tsr_t* X_hist_size, tsr_t* Y,
                                                   tsr_t* W, tsrs_t* uuid) {
  if (!Parse(X_user, X_cand, X_hist, X_hist_size, Y, W, uuid)) {
    int _X_hist_size = (int)X_hist->size();
    int batch = Y->dim(0);
    int label_size = Y->dim(1);
    X_user->add_row();
    X_cand->add_row();
    for (int i = 0; i < _X_hist_size; ++i) {
      ((*X_hist)[i])->add_row();
    }
    X_hist_size->resize(batch + 1);
    Y->resize(batch + 1, label_size);
    if (W) {
      W->resize(batch + 1, label_size);
    }
    if (uuid) {
      uuid->resize(batch + 1);
    }
  }
}

}  // namespace deepx_core
