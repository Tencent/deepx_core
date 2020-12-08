// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/instance/base.h>

namespace deepx_core {

/************************************************************************/
/* LibsvmInstanceReaderHelper */
/************************************************************************/
template <typename T, typename I>
class LibsvmInstanceReaderHelper : public InstanceReaderHelper<T, I> {
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

 public:
  explicit LibsvmInstanceReaderHelper(const std::string& line) : base_t(line) {}

 public:
  bool Parse(csr_t* X, tsr_t* Y, tsr_t* W, tsrs_t* uuid);
  void ParseNoFailure(csr_t* X, tsr_t* Y, tsr_t* W, tsrs_t* uuid);
};

template <typename T, typename I>
bool LibsvmInstanceReaderHelper<T, I>::Parse(csr_t* X, tsr_t* Y, tsr_t* W,
                                             tsrs_t* uuid) {
  int batch = Y->dim(0);
  int label_size = Y->dim(1);
  DXCHECK_THROW(label_size <= MAX_LABEL_SIZE);
  size_t X_value_size = X->value_size();
  float_t* _Y;
  float_t* _W = nullptr;
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

    if (!ParseFeatures(X)) {
      break;
    }

    X->add_row();
    return true;
  } while (0);  // NOLINT

  X->trim(X_value_size);
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
void LibsvmInstanceReaderHelper<T, I>::ParseNoFailure(csr_t* X, tsr_t* Y,
                                                      tsr_t* W, tsrs_t* uuid) {
  if (!Parse(X, Y, W, uuid)) {
    int batch = Y->dim(0);
    int label_size = Y->dim(1);
    X->add_row();
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
