// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/tensor/data_type.h>
#include <iostream>
#include <random>
#include <utility>

namespace deepx_core {

/************************************************************************/
/* TensorMap */
/************************************************************************/
class TensorMap : public AnyMap, public DataType {
 private:
  void _Write(OutputStream& os) const;    // NOLINT
  void _Read(InputStream& is);            // NOLINT
  void _ReadView(InputStringStream& is);  // NOLINT
  void _WriteText(std::ostream& os) const;

  friend OutputStream& operator<<(OutputStream& os,
                                  const TensorMap& tensor_map);
  friend InputStream& operator>>(InputStream& is, TensorMap& tensor_map);
  friend InputStringStream& ReadView(InputStringStream& is,   // NOLINT
                                     TensorMap& tensor_map);  // NOLINT
  friend std::ostream& operator<<(std::ostream& os,
                                  const TensorMap& tensor_map);

 public:
  // Call 'zeros' for value type 'srm_t', so that its shape is preserved.
  void ClearSRMValue() noexcept;
  // Call 'clear' for value type 'tsr_t', 'csr_t', 'tsri_t', 'tsrs_t'.
  // Call 'zeros' for value type 'srm_t', so that its shape is preserved.
  void ClearValue() noexcept;
  // Call 'zeros' for value type: 'tsr_t', 'srm_t', 'tsri_t'.
  void ZerosValue() noexcept;
  void RemoveEmptyValue();
};

/************************************************************************/
/* Instance */
/************************************************************************/
class Instance : public TensorMap {
 private:
  int batch_ = 0;

 public:
  void clear_batch() noexcept { batch_ = 0; }
  void set_batch(int batch) noexcept { batch_ = batch; }
  int batch() const noexcept { return batch_; }

  void clear() noexcept {
    TensorMap::clear();
    clear_batch();
  }

  void swap(Instance& other) noexcept {
    TensorMap::swap(other);
    std::swap(batch_, other.batch_);
  }
};

std::ostream& operator<<(std::ostream& os, const Instance& inst);

/************************************************************************/
/* Hidden */
/************************************************************************/
class Hidden : public TensorMap {
 private:
  std::default_random_engine engine_;
  float_t* loss_ = nullptr;
  Instance inst_;

 public:
  template <typename Int>
  void seed(Int s) {
    engine_.seed((std::default_random_engine::result_type)s);
  }
  std::default_random_engine& engine() noexcept { return engine_; }
  void clear_loss() noexcept { loss_ = nullptr; }
  void set_loss(float_t* loss) noexcept { loss_ = loss; }
  bool has_loss() const noexcept { return loss_ != nullptr; }
  float_t loss() const noexcept { return *loss_; }
  Instance* mutable_inst() noexcept { return &inst_; }
  const Instance& inst() const noexcept { return inst_; }
};

std::ostream& operator<<(std::ostream& os, const Hidden& hidden);

}  // namespace deepx_core
