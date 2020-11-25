// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <string>

namespace deepx_core {

const std::string X_NAME = "__instX";  // libsvm, libsvm_ex
// for libsvm_ex
// X_NAME + "0"
// X_NAME + "1"
// X_NAME + "2"
// ...
const std::string X_USER_NAME = "__instXuser";            // uch
const std::string X_CAND_NAME = "__instXcand";            // uch
const std::string X_HIST_NAME = "__instXhist";            // uch
const std::string X_HIST_SIZE_NAME = "__instXhist_size";  // uch
const std::string Y_NAME = "__instY";                     // all
const std::string W_NAME = "__instW";                     // all
const std::string UUID_NAME = "__instUUID";               // all

/************************************************************************/
/* InstanceReader */
/************************************************************************/
class InstanceReader : public DataType {
 public:
  virtual ~InstanceReader() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual bool InitConfig(const AnyMap& config) = 0;
  virtual bool InitConfig(const StringMap& config) = 0;
  virtual bool Open(const std::string& file) = 0;
  virtual void Close() noexcept = 0;
  virtual bool GetBatch(Instance* inst) = 0;
};

/************************************************************************/
/* InstanceReader functions */
/************************************************************************/
std::unique_ptr<InstanceReader> NewInstanceReader(const std::string& name);

}  // namespace deepx_core
