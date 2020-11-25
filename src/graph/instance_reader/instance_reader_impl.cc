// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/instance_reader_impl.h>

namespace deepx_core {

/************************************************************************/
/* InstanceReaderImpl */
/************************************************************************/
bool InstanceReaderImpl::InitConfig(const AnyMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const auto& v = entry.second.to_ref<std::string>();
    if (!InitConfigKV(k, v)) {
      return false;
    }
  }

  if (!PostInitConfig()) {
    return false;
  }
  return true;
}

bool InstanceReaderImpl::InitConfig(const StringMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const std::string& v = entry.second;
    if (!InitConfigKV(k, v)) {
      return false;
    }
  }

  if (!PostInitConfig()) {
    return false;
  }
  return true;
}

bool InstanceReaderImpl::Open(const std::string& file) {
  if (!is_.Open(file)) {
    return false;
  }
  line_.reserve(64 * 1024);  // magic number
  return is_.IsOpen();
}

bool InstanceReaderImpl::GetBatch(Instance* inst) {
  InitX(inst);
  Y_ = &inst->get_or_insert<tsr_t>(Y_NAME);
  W_ = has_w_ ? &inst->get_or_insert<tsr_t>(W_NAME) : nullptr;
  uuid_ = has_uuid_ ? &inst->get_or_insert<tsrs_t>(UUID_NAME) : nullptr;

  if (Y_->empty() || Y_->dim(0) == batch_) {
    InitXBatch(inst);
    Y_->reserve(batch_ * label_size_);
    Y_->resize(0, label_size_);
    if (W_) {
      W_->reserve(batch_ * label_size_);
      W_->resize(0, label_size_);
    }
    if (uuid_) {
      uuid_->clear();
      uuid_->reserve(batch_);
    }
    inst->clear_batch();
  }

  for (;;) {
    if (!GetLine(is_, line_)) {
      is_.Close();
      return false;
    }

    if (ParseLine()) {
      inst->set_batch(Y_->dim(0));
      if (Y_->dim(0) == batch_) {
        return true;
      }
    }
  }
}

bool InstanceReaderImpl::InitConfigKV(const std::string& k,
                                      const std::string& v) {
  if (k == "batch" || k == "batch_size") {
    batch_ = std::stoi(v);
    if (batch_ <= 0) {
      DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
      return false;
    }
  } else if (k == "label_size") {
    label_size_ = std::stoi(v);
    if (label_size_ <= 0) {
      DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
      return false;
    }
  } else if (k == "w" || k == "has_w") {
    has_w_ = std::stoi(v);
  } else if (k == "uuid" || k == "has_uuid") {
    has_uuid_ = std::stoi(v);
  } else {
    return false;
  }
  return true;
}

/************************************************************************/
/* InstanceReader functions */
/************************************************************************/
std::unique_ptr<InstanceReader> NewInstanceReader(const std::string& name) {
  std::unique_ptr<InstanceReader> instance_reader(INSTANCE_READER_NEW(name));
  if (!instance_reader) {
    DXERROR("Invalid instance reader name: %s.", name.c_str());
    DXERROR("Instance reader name can be:");
    for (const std::string& _name : INSTANCE_READER_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return instance_reader;
}

}  // namespace deepx_core
