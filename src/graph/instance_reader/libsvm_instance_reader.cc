// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/instance_reader_impl.h>
#include <deepx_core/instance/libsvm.h>

namespace deepx_core {

class LibsvmInstanceReader : public InstanceReaderImpl {
 private:
  csr_t* X_ = nullptr;

 public:
  DEFINE_INSTANCE_READER_LIKE(LibsvmInstanceReader);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else {
      DXERROR("Unexpected config: %s=%s.", k.c_str(), v.c_str());
      return false;
    }
    return true;
  }

  void InitX(Instance* inst) override {
    X_ = &inst->get_or_insert<csr_t>(X_NAME);
  }

  void InitXBatch(Instance* /*inst*/) override {
    X_->clear();
    X_->reserve(batch_);
  }

  bool ParseLine() override {
    LibsvmInstanceReaderHelper<float_t, int_t> helper(line_);
    return helper.Parse(X_, Y_, W_, uuid_);
  }
};

INSTANCE_READER_REGISTER(LibsvmInstanceReader, "LibsvmInstanceReader");
INSTANCE_READER_REGISTER(LibsvmInstanceReader, "libsvm");

}  // namespace deepx_core
