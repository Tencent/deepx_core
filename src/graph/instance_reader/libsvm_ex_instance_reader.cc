// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/instance_reader_impl.h>
#include <deepx_core/instance/libsvm_ex.h>

namespace deepx_core {

class LibsvmExInstanceReader : public InstanceReaderImpl {
 private:
  int X_size_ = 0;
  std::vector<csr_t*> X_;

 public:
  DEFINE_INSTANCE_READER_LIKE(LibsvmExInstanceReader);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "x_size") {
      X_size_ = std::stoi(v);
      if (X_size_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
      X_.resize(X_size_);
    } else {
      DXERROR("Unexpected config: %s=%s.", k.c_str(), v.c_str());
      return false;
    }
    return true;
  }

  bool PostInitConfig() override {
    if (X_size_ == 0) {
      DXERROR("Please specify x_size.");
      return false;
    }
    return true;
  }

  void InitX(Instance* inst) override {
    for (int i = 0; i < X_size_; ++i) {
      X_[i] = &inst->get_or_insert<csr_t>(X_NAME + std::to_string(i));
    }
  }

  void InitXBatch(Instance* /*inst*/) override {
    for (int i = 0; i < X_size_; ++i) {
      X_[i]->clear();
      X_[i]->reserve(batch_);
    }
  }

  bool ParseLine() override {
    LibsvmExInstanceReaderHelper<float_t, int_t> helper(line_);
    return helper.Parse(&X_, Y_, W_, uuid_);
  }
};

INSTANCE_READER_REGISTER(LibsvmExInstanceReader, "LibsvmExInstanceReader");
INSTANCE_READER_REGISTER(LibsvmExInstanceReader, "libsvm_ex");

}  // namespace deepx_core
