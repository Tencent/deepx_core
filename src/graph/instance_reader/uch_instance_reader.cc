// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/instance_reader_impl.h>
#include <deepx_core/instance/uch.h>

namespace deepx_core {

class UCHInstanceReader : public InstanceReaderImpl {
 private:
  int X_hist_item_size_ = 0;
  csr_t* X_user_ = nullptr;
  csr_t* X_cand_ = nullptr;
  std::vector<csr_t*> X_hist_;
  tsr_t* X_hist_size_ = nullptr;

 public:
  DEFINE_INSTANCE_READER_LIKE(UCHInstanceReader);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "x_hist_item_size" || k == "hist_item_size" ||
               k == "history_item_size") {
      X_hist_item_size_ = std::stoi(v);
      if (X_hist_item_size_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
      X_hist_.resize(X_hist_item_size_);
    } else {
      DXERROR("Unexpected config: %s=%s.", k.c_str(), v.c_str());
      return false;
    }
    return true;
  }

  bool PostInitConfig() override {
    if (X_hist_item_size_ == 0) {
      DXERROR("Please specify x_hist_item_size.");
      return false;
    }
    return true;
  }

  void InitX(Instance* inst) override {
    X_user_ = &inst->get_or_insert<csr_t>(X_USER_NAME);
    X_cand_ = &inst->get_or_insert<csr_t>(X_CAND_NAME);
    for (int i = 0; i < X_hist_item_size_; ++i) {
      X_hist_[i] = &inst->get_or_insert<csr_t>(X_HIST_NAME + std::to_string(i));
    }
    X_hist_size_ = &inst->get_or_insert<tsr_t>(X_HIST_SIZE_NAME);
  }

  void InitXBatch(Instance* /*inst*/) override {
    X_user_->clear();
    X_user_->reserve(batch_);
    X_cand_->clear();
    X_cand_->reserve(batch_);
    for (int i = 0; i < X_hist_item_size_; ++i) {
      X_hist_[i]->clear();
      X_hist_[i]->reserve(batch_);
    }
    X_hist_size_->resize(0);
    X_hist_size_->reserve(batch_);
  }

  bool ParseLine() override {
    UCHInstanceReaderHelper<float_t, int_t> helper(line_);
    return helper.Parse(X_user_, X_cand_, &X_hist_, X_hist_size_, Y_, W_,
                        uuid_);
  }
};

INSTANCE_READER_REGISTER(UCHInstanceReader, "UCHInstanceReader");
INSTANCE_READER_REGISTER(UCHInstanceReader, "uch");

}  // namespace deepx_core
