// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// include all headers needed by instance readers
#include <deepx_core/common/class_factory.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/instance_reader.h>
#include <vector>

namespace deepx_core {

#define INSTANCE_READER_REGISTER(class_name, name) \
  CLASS_FACTORY_REGISTER(InstanceReader, class_name, name)
#define INSTANCE_READER_NEW(name) CLASS_FACTORY_NEW(InstanceReader, name)
#define INSTANCE_READER_NAMES() CLASS_FACTORY_NAMES(InstanceReader)
#define DEFINE_INSTANCE_READER_LIKE(clazz_name) \
  const char* class_name() const noexcept override { return #clazz_name; }

/************************************************************************/
/* InstanceReaderImpl */
/************************************************************************/
class InstanceReaderImpl : public InstanceReader {
 protected:
  int batch_ = 32;
  int label_size_ = 1;
  int has_w_ = 0;
  int has_uuid_ = 0;

  AutoInputFileStream is_;
  std::string line_;
  tsr_t* Y_ = nullptr;
  tsr_t* W_ = nullptr;
  tsrs_t* uuid_ = nullptr;

 public:
  bool InitConfig(const AnyMap& config) override;
  bool InitConfig(const StringMap& config) override;
  bool Open(const std::string& file) override;
  void Close() noexcept override { is_.Close(); }
  bool GetBatch(Instance* inst) override;

 protected:
  virtual bool PreInitConfig() { return true; }
  virtual bool InitConfigKV(const std::string& k, const std::string& v) = 0;
  virtual bool PostInitConfig() { return true; }
  virtual void InitX(Instance* inst) = 0;
  virtual void InitXBatch(Instance* inst) = 0;
  virtual bool ParseLine() = 0;
};

}  // namespace deepx_core
