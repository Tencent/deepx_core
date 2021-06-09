// Copyright 2021 the deepx authors.
// Author: Xingfei Li (xingfeili@tencent.com)
//

#pragma once
// include all headers needed by WePSOptimizers
#include <deepx_core/common/class_factory.h>
#include <deepx_core/common/lru_cache.h>
#include <deepx_core/contrib/we_ps/optimizer/ll_we_ps_optimizer.h>
#include <deepx_core/contrib/we_ps/optimizer/we_ps_optimizer.h>
#include <deepx_core/dx_log.h>
#include <unordered_map>
#include <vector>

namespace deepx_core {

#define WE_PS_OPTIMIZER_REGISTER(class_name, name) \
  CLASS_FACTORY_REGISTER(WePSOptimizer, class_name, name)
#define WE_PS_OPTIMIZER_NEW(name) CLASS_FACTORY_NEW(WePSOptimizer, name)
#define WE_PS_OPTIMIZER_NAMES() CLASS_FACTORY_NAMES(WePSOptimizer)
#define DEFINE_WE_PS_OPTIMIZER_LIKE(clazz_name) \
  const char* class_name() const noexcept override { return #clazz_name; }

/************************************************************************/
/* WePSOptimizerTSRSlot */
/************************************************************************/
struct WePSOptimizerTSRSlot : DataType {
  std::vector<tsr_t> O;
};

inline OutputStream& operator<<(OutputStream& os,
                                const WePSOptimizerTSRSlot& slot) {
  os << slot.O;
  return os;
}

inline InputStream& operator>>(InputStream& is, WePSOptimizerTSRSlot& slot) {
  is >> slot.O;
  return is;
}

/************************************************************************/
/* WePSOptimizerSRMSlot */
/************************************************************************/
struct WePSOptimizerSRMSlot : DataType {
  std::vector<srm_t> O;
};

/************************************************************************/
/* WePSOptimizerImpl */
/************************************************************************/
class WePSOptimizerImpl : public WePSOptimizer {
 protected:
  using ll_we_ps_optimizer_t = LLWePSOptimizer<float_t, int_t>;

  const Graph* graph_ = nullptr;
  TensorMap* param_ = nullptr;
  StringMap config_;
  std::unordered_map<std::string, WePSOptimizerTSRSlot> tsr_slot_map_;
  std::unordered_map<std::string, WePSOptimizerSRMSlot> srm_slot_map_;
  LRUCache<srm_t::key_type, bool> cache_;
  id_set_t evicted_ids_;
  id_set_t active_ids_;
  int update_times_ = 0;

 public:
  void Init(const Graph* graph, TensorMap* param) final;
  bool InitConfig(const AnyMap& /*config*/) override;
  bool InitConfig(const StringMap& /*config*/) override;
  bool InitParam() override;
  bool Write(OutputStream& os) const override;
  bool Read(InputStream& is) override;

 public:
  void Update(TensorMap* grad, TensorMap* delta_param) override;
  void ForEachSRM(
      const std::function<void(const std::string&, srm_t*)>& func) override;

 protected:
  virtual bool PreInitConfig() { return true; }
  virtual bool InitConfigKV(const std::string& k, const std::string& v) = 0;
  virtual bool PostInitConfig() { return true; }
  virtual void InitParamTSR(const std::string& name, const tsr_t& W,
                            WePSOptimizerTSRSlot* slot) const = 0;
  virtual void InitParamSRM(const std::string& name, const srm_t& W,
                            WePSOptimizerSRMSlot* slot) const = 0;
  void Update(const std::string& name, Any* Gany, Any* Wany, Any* Dany);
  virtual void PreUpdate() {}
  virtual void PostUpdate() {}
  virtual void UpdateTSR2TSR(const std::string& name, const tsr_t& G,
                             const tsr_t& W, tsr_t* D,
                             WePSOptimizerTSRSlot* slot) const = 0;
  virtual void UpdateSRM2TSR(const std::string& name, const srm_t& G,
                             const tsr_t& W, tsr_t* D,
                             WePSOptimizerTSRSlot* slot) const = 0;
  virtual void UpdateSRM2SRM(const std::string& name, const srm_t& G,
                             const srm_t& W, srm_t* D,
                             WePSOptimizerSRMSlot* slot) const = 0;
  void InitCache(size_t cache_size);
  void UpdateCache();
};

/************************************************************************/
/* WePSOptimizerBase0 */
/************************************************************************/
class WePSOptimizerBase0 : public WePSOptimizerImpl {
 protected:
  void InitParamTSR(const std::string& /*name*/, const tsr_t& /*W*/,
                    WePSOptimizerTSRSlot* /*slot*/) const override {}
  void InitParamSRM(const std::string& /*name*/, const srm_t& /*W*/,
                    WePSOptimizerSRMSlot* /*slot*/) const override {}
};

/************************************************************************/
/* WePSOptimizerBase1 */
/************************************************************************/
class WePSOptimizerBase1 : public WePSOptimizerImpl {
 protected:
  void InitParamTSR(const std::string& name, const tsr_t& W,
                    WePSOptimizerTSRSlot* slot) const override;
  void InitParamSRM(const std::string& name, const srm_t& W,
                    WePSOptimizerSRMSlot* slot) const override;
};

/************************************************************************/
/* WePSOptimizerBase2 */
/************************************************************************/
class WePSOptimizerBase2 : public WePSOptimizerImpl {
 protected:
  void InitParamTSR(const std::string& name, const tsr_t& W,
                    WePSOptimizerTSRSlot* slot) const override;
  void InitParamSRM(const std::string& name, const srm_t& W,
                    WePSOptimizerSRMSlot* slot) const override;
};

}  // namespace deepx_core
