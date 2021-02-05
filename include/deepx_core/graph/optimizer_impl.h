// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// include all headers needed by optimizers
#include <deepx_core/common/class_factory.h>
#include <deepx_core/common/read_write_lock.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/optimizer.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace deepx_core {

#define OPTIMIZER_REGISTER(class_name, name) \
  CLASS_FACTORY_REGISTER(Optimizer, class_name, name)
#define OPTIMIZER_NEW(name) CLASS_FACTORY_NEW(Optimizer, name)
#define OPTIMIZER_NAMES() CLASS_FACTORY_NAMES(Optimizer)
#define DEFINE_OPTIMIZER_LIKE(clazz_name) \
  const char* class_name() const noexcept override { return #clazz_name; }

/************************************************************************/
/* OptimizerTSRSlot */
/************************************************************************/
struct OptimizerTSRSlot : DataType {
  std::vector<tsr_t> O;
};

inline OutputStream& operator<<(OutputStream& os,
                                const OptimizerTSRSlot& slot) {
  os << slot.O;
  return os;
}

inline InputStream& operator>>(InputStream& is, OptimizerTSRSlot& slot) {
  is >> slot.O;
  return is;
}

/************************************************************************/
/* OptimizerSRMSlot */
/************************************************************************/
struct OptimizerSRMSlot : DataType {
  std::vector<srm_t> O;
  std::shared_ptr<ReadWriteLock> Wlock;
  std::vector<std::unique_ptr<ReadWriteLock>> Olock;
};

inline OutputStream& operator<<(OutputStream& os,
                                const OptimizerSRMSlot& slot) {
  os << slot.O;
  return os;
}

inline InputStream& operator>>(InputStream& is, OptimizerSRMSlot& slot) {
  is >> slot.O;
  return is;
}

// backward compatibility
inline InputStream& ReadOptimizerSRPSlot(InputStream& is,           // NOLINT
                                         OptimizerSRMSlot& slot) {  // NOLINT
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      slot.O.resize(size);
      for (int i = 0; i < size; ++i) {
        ReadSRP(is, slot.O[i]);
        if (!is) {
          return is;
        }
      }
    } else {
      slot.O.clear();
    }
  }
  return is;
}

// backward compatibility
inline InputStream& ReadOptimizerSVPSlot(InputStream& is,           // NOLINT
                                         OptimizerSRMSlot& slot) {  // NOLINT
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      slot.O.resize(size);
      for (int i = 0; i < size; ++i) {
        ReadSVP(is, slot.O[i]);
        if (!is) {
          return is;
        }
      }
    } else {
      slot.O.clear();
    }
  }
  return is;
}

// backward compatibility
inline InputStream& ReadOptimizerSRPSlotMap(
    InputStream& is,                                                // NOLINT
    std::unordered_map<std::string, OptimizerSRMSlot>& slot_map) {  // NOLINT
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (size_t)size_u64;
  } else {
    int size_i = 0;
    is >> size_i;
    size = (size_t)size_i;
  }
  if (!is) {
    return is;
  }

  slot_map.clear();
  if (size > 0) {
    std::string name;
    OptimizerSRMSlot slot;
    slot_map.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      is >> name;
      ReadOptimizerSRPSlot(is, slot);
      if (!is) {
        return is;
      }
      slot_map.emplace(std::move(name), std::move(slot));
    }
  }
  return is;
}

// backward compatibility
inline InputStream& ReadOptimizerSVPSlotMap(
    InputStream& is,                                                // NOLINT
    std::unordered_map<std::string, OptimizerSRMSlot>& slot_map) {  // NOLINT
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    return is;
  }

  size_t size;
  if (version == 0x0a0c72e7) {  // magic number version
    uint64_t size_u64 = 0;
    is >> version;
    is >> size_u64;
    size = (size_t)size_u64;
  } else {
    int size_i = 0;
    is >> size_i;
    size = (size_t)size_i;
  }
  if (!is) {
    return is;
  }

  slot_map.clear();
  if (size > 0) {
    std::string name;
    OptimizerSRMSlot slot;
    slot_map.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      is >> name;
      ReadOptimizerSVPSlot(is, slot);
      if (!is) {
        return is;
      }
      slot_map.emplace(std::move(name), std::move(slot));
    }
  }
  return is;
}

/************************************************************************/
/* OptimizerImpl */
/************************************************************************/
class OptimizerImpl : public Optimizer {
 protected:
  const Graph* graph_ = nullptr;
  TensorMap* param_ = nullptr;
  StringMap config_;
  std::unordered_map<std::string, OptimizerTSRSlot> tsr_slot_map_;
  std::unordered_map<std::string, OptimizerSRMSlot> srm_slot_map_;
  int use_lock_ = 0;

 public:
  void Init(const Graph* graph, TensorMap* param) final;
  bool InitConfig(const AnyMap& /*config*/) override;
  bool InitConfig(const StringMap& /*config*/) override;
  bool InitParam() override;
  void InitLock(AnyMap* param_lock) override;
  bool WriteLegacy(OutputStream& os) const override;
  bool Write(OutputStream& os) const override;
  bool ReadLegacy(InputStream& is) override;
  bool Read(InputStream& is) override;
  bool MergeLegacy(Optimizer* other, const Shard* shard, int shard_id) override;
  bool Merge(Optimizer* other, const Shard* shard, int shard_id) override;

 public:
  void Update(TensorMap* grad) override;
  void ForEachSRM(
      const std::function<void(const std::string&, srm_t*)>& func) override;

 protected:
  virtual bool PreInitConfig() { return true; }
  virtual bool InitConfigKV(const std::string& k, const std::string& v) = 0;
  virtual bool PostInitConfig() { return true; }
  virtual void InitParamTSR(const std::string& name, const tsr_t& W,
                            OptimizerTSRSlot* slot) const = 0;
  virtual void InitParamSRM(const std::string& name, const srm_t& W,
                            OptimizerSRMSlot* slot) const = 0;
  // backward compatibility
  virtual void WriteConfigLegacy(OutputStream& /*os*/) const {}  // NOLINT
  // backward compatibility
  virtual void ReadConfigLegacy(InputStream& /*is*/) {}  // NOLINT
  // backward compatibility
  virtual void CopyConfigLegacy(const Optimizer& /*other*/) {}
  void UpdateParam(const std::string& name, Any* Gany, Any* Wany);
  virtual void PreUpdate() {}
  virtual void PostUpdate() {}
  virtual void UpdateTSR2TSR(const std::string& name, const tsr_t& G, tsr_t* W,
                             OptimizerTSRSlot* slot) const = 0;
  virtual void UpdateSRM2TSR(const std::string& name, const srm_t& G, tsr_t* W,
                             OptimizerTSRSlot* slot) const = 0;
  virtual void UpdateSRM2SRM(const std::string& name, const srm_t& G, srm_t* W,
                             OptimizerSRMSlot* slot) const = 0;

 private:
  using config_reduce_func_t = std::function<void(StringMap&, StringMap&)>;
  using tsr_reduce_func_t =
      std::function<void(const std::string&, tsr_t&, tsr_t&)>;
  using srm_reduce_func_t =
      std::function<void(const std::string&, srm_t&, srm_t&)>;
  bool Reduce(Optimizer* other, const config_reduce_func_t& config_reduce_func,
              const tsr_reduce_func_t& tsr_reduce_func,
              const srm_reduce_func_t& srm_reduce_func,
              const Shard* shard = nullptr, int shard_id = 0);
};

/************************************************************************/
/* OptimizerBase0 */
/************************************************************************/
class OptimizerBase0 : public OptimizerImpl {
 protected:
  void InitParamTSR(const std::string& /*name*/, const tsr_t& /*W*/,
                    OptimizerTSRSlot* /*slot*/) const override {}
  void InitParamSRM(const std::string& /*name*/, const srm_t& /*W*/,
                    OptimizerSRMSlot* /*slot*/) const override {}
};

/************************************************************************/
/* OptimizerBase1 */
/************************************************************************/
class OptimizerBase1 : public OptimizerImpl {
 protected:
  void InitParamTSR(const std::string& name, const tsr_t& W,
                    OptimizerTSRSlot* slot) const override;
  void InitParamSRM(const std::string& name, const srm_t& W,
                    OptimizerSRMSlot* slot) const override;
};

/************************************************************************/
/* OptimizerBase2 */
/************************************************************************/
class OptimizerBase2 : public OptimizerImpl {
 protected:
  void InitParamTSR(const std::string& name, const tsr_t& W,
                    OptimizerTSRSlot* slot) const override;
  void InitParamSRM(const std::string& name, const srm_t& W,
                    OptimizerSRMSlot* slot) const override;
};

}  // namespace deepx_core
