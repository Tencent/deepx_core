// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/common/hash.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/shard.h>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace deepx_core {
namespace {

/************************************************************************/
/* DefaultShardFunc */
/************************************************************************/
class DefaultShardFunc : public DataType {
 public:
  static int TSRShardFunc(const std::string& name, int shard_size) noexcept {
    return (int)((uint32_t)MurmurHash2(name) % (uint32_t)shard_size);
  }

  static int SRMShardFunc(int_t id, int shard_size) noexcept {
    return (int)((uint32_t)id % (uint32_t)shard_size);
  }
};

/************************************************************************/
/* ModuloShardFunc */
/************************************************************************/
// backward compatibility
class ModuloShardFunc : public DataType {
 public:
  static int TSRShardFunc(const std::string& /*name*/,
                          int /*shard_size*/) noexcept {
    return 0;
  }

  static int SRMShardFunc(int_t id, int shard_size) noexcept {
    id = (id & UINT64_C(0x0000ffffff000000)) >> 24;
    return (int)((uint32_t)id % (uint32_t)shard_size);
  }
};

/************************************************************************/
/* ShardFuncMap */
/************************************************************************/
class ShardFuncMap : public DataType {
 private:
  using shard_func_t = std::pair<tsr_shard_func_t, srm_shard_func_t>;
  using shard_func_map_t = std::unordered_map<std::string, shard_func_t>;
  shard_func_map_t map_;

 public:
  void Register(const std::string& name, tsr_shard_func_t tsr_shard_func,
                srm_shard_func_t srm_shard_func) {
    if (map_.count(name) > 0) {
      DXTHROW_INVALID_ARGUMENT("Duplicate registered name: %s.", name.c_str());
    }
    if (tsr_shard_func == nullptr) {
      tsr_shard_func = &DefaultShardFunc::TSRShardFunc;
    }
    if (srm_shard_func == nullptr) {
      srm_shard_func = &DefaultShardFunc::SRMShardFunc;
    }
    map_.emplace(name, std::make_pair(tsr_shard_func, srm_shard_func));
  }

  void Get(const std::string& name, tsr_shard_func_t* tsr_shard_func,
           srm_shard_func_t* srm_shard_func) {
    auto it = map_.find(name);
    if (it == map_.end()) {
      DXTHROW_INVALID_ARGUMENT("Unregistered name: %s.", name.c_str());
    }
    *tsr_shard_func = it->second.first;
    *srm_shard_func = it->second.second;
  }

 public:
  static ShardFuncMap& GetInstance() {
    static ShardFuncMap shard_func_map;
    return shard_func_map;
  }
};

/************************************************************************/
/* DefaultShardFuncRegister */
/************************************************************************/
class DefaultShardFuncRegister {
 public:
  DefaultShardFuncRegister() {
    ShardFuncMap::GetInstance().Register("default", nullptr, nullptr);
  }
} default_shard_func_register;

/************************************************************************/
/* ModuloShardFuncRegister */
/************************************************************************/
// backward compatibility
class ModuloShardFuncRegister {
 public:
  ModuloShardFuncRegister() {
    ShardFuncMap::GetInstance().Register("modulo",
                                         &ModuloShardFunc::TSRShardFunc,
                                         &ModuloShardFunc::SRMShardFunc);
  }
} modulo_shard_func_register;

}  // namespace

/************************************************************************/
/* Shard functions */
/************************************************************************/
namespace {

std::string GetShardFile(const std::string& dir) { return dir + "/shard.bin"; }

}  // namespace

bool SaveShard(const std::string& dir, const Shard& shard) {
  return shard.Save(GetShardFile(dir));
}

bool LoadShardLegacy(const std::string& dir, Shard* shard) {
  for (int i = 1; i <= 1024; ++i) {  // magic number
    std::string file = dir + "/SUCCESS_0." + std::to_string(i) + ".-2.1";
    if (AutoFileSystem::Exists(file)) {
      shard->InitShard(i, "modulo");
      return true;
    }
  }

  DXERROR("Invalid model dir: %s.", dir.c_str());
  return false;
}

bool LoadShard(const std::string& dir, Shard* shard) {
  // backward compatibility
  std::string file = dir + "/shard_info.bin";
  if (AutoFileSystem::Exists(file)) {
    return shard->Load(file);
  }
  return shard->Load(GetShardFile(dir));
}

/************************************************************************/
/* Shard */
/************************************************************************/
void Shard::RegisterShardFunc(const std::string& shard_func_name,
                              const tsr_shard_func_t& tsr_shard_func,
                              const srm_shard_func_t& srm_shard_func) {
  ShardFuncMap::GetInstance().Register(shard_func_name, tsr_shard_func,
                                       srm_shard_func);
}

void Shard::_Init(int shard_mode, int shard_size,
                  const std::string& shard_func_name) {
  shard_mode_ = shard_mode;
  shard_size_ = shard_size;
  shard_func_name_ = shard_func_name;
  ShardFuncMap::GetInstance().Get(shard_func_name_, &tsr_shard_func_,
                                  &srm_shard_func_);
}

void Shard::InitNonShard() { _Init(0, 1, "default"); }

void Shard::InitShard(int shard_size, const std::string& shard_func_name) {
  _Init(1, shard_size, shard_func_name);
}

bool Shard::Write(OutputStream& os) const {
  int version = 0x203de81b;  // magic number version
  os << version;
  os << shard_mode_ << shard_size_ << shard_func_name_;
  if (!os) {
    DXERROR("Failed to write shard.");
    return false;
  }
  return true;
}

bool Shard::Read(InputStream& is) {
  int version;
  if (is.Peek(&version, sizeof(version)) != sizeof(version)) {
    DXERROR("Failed to read shard.");
    return false;
  }

  if (version == 0x203de81b) {  // magic number version
    is >> version;
    is >> shard_mode_ >> shard_size_ >> shard_func_name_;
  } else {
    // backward compatibility
    int shard_size;
    is >> shard_size;
    if (is) {
      if (shard_size == 0) {
        shard_mode_ = 0;
        shard_size_ = 1;
      } else {
        shard_mode_ = 1;
        shard_size_ = shard_size;
      }
      shard_func_name_ = "default";
    }
  }

  if (!is) {
    DXERROR("Failed to read shard.");
    return false;
  }

  ShardFuncMap::GetInstance().Get(shard_func_name_, &tsr_shard_func_,
                                  &srm_shard_func_);
  return true;
}

bool Shard::Save(const std::string& file) const {
  AutoOutputFileStream os;
  if (!os.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Saving shard to %s...", file.c_str());
  if (!Write(os)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

bool Shard::Load(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }
  DXINFO("Loading shard from %s...", file.c_str());
  if (!Read(is)) {
    return false;
  }
  DXINFO("Done.");
  return true;
}

}  // namespace deepx_core
