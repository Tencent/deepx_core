// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/tensor_map.h>
#include <string>

namespace deepx_core {

/************************************************************************/
/* TensorMap */
/************************************************************************/
void TensorMap::_Write(OutputStream& os) const {
  int s = (int)size();
  os << s;
  for (const auto& entry : *this) {
    const std::string& k = entry.first;
    const Any& v = entry.second;
    if (v.is<tsr_t>()) {
      int type = TENSOR_TYPE_TSR;
      const auto& W = v.unsafe_to_ref<tsr_t>();
      os << k << type << W;
    } else if (v.is<srm_t>()) {
      int type = TENSOR_TYPE_SRM;
      const auto& W = v.unsafe_to_ref<srm_t>();
      os << k << type << W;
    } else if (v.is<csr_t>()) {
      int type = TENSOR_TYPE_CSR;
      const auto& W = v.unsafe_to_ref<csr_t>();
      os << k << type << W;
    } else if (v.is<tsri_t>()) {
      int type = TENSOR_TYPE_TSRI;
      const auto& W = v.unsafe_to_ref<tsri_t>();
      os << k << type << W;
    } else if (v.is<tsrs_t>()) {
      int type = TENSOR_TYPE_TSRS;
      const auto& W = v.unsafe_to_ref<tsrs_t>();
      os << k << type << W;
    } else {
      int type = TENSOR_TYPE_NONE;
      os << k << type;
    }
    if (!os) {
      break;
    }
  }
}

void TensorMap::_Read(InputStream& is) {
  int s;
  is >> s;
  if (!is) {
    return;
  }

  clear();
  for (int i = 0; i < s; ++i) {
    std::string name;
    int type;
    is >> name >> type;
    if (!is) {
      return;
    }

    switch (type) {
      case TENSOR_TYPE_TSR:
        is >> insert<tsr_t>(name);
        break;
      case TENSOR_TYPE_SRM:
        is >> insert<srm_t>(name);
        break;
      case TENSOR_TYPE_CSR:
        is >> insert<csr_t>(name);
        break;
      case TENSOR_TYPE_TSRI:
        is >> insert<tsri_t>(name);
        break;
      case TENSOR_TYPE_TSRS:
        is >> insert<tsrs_t>(name);
        break;
      case TENSOR_TYPE_SRP:  // backward compatibility
        ReadSRP(is, insert<srm_t>(name));
        break;
      case TENSOR_TYPE_SVP:  // backward compatibility
        ReadSVP(is, insert<srm_t>(name));
        break;
    }

    if (!is) {
      return;
    }
  }
}

void TensorMap::_ReadView(InputStringStream& is) {
  int s;
  ReadView(is, s);
  if (!is) {
    return;
  }

  clear();
  for (int i = 0; i < s; ++i) {
    std::string name;
    int type;
    ReadView(is, name);
    ReadView(is, type);
    if (!is) {
      return;
    }

    switch (type) {
      case TENSOR_TYPE_TSR:
        ReadView(is, insert<tsr_t>(name));
        break;
      case TENSOR_TYPE_SRM:
        ReadView(is, insert<srm_t>(name));
        break;
      case TENSOR_TYPE_CSR:
        ReadView(is, insert<csr_t>(name));
        break;
      case TENSOR_TYPE_TSRI:
        ReadView(is, insert<tsri_t>(name));
        break;
      case TENSOR_TYPE_TSRS:
        ReadView(is, insert<tsrs_t>(name));
        break;
      case TENSOR_TYPE_SRP:  // backward compatibility
        ReadSRPView(is, insert<srm_t>(name));
        break;
      case TENSOR_TYPE_SVP:  // backward compatibility
        ReadSVPView(is, insert<srm_t>(name));
        break;
    }

    if (!is) {
      return;
    }
  }
}

void TensorMap::_WriteText(std::ostream& os) const {
  for (const auto& entry : *this) {
    const std::string& k = entry.first;
    const Any& v = entry.second;
    if (v.is<tsr_t>()) {
      const auto& W = v.unsafe_to_ref<tsr_t>();
      os << k << std::endl;
      os << W << std::endl;
    } else if (v.is<srm_t>()) {
      const auto& W = v.unsafe_to_ref<srm_t>();
      os << k << std::endl;
      os << W << std::endl;
    } else if (v.is<csr_t>()) {
      const auto& W = v.unsafe_to_ref<csr_t>();
      os << k << std::endl;
      os << W << std::endl;
    } else if (v.is<tsri_t>()) {
      const auto& W = v.unsafe_to_ref<tsri_t>();
      os << k << std::endl;
      os << W << std::endl;
    } else if (v.is<tsrs_t>()) {
      const auto& W = v.unsafe_to_ref<tsrs_t>();
      os << k << std::endl;
      os << W << std::endl;
    } else {
      os << k << std::endl;
      os << std::endl;
    }
  }
}

OutputStream& operator<<(OutputStream& os, const TensorMap& tensor_map) {
  tensor_map._Write(os);
  return os;
}

InputStream& operator>>(InputStream& is, TensorMap& tensor_map) {
  tensor_map._Read(is);
  return is;
}

InputStringStream& ReadView(InputStringStream& is, TensorMap& tensor_map) {
  tensor_map._ReadView(is);
  return is;
}

std::ostream& operator<<(std::ostream& os, const TensorMap& tensor_map) {
  tensor_map._WriteText(os);
  return os;
}

void TensorMap::ClearSRMValue() noexcept {
  for (auto& entry : *this) {
    Any& Wany = entry.second;
    if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      W.zeros();
    }
  }
}

void TensorMap::ClearValue() noexcept {
  for (auto& entry : *this) {
    Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      W.clear();
    } else if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      W.zeros();
    } else if (Wany.is<csr_t>()) {
      auto& W = Wany.unsafe_to_ref<csr_t>();
      W.clear();
    } else if (Wany.is<tsri_t>()) {
      auto& W = Wany.unsafe_to_ref<tsri_t>();
      W.clear();
    } else if (Wany.is<tsrs_t>()) {
      auto& W = Wany.unsafe_to_ref<tsrs_t>();
      W.clear();
    }
  }
}

void TensorMap::ZerosValue() noexcept {
  for (auto& entry : *this) {
    Any& Wany = entry.second;
    if (Wany.is<tsr_t>()) {
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      W.zeros();
    } else if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      W.zeros();
    } else if (Wany.is<tsri_t>()) {
      auto& W = Wany.unsafe_to_ref<tsri_t>();
      W.zeros();
    }
  }
}

void TensorMap::RemoveEmptyValue() {
  auto first = begin();
  auto last = end();
  for (; first != last;) {
    bool empty = false;
    Any& Wany = first->second;
    if (Wany.is<tsr_t>()) {
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      empty = W.empty();
    } else if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      empty = W.empty();
    } else if (Wany.is<csr_t>()) {
      auto& W = Wany.unsafe_to_ref<csr_t>();
      empty = W.empty();
    } else if (Wany.is<tsri_t>()) {
      auto& W = Wany.unsafe_to_ref<tsri_t>();
      empty = W.empty();
    } else if (Wany.is<tsrs_t>()) {
      auto& W = Wany.unsafe_to_ref<tsrs_t>();
      empty = W.empty();
    }

    if (empty) {
      first = erase(first);
    } else {
      ++first;
    }
  }
}

/************************************************************************/
/* Instance */
/************************************************************************/
std::ostream& operator<<(std::ostream& os, const Instance& inst) {
  os << "batch=" << inst.batch() << std::endl;
  os << (const TensorMap&)inst;
  return os;
}

/************************************************************************/
/* Hidden */
/************************************************************************/
std::ostream& operator<<(std::ostream& os, const Hidden& hidden) {
  if (hidden.has_loss()) {
    os << "loss=" << hidden.loss() << std::endl;
  }
  os << hidden.inst();
  os << (const TensorMap&)hidden;
  return os;
}

}  // namespace deepx_core
