// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/common/vector.h>
#include <type_traits>  // std::is_pod, ...

namespace deepx_core {
namespace detail {

template <typename T>
void WriteVector(OutputStream& os, const Vector<T>& v,  // NOLINT
                 std::true_type /*is_pod*/) {
  int size = (int)v.size();
  os.Write(&size, sizeof(size));
  if (size > 0) {
    os.Write(v.data(), sizeof(T) * size);
  }
}

template <typename T>
void WriteVector(OutputStream& os, const Vector<T>& v,  // NOLINT
                 std::false_type /*is_pod*/) {
  int size = (int)v.size();
  os.Write(&size, sizeof(size));
  for (int i = 0; i < size; ++i) {
    os << v[i];
    if (!os) {
      break;
    }
  }
}

template <typename T>
void ReadVector(InputStream& is, Vector<T>& v,  // NOLINT
                std::true_type /*is_pod*/) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    v.clear();
    if (size > 0) {
      v.resize(size);
      is.Read(&v[0], sizeof(T) * size);
    }
  }
}

template <typename T>
void ReadVector(InputStream& is, Vector<T>& v,  // NOLINT
                std::false_type /*is_pod*/) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    v.clear();
    if (size > 0) {
      v.resize(size);
      for (int i = 0; i < size; ++i) {
        is >> v[i];
        if (!is) {
          return;
        }
      }
    }
  }
}

template <typename T>
void ReadVectorView(InputStringStream& is, Vector<T>& v,  // NOLINT
                    std::true_type /*is_pod*/) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    v.clear();
    if (size > 0) {
      v.view((const T*)is.GetData(), size);
      is.Skip(sizeof(T) * size);
    }
  }
}

template <typename T>
void ReadVectorView(InputStringStream& is, Vector<T>& v,  // NOLINT
                    std::false_type /*is_pod*/) {
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    v.clear();
    if (size > 0) {
      v.resize(size);
      for (int i = 0; i < size; ++i) {
        ReadView(is, v[i]);
        if (!is) {
          return;
        }
      }
    }
  }
}

}  // namespace detail

template <typename T>
OutputStream& operator<<(OutputStream& os, const Vector<T>& v) {
  detail::WriteVector(os, v, typename std::is_pod<T>::type());
  return os;
}

template <typename T>
InputStream& operator>>(InputStream& is, Vector<T>& v) {
  detail::ReadVector(is, v, typename std::is_pod<T>::type());
  return is;
}

template <typename T>
InputStringStream& ReadView(InputStringStream& is, Vector<T>& v) {  // NOLINT
  detail::ReadVectorView(is, v, typename std::is_pod<T>::type());
  return is;
}

}  // namespace deepx_core
