// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/array_view.h>
#include <deepx_core/common/stream.h>
#include <type_traits>  // std::is_pod

namespace deepx_core {

template <typename T>
OutputStream& operator<<(OutputStream& os, const ArrayView<T>& a) {
  static_assert(std::is_pod<T>::value, "T must be POD.");
  int size = (int)a.size();
  os.Write(&size, sizeof(size));
  if (size > 0) {
    os.Write(a.data(), sizeof(T) * a.size());
  }
  return os;
}

template <typename T>
InputStringStream& ReadView(InputStringStream& is, ArrayView<T>& a) {  // NOLINT
  static_assert(std::is_pod<T>::value, "T must be POD.");
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      // The cast is ugly and unsafe.
      a = ArrayView<T>((T*)is.GetData(), size);
      is.Skip(sizeof(T) * size);
    } else {
      a.clear();
    }
  }
  return is;
}

template <typename T>
OutputStream& operator<<(OutputStream& os, const ConstArrayView<T>& a) {
  static_assert(std::is_pod<T>::value, "T must be POD.");
  int size = (int)a.size();
  os.Write(&size, sizeof(size));
  if (size > 0) {
    os.Write(a.data(), sizeof(T) * a.size());
  }
  return os;
}

template <typename T>
InputStringStream& ReadView(InputStringStream& is,   // NOLINT
                            ConstArrayView<T>& a) {  // NOLINT
  static_assert(std::is_pod<T>::value, "T must be POD.");
  int size;
  is.Read(&size, sizeof(size));
  if (is) {
    if (size > 0) {
      a = ConstArrayView<T>((const T*)is.GetData(), size);
      is.Skip(sizeof(T) * size);
    } else {
      a.clear();
    }
  }
  return is;
}

}  // namespace deepx_core
