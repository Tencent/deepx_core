// Copyright 2019 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

using ai_t = std::array<int, 3>;

struct PoolAux {
  Shape Z;
  int pool_type = GraphNodePoolBase::POOL_TYPE_NONE;
  int pool_rank = 0;
  int ncx = 0;
  ai_t kernel_sizes;
  ai_t strides;
  ai_t dilations;
  ai_t paddings;
  ai_t Xspatials;
  ai_t Zspatials;
  int out_spatial_loop = 0;
  int X_out_spatial_stride = 0;
  int Z_out_spatial_stride = 0;
  int in_spatial_loop = 0;
  int in_spatial_stride = 0;
  int kernel_total_dim = 0;
  int count_include_pad = 0;
};

bool PoolCheckAttr(int pool_rank, int data_format,
                   const std::vector<int>& kernel_sizes,
                   const std::vector<int>& strides,
                   const std::vector<int>& dilations, int padding_mode,
                   const std::vector<int>& paddings, int ceil_mode) noexcept {
  if (pool_rank != 1 && pool_rank != 2 && pool_rank != 3) {
    DXERROR("Invalid pool_rank: pool_rank %d must be 1, 2 or 3.", pool_rank);
    return false;
  }

  if (pool_rank == 1) {
    if (data_format != GraphNodePoolBase::DATA_FORMAT_NCW &&
        data_format != GraphNodePoolBase::DATA_FORMAT_NWC) {
      DXERROR(
          "Invalid data_format: data_format must be DATA_FORMAT_NCW or "
          "DATA_FORMAT_NWC.");
      return false;
    }
  } else if (pool_rank == 2) {
    if (data_format != GraphNodePoolBase::DATA_FORMAT_NCHW &&
        data_format != GraphNodePoolBase::DATA_FORMAT_NHWC) {
      DXERROR(
          "Invalid data_format: data_format must be DATA_FORMAT_NCHW or "
          "DATA_FORMAT_NHWC.");
      return false;
    }
  } else {
    if (data_format != GraphNodePoolBase::DATA_FORMAT_NCDHW &&
        data_format != GraphNodePoolBase::DATA_FORMAT_NDHWC) {
      DXERROR(
          "Invalid data_format: data_format must be DATA_FORMAT_NCDHW or "
          "DATA_FORMAT_NDHWC.");
      return false;
    }
  }

  if ((int)kernel_sizes.size() != pool_rank) {
    DXERROR("Invalid kernel_sizes: size of kernel_sizes %d must be %d.",
            (int)strides.size(), pool_rank);
    return false;
  }

  for (int kernel_size : kernel_sizes) {
    if (kernel_size <= 0) {
      DXERROR("Invalid kernel_sizes: kernel_size %d must be positive.",
              kernel_size);
      return false;
    }
  }

  if ((int)strides.size() != pool_rank) {
    DXERROR("Invalid strides: size of strides %d must be %d.",
            (int)strides.size(), pool_rank);
    return false;
  }

  for (int stride : strides) {
    if (stride <= 0) {
      DXERROR("Invalid strides: stride %d must be positive.", stride);
      return false;
    }
  }

  if ((int)dilations.size() != pool_rank) {
    DXERROR("Invalid dilations: size of dilations %d must be %d.",
            (int)dilations.size(), pool_rank);
    return false;
  }

  for (int dilation : dilations) {
    if (dilation <= 0) {
      DXERROR("Invalid dilations: dilation %d must be positive.", dilation);
      return false;
    }
  }

  if (padding_mode != GraphNodePoolBase::PADDING_MODE_SAME &&
      padding_mode != GraphNodePoolBase::PADDING_MODE_VALID &&
      padding_mode != GraphNodePoolBase::PADDING_MODE_USE_PADDINGS) {
    DXERROR(
        "Invalid padding_mode: padding_mode must be PADDING_MODE_SAME, "
        "PADDING_MODE_VALID or PADDING_MODE_USE_PADDINGS.");
    return false;
  }

  if (padding_mode == GraphNodePoolBase::PADDING_MODE_USE_PADDINGS) {
    if ((int)paddings.size() != pool_rank) {
      DXERROR("Invalid paddings: size of paddings %d must be %d.",
              (int)paddings.size(), pool_rank);
      return false;
    }

    for (int padding : paddings) {
      if (padding < 0) {
        DXERROR("Invalid paddings: padding %d must be non-negative.", padding);
        return false;
      }
    }

    for (int i = 0; i < pool_rank; ++i) {
      if (paddings[i] > kernel_sizes[i] / 2) {
        DXERROR(
            "Invalid paddings and kernel_sizes: padding %d must be less than "
            "or equal to half of kernel_size %d.",
            paddings[i], kernel_sizes[i] / 2);
        return false;
      }
    }
  }

  if (padding_mode != GraphNodePoolBase::PADDING_MODE_USE_PADDINGS) {
    if (ceil_mode) {
      DXERROR(
          "Invalid ceil_mode: ceil_mode must be zero when padding_mode is not "
          "PADDING_MODE_PADDINGS.");
      return false;
    }
  }

  return true;
}

int DilateKernelSize(int kernel_size, int dilation) noexcept {
  return (kernel_size - 1) * dilation + 1;
}

bool PoolPrepare(const Shape& X, int pool_type, int pool_rank, int data_format,
                 const std::vector<int>& kernel_sizes,
                 const std::vector<int>& strides,
                 const std::vector<int>& dilations, int padding_mode,
                 const std::vector<int>& paddings, int ceil_mode,
                 int count_include_pad, PoolAux* aux) noexcept {
  int Xrank = X.rank();
  if (Xrank != pool_rank + 2) {
    DXERROR("Invalid X: rank of X %d must be %d.", Xrank, pool_rank + 2);
    return false;
  }

  int ncx = data_format == GraphNodePoolBase::DATA_FORMAT_NCW ||
            data_format == GraphNodePoolBase::DATA_FORMAT_NCHW ||
            data_format == GraphNodePoolBase::DATA_FORMAT_NCDHW;
  int X_spatial_start = ncx ? 2 : 1;
  ai_t Xspatials;
  for (int i = 0; i < pool_rank; ++i) {
    Xspatials[i] = X[X_spatial_start + i];
  }

  auto ceil_div = [](int a, int b) noexcept { return (a - 1) / b + 1; };
  auto z_dim = [ ceil_mode, &ceil_div ](int x, int k, int stride, int dilation,
                                        int padding) noexcept {
    if (ceil_mode) {
      return ceil_div(x + 2 * padding - DilateKernelSize(k, dilation), stride) +
             1;
    } else {
      return (x + 2 * padding - DilateKernelSize(k, dilation)) / stride + 1;
    }
  };

  ai_t Zspatials;
  ai_t _paddings;
  if (padding_mode == GraphNodePoolBase::PADDING_MODE_SAME) {
    for (int i = 0; i < pool_rank; ++i) {
      Zspatials[i] = ceil_div(Xspatials[i], strides[i]);
      _paddings[i] =
          ((Zspatials[i] - 1) * strides[i] +
           DilateKernelSize(kernel_sizes[i], dilations[i]) - Xspatials[i]) /
          2;
    }
  } else {
    if (padding_mode == GraphNodePoolBase::PADDING_MODE_VALID) {
      for (int i = 0; i < pool_rank; ++i) {
        _paddings[i] = 0;
        Zspatials[i] = z_dim(Xspatials[i], kernel_sizes[i], strides[i],
                             dilations[i], _paddings[i]);
      }
    } else {
      for (int i = 0; i < pool_rank; ++i) {
        _paddings[i] = paddings[i];
        Zspatials[i] = z_dim(Xspatials[i], kernel_sizes[i], strides[i],
                             dilations[i], _paddings[i]);
      }
    }

    for (int i = 0; i < pool_rank; ++i) {
      if (Zspatials[i] <= 0) {
        DXERROR(
            "Invalid X, kernel_sizes, strides, dilations and paddings "
            "combination.");
        return false;
      }
    }
  }

  int batch = X[0];
  int channel = ncx ? X[1] : X[Xrank - 1];
  int Zdims[SHAPE_MAX_RANK];
  int Zrank = 0;
  Zdims[Zrank++] = batch;
  if (ncx) {
    Zdims[Zrank++] = channel;
    for (int i = 0; i < pool_rank; ++i) {
      Zdims[Zrank++] = Zspatials[i];
    }
  } else {
    for (int i = 0; i < pool_rank; ++i) {
      Zdims[Zrank++] = Zspatials[i];
    }
    Zdims[Zrank++] = channel;
  }

  aux->Z.assign(&Zdims[0], &Zdims[Zrank]);
  aux->pool_type = pool_type;
  aux->pool_rank = pool_rank;
  aux->ncx = ncx;
  int kernel_total_dim = 1, X_spatial_total_dim = 1, Z_spatial_total_dim = 1;
  for (int i = 0; i < pool_rank; ++i) {
    aux->kernel_sizes[i] = kernel_sizes[i];
    kernel_total_dim *= kernel_sizes[i];
    aux->strides[i] = strides[i];
    aux->dilations[i] = dilations[i];
    aux->paddings[i] = _paddings[i];
    aux->Xspatials[i] = Xspatials[i];
    X_spatial_total_dim *= Xspatials[i];
    aux->Zspatials[i] = Zspatials[i];
    Z_spatial_total_dim *= Zspatials[i];
  }
  if (aux->ncx) {
    aux->out_spatial_loop = batch * channel;
    aux->X_out_spatial_stride = X_spatial_total_dim;
    aux->Z_out_spatial_stride = Z_spatial_total_dim;
    aux->in_spatial_loop = 1;
    aux->in_spatial_stride = 1;
  } else {
    aux->out_spatial_loop = batch;
    aux->X_out_spatial_stride = channel * X_spatial_total_dim;
    aux->Z_out_spatial_stride = channel * Z_spatial_total_dim;
    aux->in_spatial_loop = channel;
    aux->in_spatial_stride = channel;
  }
  aux->kernel_total_dim = kernel_total_dim;
  aux->count_include_pad = count_include_pad;

  return true;
}

bool PoolInferShape(const Shape& X, int pool_type, int pool_rank,
                    int data_format, const std::vector<int>& kernel_sizes,
                    const std::vector<int>& strides,
                    const std::vector<int>& dilations, int padding_mode,
                    const std::vector<int>& paddings, int ceil_mode,
                    int count_include_pad, Shape* Z) noexcept {
  PoolAux aux;
  if (!PoolPrepare(X, pool_type, pool_rank, data_format, kernel_sizes, strides,
                   dilations, padding_mode, paddings, ceil_mode,
                   count_include_pad, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

int FirstGE0(int start, int dilation) noexcept {
  while (start < 0) {
    start += dilation;
  }
  return start;
}

int ComputeOffset(int wi, int stride) noexcept { return wi * stride; }

int ComputeOffset(int hi, int wi, int w, int stride) noexcept {
  return (hi * w + wi) * stride;
}

int ComputeOffset(int di, int hi, int wi, int hw, int w, int stride) noexcept {
  return (di * hw + hi * w + wi) * stride;
}

/************************************************************************/
/* Pool */
/************************************************************************/
template <typename T>
void Pool(const Tensor<T>& X, Tensor<T>* Z, const PoolAux& aux) noexcept {
  if (aux.pool_type == GraphNodePoolBase::POOL_TYPE_MAX) {
    if (aux.pool_rank == 1) {
      MaxPool1d(X, Z, aux);
    } else if (aux.pool_rank == 2) {
      MaxPool2d(X, Z, aux);
    } else {
      MaxPool3d(X, Z, aux);
    }
  } else {
    if (aux.pool_rank == 1) {
      AvgPool1d(X, Z, aux);
    } else if (aux.pool_rank == 2) {
      AvgPool2d(X, Z, aux);
    } else {
      AvgPool3d(X, Z, aux);
    }
  }
}

template <typename T>
void MaxPool1d(const Tensor<T>& X, Tensor<T>* Z, const PoolAux& aux) noexcept {
  int kernel_w = aux.kernel_sizes[0];
  int stride_w = aux.strides[0];
  int dilation_w = aux.dilations[0];
  int padding_w = aux.paddings[0];
  int X_w = aux.Xspatials[0];
  int Z_w = aux.Zspatials[0];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  const T* _X = X.data();
  T* _Z = Z->data();

  Z->constant(-std::numeric_limits<T>::max());
  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
      int X_wi_start = Z_wi * stride_w - padding_w;
      int X_wi_end =
          std::min(X_wi_start + DilateKernelSize(kernel_w, dilation_w), X_w);
      X_wi_start = FirstGE0(X_wi_start, dilation_w);

      int Z_offset = ComputeOffset(Z_wi, in_spatial_stride);
      for (int X_wi = X_wi_start; X_wi < X_wi_end; X_wi += dilation_w) {
        int X_offset = ComputeOffset(X_wi, in_spatial_stride);
        for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
          int X_index = X_offset + in_i;
          int Z_index = Z_offset + in_i;
          _Z[Z_index] = std::max(_Z[Z_index], _X[X_index]);
        }
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
  }
}

template <typename T>
void MaxPool2d(const Tensor<T>& X, Tensor<T>* Z, const PoolAux& aux) noexcept {
  int kernel_h = aux.kernel_sizes[0], kernel_w = aux.kernel_sizes[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int dilation_h = aux.dilations[0], dilation_w = aux.dilations[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int X_h = aux.Xspatials[0], X_w = aux.Xspatials[1];
  int Z_h = aux.Zspatials[0], Z_w = aux.Zspatials[1];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  const T* _X = X.data();
  T* _Z = Z->data();

  Z->constant(-std::numeric_limits<T>::max());
  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
      for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
        int X_hi_start = Z_hi * stride_h - padding_h;
        int X_wi_start = Z_wi * stride_w - padding_w;
        int X_hi_end =
            std::min(X_hi_start + DilateKernelSize(kernel_h, dilation_h), X_h);
        int X_wi_end =
            std::min(X_wi_start + DilateKernelSize(kernel_w, dilation_w), X_w);
        X_hi_start = FirstGE0(X_hi_start, dilation_h);
        X_wi_start = FirstGE0(X_wi_start, dilation_w);

        int Z_offset = ComputeOffset(Z_hi, Z_wi, Z_w, in_spatial_stride);
        for (int X_hi = X_hi_start; X_hi < X_hi_end; X_hi += dilation_h) {
          for (int X_wi = X_wi_start; X_wi < X_wi_end; X_wi += dilation_w) {
            int X_offset = ComputeOffset(X_hi, X_wi, X_w, in_spatial_stride);
            for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
              int X_index = X_offset + in_i;
              int Z_index = Z_offset + in_i;
              _Z[Z_index] = std::max(_Z[Z_index], _X[X_index]);
            }
          }
        }
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
  }
}

template <typename T>
void MaxPool3d(const Tensor<T>& X, Tensor<T>* Z, const PoolAux& aux) noexcept {
  int kernel_d = aux.kernel_sizes[0], kernel_h = aux.kernel_sizes[1],
      kernel_w = aux.kernel_sizes[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int dilation_d = aux.dilations[0], dilation_h = aux.dilations[1],
      dilation_w = aux.dilations[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_d = aux.Xspatials[0], X_h = aux.Xspatials[1], X_w = aux.Xspatials[2];
  int Z_d = aux.Zspatials[0], Z_h = aux.Zspatials[1], Z_w = aux.Zspatials[2];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  const T* _X = X.data();
  T* _Z = Z->data();

  int X_hw = X_h * X_w;
  int Z_hw = Z_h * Z_w;
  Z->constant(-std::numeric_limits<T>::max());
  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
      for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
        for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
          int X_di_start = Z_di * stride_d - padding_d;
          int X_hi_start = Z_hi * stride_h - padding_h;
          int X_wi_start = Z_wi * stride_w - padding_w;
          int X_di_end = std::min(
              X_di_start + DilateKernelSize(kernel_d, dilation_d), X_d);
          int X_hi_end = std::min(
              X_hi_start + DilateKernelSize(kernel_h, dilation_h), X_h);
          int X_wi_end = std::min(
              X_wi_start + DilateKernelSize(kernel_w, dilation_w), X_w);
          X_di_start = FirstGE0(X_di_start, dilation_d);
          X_hi_start = FirstGE0(X_hi_start, dilation_h);
          X_wi_start = FirstGE0(X_wi_start, dilation_w);

          int Z_offset =
              ComputeOffset(Z_di, Z_hi, Z_wi, Z_hw, Z_w, in_spatial_stride);
          for (int X_di = X_di_start; X_di < X_di_end; X_di += dilation_d) {
            for (int X_hi = X_hi_start; X_hi < X_hi_end; X_hi += dilation_h) {
              for (int X_wi = X_wi_start; X_wi < X_wi_end; X_wi += dilation_w) {
                int X_offset = ComputeOffset(X_di, X_hi, X_wi, X_hw, X_w,
                                             in_spatial_stride);
                for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
                  int X_index = X_offset + in_i;
                  int Z_index = Z_offset + in_i;
                  _Z[Z_index] = std::max(_Z[Z_index], _X[X_index]);
                }
              }
            }
          }
        }
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
  }
}

template <typename T>
void AvgPool1d(const Tensor<T>& X, Tensor<T>* Z, const PoolAux& aux) noexcept {
  int kernel_w = aux.kernel_sizes[0];
  int stride_w = aux.strides[0];
  int padding_w = aux.paddings[0];
  int X_w = aux.Xspatials[0];
  int Z_w = aux.Zspatials[0];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  int kernel_total_dim = aux.kernel_total_dim;
  int count_include_pad = aux.count_include_pad;
  const T* _X = X.data();
  T* _Z = Z->data();

  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
      int X_wi_start = Z_wi * stride_w - padding_w;
      int X_wi_end = std::min(X_wi_start + kernel_w, X_w);
      X_wi_start = std::max(X_wi_start, 0);

      int Z_offset = ComputeOffset(Z_wi, in_spatial_stride);
      int no_pad_total_dim = (X_wi_end - X_wi_start);
      int count = count_include_pad ? kernel_total_dim : no_pad_total_dim;
      for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
        T sum = 0;
        for (int X_wi = X_wi_start; X_wi < X_wi_end; ++X_wi) {
          int X_offset = ComputeOffset(X_wi, in_spatial_stride);
          sum += _X[X_offset + in_i];
        }
        _Z[Z_offset + in_i] = sum / count;
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
  }
}

template <typename T>
void AvgPool2d(const Tensor<T>& X, Tensor<T>* Z, const PoolAux& aux) noexcept {
  int kernel_h = aux.kernel_sizes[0], kernel_w = aux.kernel_sizes[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int X_h = aux.Xspatials[0], X_w = aux.Xspatials[1];
  int Z_h = aux.Zspatials[0], Z_w = aux.Zspatials[1];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  int kernel_total_dim = aux.kernel_total_dim;
  int count_include_pad = aux.count_include_pad;
  const T* _X = X.data();
  T* _Z = Z->data();

  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
      for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
        int X_hi_start = Z_hi * stride_h - padding_h;
        int X_wi_start = Z_wi * stride_w - padding_w;
        int X_hi_end = std::min(X_hi_start + kernel_h, X_h);
        int X_wi_end = std::min(X_wi_start + kernel_w, X_w);
        X_hi_start = std::max(X_hi_start, 0);
        X_wi_start = std::max(X_wi_start, 0);

        int Z_offset = ComputeOffset(Z_hi, Z_wi, Z_w, in_spatial_stride);
        int no_pad_total_dim =
            (X_hi_end - X_hi_start) * (X_wi_end - X_wi_start);
        int count = count_include_pad ? kernel_total_dim : no_pad_total_dim;
        for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
          T sum = 0;
          for (int X_hi = X_hi_start; X_hi < X_hi_end; ++X_hi) {
            for (int X_wi = X_wi_start; X_wi < X_wi_end; ++X_wi) {
              int X_index =
                  ComputeOffset(X_hi, X_wi, X_w, aux.in_spatial_stride) + in_i;
              sum += _X[X_index];
            }
          }
          _Z[Z_offset + in_i] = sum / count;
        }
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
  }
}

template <typename T>
void AvgPool3d(const Tensor<T>& X, Tensor<T>* Z, const PoolAux& aux) noexcept {
  int kernel_d = aux.kernel_sizes[0], kernel_h = aux.kernel_sizes[1],
      kernel_w = aux.kernel_sizes[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_d = aux.Xspatials[0], X_h = aux.Xspatials[1], X_w = aux.Xspatials[2];
  int Z_d = aux.Zspatials[0], Z_h = aux.Zspatials[1], Z_w = aux.Zspatials[2];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  int kernel_total_dim = aux.kernel_total_dim;
  int count_include_pad = aux.count_include_pad;
  const T* _X = X.data();
  T* _Z = Z->data();

  int X_hw = X_h * X_w;
  int Z_hw = Z_h * Z_w;
  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
      for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
        for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
          int X_di_start = Z_di * stride_d - padding_d;
          int X_hi_start = Z_hi * stride_h - padding_h;
          int X_wi_start = Z_wi * stride_w - padding_w;
          int X_di_end = std::min(X_di_start + kernel_d, X_d);
          int X_hi_end = std::min(X_hi_start + kernel_h, X_h);
          int X_wi_end = std::min(X_wi_start + kernel_w, X_w);
          X_di_start = std::max(X_di_start, 0);
          X_hi_start = std::max(X_hi_start, 0);
          X_wi_start = std::max(X_wi_start, 0);

          int Z_offset =
              ComputeOffset(Z_di, Z_hi, Z_wi, Z_hw, Z_w, in_spatial_stride);
          int no_pad_total_dim = (X_di_end - X_di_start) *
                                 (X_hi_end - X_hi_start) *
                                 (X_wi_end - X_wi_start);
          int count = count_include_pad ? kernel_total_dim : no_pad_total_dim;
          for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
            T sum = 0;
            for (int X_di = X_di_start; X_di < X_di_end; ++X_di) {
              for (int X_hi = X_hi_start; X_hi < X_hi_end; ++X_hi) {
                for (int X_wi = X_wi_start; X_wi < X_wi_end; ++X_wi) {
                  int X_index = ComputeOffset(X_di, X_hi, X_wi, X_hw, X_w,
                                              in_spatial_stride) +
                                in_i;
                  sum += _X[X_index];
                }
              }
            }
            _Z[Z_offset + in_i] = sum / count;
          }
        }
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
  }
}

/************************************************************************/
/* PoolBackward */
/************************************************************************/
template <typename T>
void PoolBackward(const Tensor<T>& X, const Tensor<T>& Z, const Tensor<T>& gZ,
                  Tensor<T>* gX, const PoolAux& aux) noexcept {
  if (aux.pool_type == GraphNodePoolBase::POOL_TYPE_MAX) {
    if (aux.pool_rank == 1) {
      MaxPoolBackward1d(X, Z, gZ, gX, aux);
    } else if (aux.pool_rank == 2) {
      MaxPoolBackward2d(X, Z, gZ, gX, aux);
    } else {
      MaxPoolBackward3d(X, Z, gZ, gX, aux);
    }
  } else {
    if (aux.pool_rank == 1) {
      AvgPoolBackward1d(X, Z, gZ, gX, aux);
    } else if (aux.pool_rank == 2) {
      AvgPoolBackward2d(X, Z, gZ, gX, aux);
    } else {
      AvgPoolBackward3d(X, Z, gZ, gX, aux);
    }
  }
}

template <typename T>
void MaxPoolBackward1d(const Tensor<T>& X, const Tensor<T>& Z,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const PoolAux& aux) noexcept {
  int kernel_w = aux.kernel_sizes[0];
  int stride_w = aux.strides[0];
  int padding_w = aux.paddings[0];
  int dilation_w = aux.dilations[0];
  int X_w = aux.Xspatials[0];
  int Z_w = aux.Zspatials[0];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  const T* _X = X.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();

  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
      int X_wi_start = Z_wi * stride_w - padding_w;
      int X_wi_end =
          std::min(X_wi_start + DilateKernelSize(kernel_w, dilation_w), X_w);
      X_wi_start = FirstGE0(X_wi_start, dilation_w);

      int Z_offset = ComputeOffset(Z_wi, in_spatial_stride);
      for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
        int Z_index = Z_offset + in_i;
        T Z_value = _Z[Z_index];
        int X_max_index = -1;
        for (int X_wi = X_wi_start; X_wi < X_wi_end; X_wi += dilation_w) {
          int X_index = ComputeOffset(X_wi, in_spatial_stride) + in_i;
          if (_X[X_index] == Z_value) {
            X_max_index = X_index;
            break;
          }
        }
        _gX[X_max_index] += _gZ[Z_index];
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
    _gX += aux.X_out_spatial_stride;
    _gZ += aux.Z_out_spatial_stride;
  }
}

template <typename T>
void MaxPoolBackward2d(const Tensor<T>& X, const Tensor<T>& Z,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const PoolAux& aux) noexcept {
  int kernel_h = aux.kernel_sizes[0], kernel_w = aux.kernel_sizes[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int dilation_h = aux.dilations[0], dilation_w = aux.dilations[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int X_h = aux.Xspatials[0], X_w = aux.Xspatials[1];
  int Z_h = aux.Zspatials[0], Z_w = aux.Zspatials[1];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  const T* _X = X.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();

  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
      for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
        int X_hi_start = Z_hi * stride_h - padding_h;
        int X_wi_start = Z_wi * stride_w - padding_w;
        int X_hi_end =
            std::min(X_hi_start + DilateKernelSize(kernel_h, dilation_h), X_h);
        int X_wi_end =
            std::min(X_wi_start + DilateKernelSize(kernel_w, dilation_w), X_w);
        X_hi_start = FirstGE0(X_hi_start, dilation_h);
        X_wi_start = FirstGE0(X_wi_start, dilation_w);

        int Z_offset = ComputeOffset(Z_hi, Z_wi, Z_w, in_spatial_stride);
        for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
          int Z_index = Z_offset + in_i;
          T Z_value = _Z[Z_index];
          int X_max_index = -1;
          bool found = false;
          for (int X_hi = X_hi_start; X_hi < X_hi_end; X_hi += dilation_h) {
            for (int X_wi = X_wi_start; X_wi < X_wi_end; X_wi += dilation_w) {
              int X_index =
                  ComputeOffset(X_hi, X_wi, X_w, aux.in_spatial_stride) + in_i;
              if (_X[X_index] == Z_value) {
                X_max_index = X_index;
                found = true;
                break;
              }
            }
            if (found) {
              break;
            }
          }
          _gX[X_max_index] += _gZ[Z_index];
        }
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
    _gX += X_out_spatial_stride;
    _gZ += Z_out_spatial_stride;
  }
}

template <typename T>
void MaxPoolBackward3d(const Tensor<T>& X, const Tensor<T>& Z,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const PoolAux& aux) noexcept {
  int kernel_d = aux.kernel_sizes[0], kernel_h = aux.kernel_sizes[1],
      kernel_w = aux.kernel_sizes[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int dilation_d = aux.dilations[0], dilation_h = aux.dilations[1],
      dilation_w = aux.dilations[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_d = aux.Xspatials[0], X_h = aux.Xspatials[1], X_w = aux.Xspatials[2];
  int Z_d = aux.Zspatials[0], Z_h = aux.Zspatials[1], Z_w = aux.Zspatials[2];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  const T* _X = X.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();

  int X_hw = X_h * X_w;
  int Z_hw = Z_h * Z_w;
  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
      for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
        for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
          int X_di_start = Z_di * stride_d - padding_d;
          int X_hi_start = Z_hi * stride_h - padding_h;
          int X_wi_start = Z_wi * stride_w - padding_w;
          int X_di_end = std::min(
              X_di_start + DilateKernelSize(kernel_d, dilation_d), X_d);
          int X_hi_end = std::min(
              X_hi_start + DilateKernelSize(kernel_h, dilation_h), X_h);
          int X_wi_end = std::min(
              X_wi_start + DilateKernelSize(kernel_w, dilation_w), X_w);
          X_di_start = FirstGE0(X_di_start, dilation_d);
          X_hi_start = FirstGE0(X_hi_start, dilation_h);
          X_wi_start = FirstGE0(X_wi_start, dilation_w);

          int Z_offset =
              ComputeOffset(Z_di, Z_hi, Z_wi, Z_hw, Z_w, in_spatial_stride);
          for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
            int Z_index = Z_offset + in_i;
            T Z_value = _Z[Z_index];
            int X_max_index = -1;
            bool found = false;
            for (int X_di = X_di_start; X_di < X_di_end; X_di += dilation_d) {
              for (int X_hi = X_hi_start; X_hi < X_hi_end; X_hi += dilation_h) {
                for (int X_wi = X_wi_start; X_wi < X_wi_end;
                     X_wi += dilation_w) {
                  int X_index = ComputeOffset(X_di, X_hi, X_wi, X_hw, X_w,
                                              in_spatial_stride) +
                                in_i;
                  if (_X[X_index] == Z_value) {
                    X_max_index = X_index;
                    found = true;
                    break;
                  }
                }
                if (found) {
                  break;
                }
              }
              if (found) {
                break;
              }
            }
            _gX[X_max_index] += _gZ[Z_index];
          }
        }
      }
    }
    _X += X_out_spatial_stride;
    _Z += Z_out_spatial_stride;
    _gX += X_out_spatial_stride;
    _gZ += Z_out_spatial_stride;
  }
}

template <typename T>
void AvgPoolBackward1d(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const PoolAux& aux) noexcept {
  int kernel_w = aux.kernel_sizes[0];
  int stride_w = aux.strides[0];
  int padding_w = aux.paddings[0];
  int X_w = aux.Xspatials[0];
  int Z_w = aux.Zspatials[0];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  int kernel_total_dim = aux.kernel_total_dim;
  int count_include_pad = aux.count_include_pad;
  const T* _gZ = gZ.data();
  T* _gX = gX->data();

  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
      int X_wi_start = Z_wi * stride_w - padding_w;
      int X_wi_end = std::min(X_wi_start + kernel_w, X_w);
      X_wi_start = std::max(X_wi_start, 0);

      int Z_offset = ComputeOffset(Z_wi, in_spatial_stride);
      int no_pad_total_dim = (X_wi_end - X_wi_start);
      int count = count_include_pad ? kernel_total_dim : no_pad_total_dim;
      for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
        T grad = _gZ[Z_offset + in_i] / count;
        for (int X_wi = X_wi_start; X_wi < X_wi_end; ++X_wi) {
          _gX[ComputeOffset(X_wi, in_spatial_stride) + in_i] += grad;
        }
      }
    }
    _gX += X_out_spatial_stride;
    _gZ += Z_out_spatial_stride;
  }
}

template <typename T>
void AvgPoolBackward2d(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const PoolAux& aux) noexcept {
  int kernel_h = aux.kernel_sizes[0], kernel_w = aux.kernel_sizes[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int X_h = aux.Xspatials[0], X_w = aux.Xspatials[1];
  int Z_h = aux.Zspatials[0], Z_w = aux.Zspatials[1];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  int kernel_total_dim = aux.kernel_total_dim;
  int count_include_pad = aux.count_include_pad;
  const T* _gZ = gZ.data();
  T* _gX = gX->data();

  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
      for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
        int X_hi_start = Z_hi * stride_h - padding_h;
        int X_wi_start = Z_wi * stride_w - padding_w;
        int X_hi_end = std::min(X_hi_start + kernel_h, X_h);
        int X_wi_end = std::min(X_wi_start + kernel_w, X_w);
        X_hi_start = std::max(X_hi_start, 0);
        X_wi_start = std::max(X_wi_start, 0);

        int Z_offset = ComputeOffset(Z_hi, Z_wi, Z_w, in_spatial_stride);
        int count = count_include_pad
                        ? kernel_total_dim
                        : (X_hi_end - X_hi_start) * (X_wi_end - X_wi_start);
        for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
          T grad = _gZ[Z_offset + in_i] / count;
          for (int X_hi = X_hi_start; X_hi < X_hi_end; ++X_hi) {
            for (int X_wi = X_wi_start; X_wi < X_wi_end; ++X_wi) {
              int X_index =
                  ComputeOffset(X_hi, X_wi, X_w, aux.in_spatial_stride) + in_i;
              _gX[X_index] += grad;
            }
          }
        }
      }
    }
    _gX += X_out_spatial_stride;
    _gZ += Z_out_spatial_stride;
  }
}

template <typename T>
void AvgPoolBackward3d(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const PoolAux& aux) noexcept {
  int kernel_d = aux.kernel_sizes[0], kernel_h = aux.kernel_sizes[1],
      kernel_w = aux.kernel_sizes[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_d = aux.Xspatials[0], X_h = aux.Xspatials[1], X_w = aux.Xspatials[2];
  int Z_d = aux.Zspatials[0], Z_h = aux.Zspatials[1], Z_w = aux.Zspatials[2];
  int out_spatial_loop = aux.out_spatial_loop;
  int X_out_spatial_stride = aux.X_out_spatial_stride;
  int Z_out_spatial_stride = aux.Z_out_spatial_stride;
  int in_spatial_loop = aux.in_spatial_loop;
  int in_spatial_stride = aux.in_spatial_stride;
  int kernel_total_dim = aux.kernel_total_dim;
  int count_include_pad = aux.count_include_pad;
  const T* _gZ = gZ.data();
  T* _gX = gX->data();

  int X_hw = X_h * X_w;
  int Z_hw = Z_h * Z_w;
  for (int out_i = 0; out_i < out_spatial_loop; ++out_i) {
    for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
      for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
        for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
          int X_di_start = Z_di * stride_d - padding_d;
          int X_hi_start = Z_hi * stride_h - padding_h;
          int X_wi_start = Z_wi * stride_w - padding_w;
          int X_di_end = std::min(X_di_start + kernel_d, X_d);
          int X_hi_end = std::min(X_hi_start + kernel_h, X_h);
          int X_wi_end = std::min(X_wi_start + kernel_w, X_w);
          X_di_start = std::max(X_di_start, 0);
          X_hi_start = std::max(X_hi_start, 0);
          X_wi_start = std::max(X_wi_start, 0);

          int Z_offset =
              ComputeOffset(Z_di, Z_hi, Z_wi, Z_hw, Z_w, in_spatial_stride);
          int no_pad_total_dim = (X_di_end - X_di_start) *
                                 (X_hi_end - X_hi_start) *
                                 (X_wi_end - X_wi_start);
          int count = count_include_pad ? kernel_total_dim : no_pad_total_dim;
          for (int in_i = 0; in_i < in_spatial_loop; ++in_i) {
            T grad = _gZ[Z_offset + in_i] / count;
            for (int X_di = X_di_start; X_di < X_di_end; ++X_di) {
              for (int X_hi = X_hi_start; X_hi < X_hi_end; ++X_hi) {
                for (int X_wi = X_wi_start; X_wi < X_wi_end; ++X_wi) {
                  int X_index = ComputeOffset(X_di, X_hi, X_wi, X_hw, X_w,
                                              in_spatial_stride) +
                                in_i;
                  _gX[X_index] += grad;
                }
              }
            }
          }
        }
      }
    }
    _gX += X_out_spatial_stride;
    _gZ += Z_out_spatial_stride;
  }
}

}  // namespace

/************************************************************************/
/* PoolBase */
/************************************************************************/
GraphNodePoolBase::GraphNodePoolBase(std::string name, GraphNode* X,
                                     int pool_type, int pool_rank,
                                     int data_format, int kernel_size,
                                     int stride, int dilation, int padding_mode,
                                     int padding, int ceil_mode,
                                     int count_include_pad)
    : GraphNodePoolBase(std::move(name), X, pool_type, pool_rank, data_format,
                        std::vector<int>{kernel_size}, std::vector<int>{stride},
                        std::vector<int>{dilation}, padding_mode,
                        std::vector<int>{padding}, ceil_mode,
                        count_include_pad) {}

GraphNodePoolBase::GraphNodePoolBase(
    std::string name, GraphNode* X, int pool_type, int pool_rank,
    int data_format, std::vector<int> kernel_sizes, std::vector<int> strides,
    std::vector<int> dilations, int padding_mode, std::vector<int> paddings,
    int ceil_mode, int count_include_pad)
    : GraphNodeUnaryBase(std::move(name), X),
      pool_type_(pool_type),
      pool_rank_(pool_rank),
      data_format_(data_format),
      kernel_sizes_(std::move(kernel_sizes)),
      strides_(std::move(strides)),
      dilations_(std::move(dilations)),
      padding_mode_(padding_mode),
      paddings_(std::move(paddings)),
      ceil_mode_(ceil_mode),
      count_include_pad_(count_include_pad) {
  DXCHECK_THROW(PoolCheckAttr(pool_rank_, data_format_, kernel_sizes_, strides_,
                              dilations_, padding_mode_, paddings_,
                              ceil_mode_));
  if (!X->shape().empty()) {
    (void)PoolInferShape(X->shape(), pool_type_, pool_rank_, data_format_,
                         kernel_sizes_, strides_, dilations_, padding_mode_,
                         paddings_, ceil_mode_, count_include_pad_, &shape_);
  }
}

class OpPoolBase : public OpUnaryBase {
 protected:
  PoolAux aux_;

 public:
  const Shape& InferShape() override {
    const GraphNodePoolBase* node = (const GraphNodePoolBase*)node_;  // NOLINT
    DXCHECK_THROW(
        PoolPrepare(X_->shape(), node->pool_type(), node->pool_rank(),
                    node->data_format(), node->kernel_sizes(), node->strides(),
                    node->dilations(), node->padding_mode(), node->paddings(),
                    node->ceil_mode(), node->count_include_pad(), &aux_));
    return aux_.Z;
  }

  void Forward() override { Pool(*X_, Z_, aux_); }

  void Backward() override {
    if (gZ_) {
      PoolBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

/************************************************************************/
/* MaxPool1d */
/************************************************************************/
MaxPool1dNode::MaxPool1dNode(std::string name, GraphNode* X, int data_format,
                             int kernel_size, int stride, int dilation,
                             int padding, int ceil_mode)
    : MaxPool1dNode(std::move(name), X, data_format, kernel_size, stride,
                    dilation, PADDING_MODE_USE_PADDINGS, padding, ceil_mode) {}

MaxPool1dNode::MaxPool1dNode(std::string name, GraphNode* X, int data_format,
                             int kernel_size, int stride, int dilation,
                             int padding_mode, int padding, int ceil_mode)
    : GraphNodePoolBase(std::move(name), X, POOL_TYPE_MAX, 1, data_format,
                        kernel_size, stride, dilation, padding_mode, padding,
                        ceil_mode, 0) {}

class MaxPool1dOp : public OpPoolBase {
 public:
  DEFINE_OP_LIKE(MaxPool1d);
};

GRAPH_NODE_OP_REGISTER(MaxPool1d);

/************************************************************************/
/* MaxPool2d */
/************************************************************************/
MaxPool2dNode::MaxPool2dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides,
                             std::vector<int> dilations,
                             std::vector<int> paddings, int ceil_mode)
    : MaxPool2dNode(std::move(name), X, data_format, std::move(kernel_sizes),
                    std::move(strides), std::move(dilations),
                    PADDING_MODE_USE_PADDINGS, std::move(paddings), ceil_mode) {
}

MaxPool2dNode::MaxPool2dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides,
                             std::vector<int> dilations, int padding_mode,
                             std::vector<int> paddings, int ceil_mode)
    : GraphNodePoolBase(std::move(name), X, POOL_TYPE_MAX, 2, data_format,
                        std::move(kernel_sizes), std::move(strides),
                        std::move(dilations), padding_mode, std::move(paddings),
                        ceil_mode, 0) {}

class MaxPool2dOp : public OpPoolBase {
 public:
  DEFINE_OP_LIKE(MaxPool2d);
};

GRAPH_NODE_OP_REGISTER(MaxPool2d);

/************************************************************************/
/* MaxPool3d */
/************************************************************************/
MaxPool3dNode::MaxPool3dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides,
                             std::vector<int> dilations,
                             std::vector<int> paddings, int ceil_mode)
    : MaxPool3dNode(std::move(name), X, data_format, std::move(kernel_sizes),
                    std::move(strides), std::move(dilations),
                    PADDING_MODE_USE_PADDINGS, std::move(paddings), ceil_mode) {
}

MaxPool3dNode::MaxPool3dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides,
                             std::vector<int> dilations, int padding_mode,
                             std::vector<int> paddings, int ceil_mode)
    : GraphNodePoolBase(std::move(name), X, POOL_TYPE_MAX, 3, data_format,
                        std::move(kernel_sizes), std::move(strides),
                        std::move(dilations), padding_mode, std::move(paddings),
                        ceil_mode, 0) {}

class MaxPool3dOp : public OpPoolBase {
 public:
  DEFINE_OP_LIKE(MaxPool3d);
};

GRAPH_NODE_OP_REGISTER(MaxPool3d);

/************************************************************************/
/* AvgPool1d */
/************************************************************************/
AvgPool1dNode::AvgPool1dNode(std::string name, GraphNode* X, int data_format,
                             int kernel_size, int stride, int padding,
                             int ceil_mode, int count_include_pad)
    : AvgPool1dNode(std::move(name), X, data_format, kernel_size, stride,
                    PADDING_MODE_USE_PADDINGS, padding, ceil_mode,
                    count_include_pad) {}

AvgPool1dNode::AvgPool1dNode(std::string name, GraphNode* X, int data_format,
                             int kernel_size, int stride, int padding_mode,
                             int padding, int ceil_mode, int count_include_pad)
    : GraphNodePoolBase(std::move(name), X, POOL_TYPE_AVG, 1, data_format,
                        kernel_size, stride, 1, padding_mode, padding,
                        ceil_mode, count_include_pad) {}

class AvgPool1dOp : public OpPoolBase {
 public:
  DEFINE_OP_LIKE(AvgPool1dOp);
};

GRAPH_NODE_OP_REGISTER(AvgPool1d);

/************************************************************************/
/* AvgPool2d */
/************************************************************************/
AvgPool2dNode::AvgPool2dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides,
                             std::vector<int> paddings, int ceil_mode,
                             int count_include_pad)
    : AvgPool2dNode(std::move(name), X, data_format, std::move(kernel_sizes),
                    std::move(strides), PADDING_MODE_USE_PADDINGS,
                    std::move(paddings), ceil_mode, count_include_pad) {}

AvgPool2dNode::AvgPool2dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides, int padding_mode,
                             std::vector<int> paddings, int ceil_mode,
                             int count_include_pad)
    : GraphNodePoolBase(std::move(name), X, POOL_TYPE_AVG, 2, data_format,
                        std::move(kernel_sizes), std::move(strides),
                        std::vector<int>(2, 1), padding_mode,
                        std::move(paddings), ceil_mode, count_include_pad) {}

class AvgPool2dOp : public OpPoolBase {
 public:
  DEFINE_OP_LIKE(AvgPool2dOp);
};

GRAPH_NODE_OP_REGISTER(AvgPool2d);

/************************************************************************/
/* AvgPool3d */
/************************************************************************/
AvgPool3dNode::AvgPool3dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides,
                             std::vector<int> paddings, int ceil_mode,
                             int count_include_pad)
    : AvgPool3dNode(std::move(name), X, data_format, std::move(kernel_sizes),
                    std::move(strides), PADDING_MODE_USE_PADDINGS,
                    std::move(paddings), ceil_mode, count_include_pad) {}

AvgPool3dNode::AvgPool3dNode(std::string name, GraphNode* X, int data_format,
                             std::vector<int> kernel_sizes,
                             std::vector<int> strides, int padding_mode,
                             std::vector<int> paddings, int ceil_mode,
                             int count_include_pad)
    : GraphNodePoolBase(std::move(name), X, POOL_TYPE_AVG, 3, data_format,
                        std::move(kernel_sizes), std::move(strides),
                        std::vector<int>(3, 1), padding_mode,
                        std::move(paddings), ceil_mode, count_include_pad) {}

class AvgPool3dOp : public OpPoolBase {
 public:
  DEFINE_OP_LIKE(AvgPool3dOp);
};

GRAPH_NODE_OP_REGISTER(AvgPool3d);

}  // namespace deepx_core
