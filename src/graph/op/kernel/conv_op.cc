// Copyright 2019 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

using ai_t = std::array<int, 3>;

/************************************************************************/
/* Im2col & Col2im */
/************************************************************************/
struct Im2colAux {
  int conv_rank = 0;
  ai_t strides;
  ai_t dilations;
  ai_t paddings;
  ai_t X;
  ai_t K;
  ai_t Z;
  int in_channel = 0;
};

void Im2colPrepare(int conv_rank, const ai_t& strides, const ai_t& dilations,
                   const ai_t& paddings, const ai_t& X, const ai_t& K,
                   const ai_t& Z, int in_channel, Im2colAux* aux) noexcept {
  aux->conv_rank = conv_rank;
  for (int i = 0; i < conv_rank; ++i) {
    aux->strides[i] = strides[i];
    aux->dilations[i] = dilations[i];
    aux->paddings[i] = paddings[i];
    aux->X[i] = X[i];
    aux->K[i] = K[i];
    aux->Z[i] = Z[i];
  }
  aux->in_channel = in_channel;
}

bool GE0AndLT(int a, int b) noexcept { return a >= 0 && a < b; }

int ComputeOffset(int wi, int stride) noexcept { return wi * stride; }

int ComputeOffset(int hi, int wi, int w, int stride) noexcept {
  return (hi * w + wi) * stride;
}

int ComputeOffset(int di, int hi, int wi, int hw, int w, int stride) noexcept {
  return (di * hw + hi * w + wi) * stride;
}

template <typename T>
void Im2colNCX(const T* in, T* out, const Im2colAux& aux) noexcept {
  if (aux.conv_rank == 1) {
    Im2colNCW(in, out, aux);
  } else if (aux.conv_rank == 2) {
    Im2colNCHW(in, out, aux);
  } else {
    Im2colNCDHW(in, out, aux);
  }
}

template <typename T>
void Im2colNCW(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_w = aux.X[0];
  int K_w = aux.K[0];
  int Z_w = aux.Z[0];
  int stride_w = aux.strides[0];
  int dilation_w = aux.dilations[0];
  int padding_w = aux.paddings[0];

  for (int i = 0; i < in_channel; ++i) {
    for (int K_wi = 0; K_wi < K_w; ++K_wi) {
      int X_wi = -padding_w + K_wi * dilation_w;
      for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
        if (GE0AndLT(X_wi, X_w)) {
          *out++ = *(in + X_wi);
        } else {
          *out++ = 0;
        }
        X_wi += stride_w;
      }
    }
    in += X_w;
  }
}

template <typename T>
void Im2colNCHW(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_h = aux.X[0], X_w = aux.X[1];
  int K_h = aux.K[0], K_w = aux.K[1];
  int Z_h = aux.Z[0], Z_w = aux.Z[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int dilation_h = aux.dilations[0], dilation_w = aux.dilations[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int X_hw = X_h * X_w;

  for (int i = 0; i < in_channel; ++i) {
    for (int K_hi = 0; K_hi < K_h; ++K_hi) {
      for (int K_wi = 0; K_wi < K_w; ++K_wi) {
        int X_hi = -padding_h + K_hi * dilation_h;
        for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
          bool hi_not_padding = GE0AndLT(X_hi, X_h);
          int X_hi_offset = X_hi * X_w;
          int X_wi = -padding_w + K_wi * dilation_w;
          for (int col_wi = 0; col_wi < Z_w; ++col_wi) {
            if (hi_not_padding && GE0AndLT(X_wi, X_w)) {
              *out++ = *(in + X_hi_offset + X_wi);
            } else {
              *out++ = 0;
            }
            X_wi += stride_w;
          }
          X_hi += stride_h;
        }
      }
    }
    in += X_hw;
  }
}

template <typename T>
void Im2colNCDHW(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_d = aux.X[0], X_h = aux.X[1], X_w = aux.X[2];
  int K_d = aux.K[0], K_h = aux.K[1], K_w = aux.K[2];
  int Z_d = aux.Z[0], Z_h = aux.Z[1], Z_w = aux.Z[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int dilation_d = aux.dilations[0], dilation_h = aux.dilations[1],
      dilation_w = aux.dilations[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_hw = X_h * X_w;
  int X_dhw = X_d * X_hw;

  for (int i = 0; i < in_channel; ++i) {
    for (int K_di = 0; K_di < K_d; ++K_di) {
      for (int K_hi = 0; K_hi < K_h; ++K_hi) {
        for (int K_wi = 0; K_wi < K_w; ++K_wi) {
          int X_di = -padding_d + K_di * dilation_d;
          for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
            bool di_not_padding = GE0AndLT(X_di, X_d);
            int X_di_offset = X_di * X_hw;
            int X_hi = -padding_h + K_hi * dilation_h;
            for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
              bool hi_not_padding = di_not_padding && GE0AndLT(X_hi, X_h);
              int X_hi_offset = X_hi * X_w;
              int X_wi = -padding_w + K_wi * dilation_w;
              for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
                if (hi_not_padding && GE0AndLT(X_wi, X_w)) {
                  *out++ = *(in + X_di_offset + X_hi_offset + X_wi);
                } else {
                  *out++ = 0;
                }
                X_wi += stride_w;
              }
              X_hi += stride_h;
            }
            X_di += stride_d;
          }
        }
      }
    }
    in += X_dhw;
  }
}

template <typename T>
void Col2imNCX(const T* in, T* out, const Im2colAux& aux) noexcept {
  if (aux.conv_rank == 1) {
    Col2imNCW(in, out, aux);
  } else if (aux.conv_rank == 2) {
    Col2imNCHW(in, out, aux);
  } else {
    Col2imNCDHW(in, out, aux);
  }
}

template <typename T>
void Col2imNCW(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_w = aux.X[0];
  int K_w = aux.K[0];
  int Z_w = aux.Z[0];
  int stride_w = aux.strides[0];
  int dilation_w = aux.dilations[0];
  int padding_w = aux.paddings[0];

  for (int i = 0; i < in_channel; ++i) {
    for (int K_wi = 0; K_wi < K_w; ++K_wi) {
      int image_wi = -padding_w + K_wi * dilation_w;
      for (int col_wi = 0; col_wi < Z_w; ++col_wi) {
        if (GE0AndLT(image_wi, X_w)) {
          *(out + image_wi) += *in;
        }
        ++in;
        image_wi += stride_w;
      }
    }
    out += X_w;
  }
}

template <typename T>
void Col2imNCHW(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_h = aux.X[0], X_w = aux.X[1];
  int K_h = aux.K[0], K_w = aux.K[1];
  int Z_h = aux.Z[0], Z_w = aux.Z[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int dilation_h = aux.dilations[0], dilation_w = aux.dilations[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int X_hw = X_h * X_w;

  for (int i = 0; i < in_channel; ++i) {
    for (int K_hi = 0; K_hi < K_h; ++K_hi) {
      for (int K_wi = 0; K_wi < K_w; ++K_wi) {
        int X_hi = -padding_h + K_hi * dilation_h;
        for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
          bool hi_not_padding = GE0AndLT(X_hi, X_h);
          int X_hi_offset = X_hi * X_w;
          int X_wi = -padding_w + K_wi * dilation_w;
          for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
            if (hi_not_padding && GE0AndLT(X_wi, X_w)) {
              *(out + X_hi_offset + X_wi) += *in;
            }
            ++in;
            X_wi += stride_w;
          }
          X_hi += stride_h;
        }
      }
    }
    out += X_hw;
  }
}

template <typename T>
void Col2imNCDHW(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_d = aux.X[0], X_h = aux.X[1], X_w = aux.X[2];
  int K_d = aux.K[0], K_h = aux.K[1], K_w = aux.K[2];
  int Z_d = aux.Z[0], Z_h = aux.Z[1], Z_w = aux.Z[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int dilation_d = aux.dilations[0], dilation_h = aux.dilations[1],
      dilation_w = aux.dilations[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_hw = X_h * X_w;
  int Z_dhw = X_d * X_hw;

  for (int i = 0; i < in_channel; ++i) {
    for (int K_di = 0; K_di < K_d; ++K_di) {
      for (int K_hi = 0; K_hi < K_h; ++K_hi) {
        for (int K_wi = 0; K_wi < K_w; ++K_wi) {
          int X_di = -padding_d + K_di * dilation_d;
          for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
            bool di_not_padding = GE0AndLT(X_di, X_d);
            int X_di_offset = X_di * X_hw;
            int X_hi = -padding_h + K_hi * dilation_h;
            for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
              bool hi_not_padding = di_not_padding && GE0AndLT(X_hi, X_h);
              int X_hi_offset = X_hi * X_w;
              int X_wi = -padding_w + K_wi * dilation_w;
              for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
                if (hi_not_padding && GE0AndLT(X_wi, X_w)) {
                  *(out + X_di_offset + X_hi_offset + X_wi) += *in;
                }
                ++in;
                X_wi += stride_w;
              }
              X_hi += stride_h;
            }
            X_di += stride_d;
          }
        }
      }
    }
    out += Z_dhw;
  }
}

template <typename T>
void Im2colNXC(const T* in, T* out, const Im2colAux& aux) noexcept {
  if (aux.conv_rank == 1) {
    Im2colNWC(in, out, aux);
  } else if (aux.conv_rank == 2) {
    Im2colNHWC(in, out, aux);
  } else {
    Im2colNDHWC(in, out, aux);
  }
}

template <typename T>
void Im2colNWC(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_w = aux.X[0];
  int K_w = aux.K[0];
  int Z_w = aux.Z[0];
  int stride_w = aux.strides[0];
  int dilation_w = aux.dilations[0];
  int padding_w = aux.paddings[0];
  int row_stride = K_w * in_channel;

  for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
    int row_offset = ComputeOffset(Z_wi, row_stride);
    int X_wi_start = Z_wi * stride_w - padding_w;
    for (int K_wi = 0; K_wi < K_w; ++K_wi) {
      int col_offset = ComputeOffset(K_wi, in_channel);
      int X_wi = X_wi_start + K_wi * dilation_w;
      if (GE0AndLT(X_wi, X_w)) {
        int in_offset = ComputeOffset(X_wi, in_channel);
        memcpy(out + row_offset + col_offset, in + in_offset,
               in_channel * sizeof(T));
      } else {
        memset(out + row_offset + col_offset, 0, in_channel * sizeof(T));
      }
    }
  }
}

template <typename T>
void Im2colNHWC(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_h = aux.X[0], X_w = aux.X[1];
  int K_h = aux.K[0], K_w = aux.K[1];
  int Z_h = aux.Z[0], Z_w = aux.Z[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int dilation_h = aux.dilations[0], dilation_w = aux.dilations[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int row_stride = K_h * K_w * in_channel;

  for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
    for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
      int row_offset = ComputeOffset(Z_hi, Z_wi, Z_w, row_stride);
      int X_hi_start = Z_hi * stride_h - padding_h;
      int X_wi_start = Z_wi * stride_w - padding_w;
      for (int K_hi = 0; K_hi < K_h; ++K_hi) {
        int X_hi = X_hi_start + K_hi * dilation_h;
        if (GE0AndLT(X_hi, X_h)) {
          for (int K_wi = 0; K_wi < K_w; ++K_wi) {
            int col_offset = ComputeOffset(K_hi, K_wi, K_w, in_channel);
            int X_wi = X_wi_start + K_wi * dilation_w;
            if (GE0AndLT(X_wi, X_w)) {
              int in_offset = ComputeOffset(X_hi, X_wi, X_w, in_channel);
              memcpy(out + row_offset + col_offset, in + in_offset,
                     in_channel * sizeof(T));
            } else {
              memset(out + row_offset + col_offset, 0, in_channel * sizeof(T));
            }
          }
        } else {
          int col_offset = K_hi * K_w * in_channel;
          memset(out + row_offset + col_offset, 0,
                 K_w * in_channel * sizeof(T));
        }
      }
    }
  }
}

template <typename T>
void Im2colNDHWC(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_d = aux.X[0], X_h = aux.X[1], X_w = aux.X[2];
  int K_d = aux.K[0], K_h = aux.K[1], K_w = aux.K[2];
  int Z_d = aux.Z[0], Z_h = aux.Z[1], Z_w = aux.Z[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int dilation_d = aux.dilations[0], dilation_h = aux.dilations[1],
      dilation_w = aux.dilations[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_hw = X_h * X_w;
  int K_hw = K_h * K_w;
  int Z_hw = Z_h * Z_w;
  int row_stride = K_d * K_hw * in_channel;

  for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
    for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
      for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
        int row_offset = ComputeOffset(Z_di, Z_hi, Z_wi, Z_hw, Z_w, row_stride);
        int X_di_start = Z_di * stride_d - padding_d;
        int X_hi_start = Z_hi * stride_h - padding_h;
        int X_wi_start = Z_wi * stride_w - padding_w;
        for (int K_di = 0; K_di < K_d; ++K_di) {
          int X_di = X_di_start + K_di * dilation_d;
          if (GE0AndLT(X_di, X_d)) {
            for (int K_hi = 0; K_hi < K_h; ++K_hi) {
              int X_hi = X_hi_start + K_hi * dilation_h;
              if (GE0AndLT(X_hi, X_h)) {
                for (int K_wi = 0; K_wi < K_w; ++K_wi) {
                  int col_offset =
                      ComputeOffset(K_di, K_hi, K_wi, K_hw, K_w, in_channel);
                  int X_wi = X_wi_start + K_wi * dilation_w;
                  if (GE0AndLT(X_wi, X_w)) {
                    int in_offset =
                        ComputeOffset(X_di, X_hi, X_wi, X_hw, X_w, in_channel);
                    memcpy(out + row_offset + col_offset, in + in_offset,
                           in_channel * sizeof(T));
                  } else {
                    memset(out + row_offset + col_offset, 0,
                           in_channel * sizeof(T));
                  }
                }
              } else {
                int col_offset = (K_di * K_hw + K_hi * K_w) * in_channel;
                memset(out + row_offset + col_offset, 0,
                       K_w * in_channel * sizeof(T));
              }
            }
          } else {
            int col_offset = K_di * K_hw * in_channel;
            memset(out + row_offset + col_offset, 0,
                   K_hw * in_channel * sizeof(T));
          }
        }
      }
    }
  }
}

template <typename T>
void Col2imNXC(const T* in, T* out, const Im2colAux& aux) noexcept {
  if (aux.conv_rank == 1) {
    Col2imNWC(in, out, aux);
  } else if (aux.conv_rank == 2) {
    Col2imNHWC(in, out, aux);
  } else {
    Col2imNDHWC(in, out, aux);
  }
}

template <typename T>
void Col2imNWC(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_w = aux.X[0];
  int K_w = aux.K[0];
  int Z_w = aux.Z[0];
  int stride_w = aux.strides[0];
  int dilation_w = aux.dilations[0];
  int padding_w = aux.paddings[0];
  int row_stride = K_w * in_channel;

  for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
    int row_offset = ComputeOffset(Z_wi, row_stride);
    int X_wi_start = Z_wi * stride_w - padding_w;
    for (int K_wi = 0; K_wi < K_w; ++K_wi) {
      int X_wi = X_wi_start + K_wi * dilation_w;
      if (GE0AndLT(X_wi, X_w)) {
        int out_offset = ComputeOffset(X_wi, in_channel);
        int col_offset = ComputeOffset(K_wi, in_channel);
        for (int i = 0; i < in_channel; ++i) {
          *(out + out_offset + i) += *(in + row_offset + col_offset + i);
        }
      }
    }
  }
}

template <typename T>
void Col2imNHWC(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_h = aux.X[0], X_w = aux.X[1];
  int K_h = aux.K[0], K_w = aux.K[1];
  int Z_h = aux.Z[0], Z_w = aux.Z[1];
  int stride_h = aux.strides[0], stride_w = aux.strides[1];
  int dilation_h = aux.dilations[0], dilation_w = aux.dilations[1];
  int padding_h = aux.paddings[0], padding_w = aux.paddings[1];
  int row_stride = K_h * K_w * in_channel;

  for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
    for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
      int row_offset = ComputeOffset(Z_hi, Z_wi, Z_w, row_stride);
      int X_hi_start = Z_hi * stride_h - padding_h;
      int X_wi_start = Z_wi * stride_w - padding_w;
      for (int K_hi = 0; K_hi < K_h; ++K_hi) {
        int X_hi = X_hi_start + K_hi * dilation_h;
        if (GE0AndLT(X_hi, X_h)) {
          for (int K_wi = 0; K_wi < K_w; ++K_wi) {
            int X_wi = X_wi_start + K_wi * dilation_w;
            if (GE0AndLT(X_wi, X_w)) {
              int out_offset = ComputeOffset(X_hi, X_wi, X_w, in_channel);
              int col_offset = ComputeOffset(K_hi, K_wi, K_w, in_channel);
              for (int i = 0; i < in_channel; ++i) {
                *(out + out_offset + i) += *(in + row_offset + col_offset + i);
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void Col2imNDHWC(const T* in, T* out, const Im2colAux& aux) noexcept {
  int in_channel = aux.in_channel;
  int X_d = aux.X[0], X_h = aux.X[1], X_w = aux.X[2];
  int K_d = aux.K[0], K_h = aux.K[1], K_w = aux.K[2];
  int Z_d = aux.Z[0], Z_h = aux.Z[1], Z_w = aux.Z[2];
  int stride_d = aux.strides[0], stride_h = aux.strides[1],
      stride_w = aux.strides[2];
  int dilation_d = aux.dilations[0], dilation_h = aux.dilations[1],
      dilation_w = aux.dilations[2];
  int padding_d = aux.paddings[0], padding_h = aux.paddings[1],
      padding_w = aux.paddings[2];
  int X_hw = X_h * X_w;
  int K_hw = K_h * K_w;
  int Z_hw = Z_h * Z_w;
  int row_stride = K_d * K_hw * in_channel;

  for (int Z_di = 0; Z_di < Z_d; ++Z_di) {
    for (int Z_hi = 0; Z_hi < Z_h; ++Z_hi) {
      for (int Z_wi = 0; Z_wi < Z_w; ++Z_wi) {
        int row_offset = ComputeOffset(Z_di, Z_hi, Z_wi, Z_hw, Z_w, row_stride);
        int X_di_start = Z_di * stride_d - padding_d;
        int X_hi_start = Z_hi * stride_h - padding_h;
        int X_wi_start = Z_wi * stride_w - padding_w;
        for (int K_di = 0; K_di < K_d; ++K_di) {
          int X_di = X_di_start + K_di * dilation_d;
          if (GE0AndLT(X_di, X_d)) {
            for (int K_hi = 0; K_hi < K_h; ++K_hi) {
              int X_hi = X_hi_start + K_hi * dilation_h;
              if (GE0AndLT(X_hi, X_h)) {
                for (int K_wi = 0; K_wi < K_w; ++K_wi) {
                  int X_wi = X_wi_start + K_wi * dilation_w;
                  if (GE0AndLT(X_wi, X_w)) {
                    int out_offset =
                        ComputeOffset(X_di, X_hi, X_wi, X_hw, X_w, in_channel);
                    int col_offset =
                        ComputeOffset(K_di, K_hi, K_wi, K_hw, K_w, in_channel);
                    for (int i = 0; i < in_channel; ++i) {
                      *(out + out_offset + i) +=
                          *(in + row_offset + col_offset + i);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/************************************************************************/
/* Conv */
/************************************************************************/
struct ConvAux {
  Shape Z;
  int batch = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  int ncx = 0;
  int X_batch_stride = 0;
  int K_spatial_total_dim = 0;
  int Z_spatial_total_dim = 0;
  int im2col = 0;
  Im2colAux im2col_aux;
};

template <typename T>
struct ConvMutableAux {
  Tensor<T> buf;
};

bool ConvCheckAttr(int conv_rank, int data_format,
                   const std::vector<int>& strides,
                   const std::vector<int>& dilations, int padding_mode,
                   const std::vector<int>& paddings) noexcept {
  if (conv_rank != 1 && conv_rank != 2 && conv_rank != 3) {
    DXERROR("Invalid conv_rank: conv_rank %d must be 1, 2 or 3.", conv_rank);
    return false;
  }

  if (conv_rank == 1) {
    if (data_format != GraphNodeConvBase::DATA_FORMAT_NCW &&
        data_format != GraphNodeConvBase::DATA_FORMAT_NWC) {
      DXERROR(
          "Invalid data_format: data_format must be DATA_FORMAT_NCW or "
          "DATA_FORMAT_NWC.");
      return false;
    }
  } else if (conv_rank == 2) {
    if (data_format != GraphNodeConvBase::DATA_FORMAT_NCHW &&
        data_format != GraphNodeConvBase::DATA_FORMAT_NHWC) {
      DXERROR(
          "Invalid data_format: data_format must be DATA_FORMAT_NCHW or "
          "DATA_FORMAT_NHWC.");
      return false;
    }
  } else {
    if (data_format != GraphNodeConvBase::DATA_FORMAT_NCDHW &&
        data_format != GraphNodeConvBase::DATA_FORMAT_NDHWC) {
      DXERROR(
          "Invalid data_format: data_format must be DATA_FORMAT_NCDHW or "
          "DATA_FORMAT_NDHWC.");
      return false;
    }
  }

  if ((int)strides.size() != conv_rank) {
    DXERROR("Invalid strides: size of strides %d must be %d.",
            (int)strides.size(), conv_rank);
    return false;
  }

  for (int stride : strides) {
    if (stride <= 0) {
      DXERROR("Invalid strides: stride %d must be positive.", stride);
      return false;
    }
  }

  if ((int)dilations.size() != conv_rank) {
    DXERROR("Invalid dilations: size of dilations %d must be %d.",
            (int)dilations.size(), conv_rank);
    return false;
  }

  for (int dilation : dilations) {
    if (dilation <= 0) {
      DXERROR("Invalid dilations: dilation %d must be positive.", dilation);
      return false;
    }
  }

  if (padding_mode != GraphNodeConvBase::PADDING_MODE_SAME &&
      padding_mode != GraphNodeConvBase::PADDING_MODE_VALID &&
      padding_mode != GraphNodeConvBase::PADDING_MODE_USE_PADDINGS) {
    DXERROR(
        "Invalid padding_mode: padding_mode must be PADDING_MODE_SAME, "
        "PADDING_MODE_VALID or PADDING_MODE_USE_PADDINGS.");
    return false;
  }

  if (padding_mode == GraphNodeConvBase::PADDING_MODE_USE_PADDINGS) {
    if ((int)paddings.size() != conv_rank) {
      DXERROR("Invalid paddings: size of paddings %d must be %d.",
              (int)paddings.size(), conv_rank);
      return false;
    }

    for (int padding : paddings) {
      if (padding < 0) {
        DXERROR("Invalid paddings: padding %d must be non-negative.", padding);
        return false;
      }
    }
  }

  return true;
}

int DilateKernelSize(int kernel_size, int dilation) noexcept {
  return (kernel_size - 1) * dilation + 1;
}

bool ConvPrepare(const Shape& X, const Shape& K, int conv_rank, int data_format,
                 const std::vector<int>& strides,
                 const std::vector<int>& dilations, int padding_mode,
                 const std::vector<int>& paddings, ConvAux* aux) noexcept {
  int Xrank = X.rank();
  if (Xrank != conv_rank + 2) {
    DXERROR("Invalid X: rank of X %d must be %d.", Xrank, conv_rank + 2);
    return false;
  }

  int Krank = K.rank();
  if (Krank != conv_rank + 2) {
    DXERROR("Invalid K: rank of K %d must be %d.", Krank, conv_rank + 2);
    return false;
  }

  int ncx = data_format == GraphNodeConvBase::DATA_FORMAT_NCW ||
            data_format == GraphNodeConvBase::DATA_FORMAT_NCHW ||
            data_format == GraphNodeConvBase::DATA_FORMAT_NCDHW;
  int X_in_channel_axis = ncx ? 1 : Xrank - 1;
  int K_in_channel_axis = ncx ? 1 : Xrank - 2;
  if (X[X_in_channel_axis] != K[K_in_channel_axis]) {
    DXERROR("Invalid X and K: inconsistent in_channel dim %d vs %d.",
            X[X_in_channel_axis], K[K_in_channel_axis]);
    return false;
  }

  ai_t Xspatials, Kspatials;
  int X_spatial_begin = ncx ? 2 : 1;
  int K_spatial_begin = ncx ? 2 : 0;
  int X_spatial_total_dim = 1;
  int K_spatial_total_dim = 1;
  for (int i = 0; i < conv_rank; ++i) {
    Xspatials[i] = X[X_spatial_begin + i];
    Kspatials[i] = K[K_spatial_begin + i];
    X_spatial_total_dim *= Xspatials[i];
    K_spatial_total_dim *= Kspatials[i];
  }

  auto z_dim =
      [](int x, int k, int stride, int dilation, int padding) noexcept {
    return (x + 2 * padding - DilateKernelSize(k, dilation)) / stride + 1;
  };

  ai_t Zspatials;
  ai_t _paddings;
  if (padding_mode == GraphNodeConvBase::PADDING_MODE_SAME) {
    for (int i = 0; i < conv_rank; ++i) {
      Zspatials[i] = (Xspatials[i] - 1) / strides[i] + 1;
      _paddings[i] =
          ((Zspatials[i] - 1) * strides[i] +
           DilateKernelSize(Kspatials[i], dilations[i]) - Xspatials[i]) /
          2;
    }
  } else {
    if (padding_mode == GraphNodeConvBase::PADDING_MODE_VALID) {
      for (int i = 0; i < conv_rank; ++i) {
        _paddings[i] = 0;
        Zspatials[i] = z_dim(Xspatials[i], Kspatials[i], strides[i],
                             dilations[i], _paddings[i]);
      }
    } else {
      for (int i = 0; i < conv_rank; ++i) {
        _paddings[i] = paddings[i];
        Zspatials[i] = z_dim(Xspatials[i], Kspatials[i], strides[i],
                             dilations[i], _paddings[i]);
      }
    }

    for (int i = 0; i < conv_rank; ++i) {
      if (Zspatials[i] <= 0) {
        DXERROR("Invalid X, K, strides, dilations and paddings combination.");
        return false;
      }
    }
  }

  int Z_spatial_total_dim = 1;
  for (int i = 0; i < conv_rank; ++i) {
    Z_spatial_total_dim *= Zspatials[i];
  }

  int batch = X[0];
  int in_channel = X[X_in_channel_axis];
  int out_channel = ncx ? K[0] : K[Krank - 1];

  int Zdims[SHAPE_MAX_RANK];
  int Zrank = 0;
  Zdims[Zrank++] = batch;
  if (ncx) {
    Zdims[Zrank++] = out_channel;
    for (int i = 0; i < conv_rank; ++i) {
      Zdims[Zrank++] = Zspatials[i];
    }
  } else {
    for (int i = 0; i < conv_rank; ++i) {
      Zdims[Zrank++] = Zspatials[i];
    }
    Zdims[Zrank++] = out_channel;
  }

  int im2col = 0;
  for (int i = 0; i < conv_rank; ++i) {
    if (Kspatials[i] != 1 || strides[i] != 1 || _paddings[i] != 0) {
      im2col = 1;
      break;
    }
  }

  aux->Z.assign(&Zdims[0], &Zdims[Zrank]);
  aux->batch = batch;
  aux->m = ncx ? out_channel : Z_spatial_total_dim;
  aux->n = ncx ? Z_spatial_total_dim : out_channel;
  aux->k = in_channel * K_spatial_total_dim;
  aux->ncx = ncx;
  aux->X_batch_stride = in_channel * X_spatial_total_dim;
  aux->K_spatial_total_dim = K_spatial_total_dim;
  aux->Z_spatial_total_dim = Z_spatial_total_dim;
  aux->im2col = im2col;
  if (im2col) {
    ai_t _strides, _dilations;
    for (int i = 0; i < conv_rank; ++i) {
      _strides[i] = strides[i];
      _dilations[i] = dilations[i];
    }
    Im2colPrepare(conv_rank, _strides, _dilations, _paddings, Xspatials,
                  Kspatials, Zspatials, in_channel, &aux->im2col_aux);
  }
  return true;
}

bool ConvInferShape(const Shape& X, const Shape& K, int conv_rank,
                    int data_format, const std::vector<int>& strides,
                    const std::vector<int>& dilations, int padding_mode,
                    const std::vector<int>& paddings, Shape* Z) noexcept {
  ConvAux aux;
  if (!ConvPrepare(X, K, conv_rank, data_format, strides, dilations,
                   padding_mode, paddings, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void ConvPrepare(const ConvAux& aux, ConvMutableAux<T>* maux) {
  if (aux.im2col) {
    maux->buf.resize(aux.im2col_aux.in_channel * aux.K_spatial_total_dim *
                     aux.Z_spatial_total_dim);
  }
}

template <typename T>
void Conv(const Tensor<T>& X, const Tensor<T>& K, Tensor<T>* Z,
          const ConvAux& aux, ConvMutableAux<T>* maux) noexcept {
  int batch = aux.batch;
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  const T* _K = K.data();
  T* _Z = Z->data();

  if (aux.im2col) {
    T* _buf = maux->buf.data();
    for (int i = 0; i < batch; ++i) {
      if (aux.ncx) {
        Im2colNCX(_X, _buf, aux.im2col_aux);
        LLMath<T>::gemm(0, 0, m, n, k, 1, _K, _buf, 0, _Z);
      } else {
        Im2colNXC(_X, _buf, aux.im2col_aux);
        LLMath<T>::gemm(0, 0, m, n, k, 1, _buf, _K, 0, _Z);
      }
      _X += aux.X_batch_stride;
      _Z += m * n;
    }
  } else {
    for (int i = 0; i < batch; ++i) {
      if (aux.ncx) {
        LLMath<T>::gemm(0, 0, m, n, k, 1, _K, _X, 0, _Z);
      } else {
        LLMath<T>::gemm(0, 0, m, n, k, 1, _X, _K, 0, _Z);
      }
      _X += aux.X_batch_stride;
      _Z += m * n;
    }
  }
}

template <typename T>
void ConvBackward(const Tensor<T>& X, const Tensor<T>& K,
                  const Tensor<T>& /*Z*/, const Tensor<T>& gZ, Tensor<T>* gX,
                  Tensor<T>* gK, const ConvAux& aux,
                  ConvMutableAux<T>* maux) noexcept {
  int batch = aux.batch;
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  const T* _K = K.data();
  T* _gX = gX ? gX->data() : nullptr;
  T* _gK = gK ? gK->data() : nullptr;

  if (_gX) {
    const T* _gZ = gZ.data();
    if (aux.im2col) {
      T* _buf = maux->buf.data();
      for (int i = 0; i < batch; ++i) {
        if (aux.ncx) {
          LLMath<T>::gemm(1, 0, k, n, m, 1, _K, _gZ, 0, _buf);
          Col2imNCX(_buf, _gX, aux.im2col_aux);
        } else {
          LLMath<T>::gemm(0, 1, m, k, n, 1, _gZ, _K, 0, _buf);
          Col2imNXC(_buf, _gX, aux.im2col_aux);
        }
        _gX += aux.X_batch_stride;
        _gZ += m * n;
      }
    } else {
      for (int i = 0; i < batch; ++i) {
        if (aux.ncx) {
          LLMath<T>::gemm(1, 0, k, n, m, 1, _K, _gZ, 1, _gX);
        } else {
          LLMath<T>::gemm(0, 1, m, k, n, 1, _gZ, _K, 1, _gX);
        }
        _gX += aux.X_batch_stride;
        _gZ += m * n;
      }
    }
  }

  if (_gK) {
    const T* _gZ = gZ.data();
    if (aux.im2col) {
      T* _buf = maux->buf.data();
      for (int i = 0; i < batch; ++i) {
        if (aux.ncx) {
          Im2colNCX(_X, _buf, aux.im2col_aux);
          LLMath<T>::gemm(0, 1, m, k, n, 1, _gZ, _buf, 1, _gK);
        } else {
          Im2colNXC(_X, _buf, aux.im2col_aux);
          LLMath<T>::gemm(1, 0, k, n, m, 1, _buf, _gZ, 1, _gK);
        }
        _X += aux.X_batch_stride;
        _gZ += m * n;
      }
    } else {
      for (int i = 0; i < batch; ++i) {
        if (aux.ncx) {
          LLMath<T>::gemm(0, 1, m, k, n, 1, _gZ, _X, 1, _gK);
        } else {
          LLMath<T>::gemm(1, 0, k, n, m, 1, _X, _gZ, 1, _gK);
        }
        _X += aux.X_batch_stride;
        _gZ += m * n;
      }
    }
  }
}

}  // namespace

/************************************************************************/
/* ConvBase */
/************************************************************************/
GraphNodeConvBase::GraphNodeConvBase(std::string name, GraphNode* X,
                                     GraphNode* K, int conv_rank,
                                     int data_format, int stride, int dilation,
                                     int padding_mode, int padding)
    : GraphNodeConvBase(std::move(name), X, K, conv_rank, data_format,
                        std::vector<int>{stride}, std::vector<int>{dilation},
                        padding_mode, std::vector<int>{padding}) {}

GraphNodeConvBase::GraphNodeConvBase(std::string name, GraphNode* X,
                                     GraphNode* K, int conv_rank,
                                     int data_format, std::vector<int> strides,
                                     std::vector<int> dilations,
                                     int padding_mode,
                                     std::vector<int> paddings)
    : GraphNodeBinaryBase(std::move(name), X, K),
      conv_rank_(conv_rank),
      data_format_(data_format),
      strides_(std::move(strides)),
      dilations_(std::move(dilations)),
      padding_mode_(padding_mode),
      paddings_(std::move(paddings)) {
  DXCHECK_THROW(ConvCheckAttr(conv_rank_, data_format_, strides_, dilations_,
                              padding_mode_, paddings_));
  if (!X->shape().empty() && !K->shape().empty()) {
    (void)ConvInferShape(X->shape(), K->shape(), conv_rank_, data_format_,
                         strides_, dilations_, padding_mode_, paddings_,
                         &shape_);
  }
}

class OpConvBase : public OpBinaryBase {
 protected:
  ConvAux aux_;
  ConvMutableAux<float_t> maux_;

 public:
  const Shape& InferShape() override {
    const GraphNodeConvBase* node = (const GraphNodeConvBase*)node_;  // NOLINT
    DXCHECK_THROW(ConvPrepare(X_->shape(), Y_->shape(), node->conv_rank(),
                              node->data_format(), node->strides(),
                              node->dilations(), node->padding_mode(),
                              node->paddings(), &aux_));
    return aux_.Z;
  }

  void InitForward() override {
    OpBinaryBase::InitForward();
    ConvPrepare(aux_, &maux_);
  }

  void Forward() override { Conv(*X_, *Y_, Z_, aux_, &maux_); }

  void Backward() override {
    if (gZ_) {
      ConvBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, &maux_);
    }
  }
};

/************************************************************************/
/* Conv1d */
/************************************************************************/
Conv1dNode::Conv1dNode(std::string name, GraphNode* X, GraphNode* K,
                       int data_format, int stride, int dilation, int padding)
    : Conv1dNode(std::move(name), X, K, data_format, stride, dilation,
                 PADDING_MODE_USE_PADDINGS, padding) {}

Conv1dNode::Conv1dNode(std::string name, GraphNode* X, GraphNode* K,
                       int data_format, int stride, int dilation,
                       int padding_mode, int padding)
    : GraphNodeConvBase(std::move(name), X, K, 1, data_format,
                        std::vector<int>{stride}, std::vector<int>{dilation},
                        padding_mode, std::vector<int>{padding}) {}

class Conv1dOp : public OpConvBase {
 public:
  DEFINE_OP_LIKE(Conv1dOp);
};

GRAPH_NODE_OP_REGISTER(Conv1d);

/************************************************************************/
/* Conv2d */
/************************************************************************/
Conv2dNode::Conv2dNode(std::string name, GraphNode* X, GraphNode* K,
                       int data_format, std::vector<int> strides,
                       std::vector<int> dilations, std::vector<int> paddings)
    : Conv2dNode(std::move(name), X, K, data_format, std::move(strides),
                 std::move(dilations), PADDING_MODE_USE_PADDINGS,
                 std::move(paddings)) {}

Conv2dNode::Conv2dNode(std::string name, GraphNode* X, GraphNode* K,
                       int data_format, std::vector<int> strides,
                       std::vector<int> dilations, int padding_mode,
                       std::vector<int> paddings)
    : GraphNodeConvBase(std::move(name), X, K, 2, data_format,
                        std::move(strides), std::move(dilations), padding_mode,
                        std::move(paddings)) {}

class Conv2dOp : public OpConvBase {
 public:
  DEFINE_OP_LIKE(Conv2dOp);
};

GRAPH_NODE_OP_REGISTER(Conv2d);

/************************************************************************/
/* Conv3d */
/************************************************************************/
Conv3dNode::Conv3dNode(std::string name, GraphNode* X, GraphNode* K,
                       int data_format, std::vector<int> strides,
                       std::vector<int> dilations, std::vector<int> paddings)
    : Conv3dNode(std::move(name), X, K, data_format, std::move(strides),
                 std::move(dilations), PADDING_MODE_USE_PADDINGS,
                 std::move(paddings)) {}

Conv3dNode::Conv3dNode(std::string name, GraphNode* X, GraphNode* K,
                       int data_format, std::vector<int> strides,
                       std::vector<int> dilations, int padding_mode,
                       std::vector<int> paddings)
    : GraphNodeConvBase(std::move(name), X, K, 3, data_format,
                        std::move(strides), std::move(dilations), padding_mode,
                        std::move(paddings)) {}

class Conv3dOp : public OpConvBase {
 public:
  DEFINE_OP_LIKE(Conv3dOp);
};

GRAPH_NODE_OP_REGISTER(Conv3d);

}  // namespace deepx_core
