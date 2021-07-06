// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_LEGACY_H_
#define SAGE2_LEGACY_H_

#include <sage2/half.h>
#include <sage2/vmf.h>
#include <stdint.h>  // NOLINT

#define sage_zero_ps sage2_zero_ps

#define sage_prefetch_range(addr, len)         \
  do {                                         \
    const char* cp = (const char*)addr;        \
    const char* end = (const char*)addr + len; \
    for (; cp < end; cp += 64) {               \
      __builtin_prefetch(cp);                  \
    }                                          \
  } while (0)

inline void sage_axpy_ps(int n, float a, const float* x, float* y) {
  sage2_axpy_ps(n, a, x, y);
}

inline void sage_axpby_ps(int n, float a, const float* x, float b, float* y) {
  sage2_axpby_ps(n, a, x, b, y);
}

inline void sage_scal_ps(int n, float a, const float* x, float* y) {
  sage2_mul_scalar_ps(n, x, a, y);
}

inline void sage_sqrt_ps(int n, const float* x, float* y) {
  sage2_sqrt_ps(n, x, y);
}

inline float sage_dot_ps(int n, const float* x, const float* y) {
  return sage2_dot_ps(n, x, y);
}

inline float sage_nrm2_ps(int n, const float* x) { return sage2_nrm2_ps(n, x); }

inline void sage_relu_ps(int n, const float* x, float* y) {
  sage2_relu_ps(n, x, y);
}

typedef sage2_half_t sage_half_t;

inline sage_half_t sage_d2h(double d) { return sage2_d2h(d); }

inline double sage_h2d(sage_half_t h) { return sage2_h2d(h); }

inline sage_half_t sage_s2h(float s) { return sage2_s2h(s); }

inline float sage_h2s(sage_half_t h) { return sage2_h2s(h); }

inline void sage_pd2ph(int n, const double* pd, sage_half_t* ph) {
  sage2_pd2ph(n, pd, ph);
}

inline void sage_ph2pd(int n, const sage_half_t* ph, double* pd) {
  sage2_ph2pd(n, ph, pd);
}

inline void sage_ps2ph(int n, const float* ps, sage_half_t* ph) {
  sage2_ps2ph(n, ps, ph);
}

inline void sage_ph2ps(int n, const sage_half_t* ph, float* ps) {
  sage2_ph2ps(n, ph, ps);
}

inline void sage_embedding_lookup_ps_Wrowpow2_Wcol1(
    int Xrow, int Wrow_mask, int Wcol, const int* Xrow_offsets,
    const uint64_t* Xcols, const float* Xvalues, const float* W, const float* b,
    float* Z) {
  int i, k_begin, k_end, k_end8, k;
  int j0, j1, j2, j3, j4, j5, j6, j7;
  float Wj0, Wj1, Wj2, Wj3, Wj4, Wj5, Wj6, Wj7;
  float* Zi;
  float dZi;
  (void)Wcol;

  if (b) {
    // Z = b
    Zi = Z;
    for (i = 0; i < (Xrow & -8); i += 8) {
      Zi[0] = b[0];
      Zi[1] = b[0];
      Zi[2] = b[0];
      Zi[3] = b[0];
      Zi[4] = b[0];
      Zi[5] = b[0];
      Zi[6] = b[0];
      Zi[7] = b[0];
      Zi += 8;
    }
    for (; i < Xrow; ++i) {
      *Zi++ = b[0];
    }
  } else {
    sage_zero_ps(Xrow * Wcol, Z);
  }

  // Z = Z + X * W
  Zi = Z;
  for (i = 0; i < Xrow; ++i) {
    k_begin = Xrow_offsets[i];
    k_end = Xrow_offsets[i + 1];
    k_end8 = k_begin + ((k_end - k_begin) & -8);
    k = k_begin;
    dZi = 0;
    for (; k < k_end8; k += 8) {
      j0 = (int)(Xcols[k + 0] & Wrow_mask);
      j1 = (int)(Xcols[k + 1] & Wrow_mask);
      j2 = (int)(Xcols[k + 2] & Wrow_mask);
      j3 = (int)(Xcols[k + 3] & Wrow_mask);
      j4 = (int)(Xcols[k + 4] & Wrow_mask);
      j5 = (int)(Xcols[k + 5] & Wrow_mask);
      j6 = (int)(Xcols[k + 6] & Wrow_mask);
      j7 = (int)(Xcols[k + 7] & Wrow_mask);
      Wj0 = W[j0];
      Wj1 = W[j1];
      Wj2 = W[j2];
      Wj3 = W[j3];
      Wj4 = W[j4];
      Wj5 = W[j5];
      Wj6 = W[j6];
      Wj7 = W[j7];
      dZi += Xvalues[k] * Wj0;
      dZi += Xvalues[k + 1] * Wj1;
      dZi += Xvalues[k + 2] * Wj2;
      dZi += Xvalues[k + 3] * Wj3;
      dZi += Xvalues[k + 4] * Wj4;
      dZi += Xvalues[k + 5] * Wj5;
      dZi += Xvalues[k + 6] * Wj6;
      dZi += Xvalues[k + 7] * Wj7;
    }
    for (; k < k_end; ++k) {
      j0 = (int)(Xcols[k] & Wrow_mask);
      Wj0 = W[j0];
      dZi += Xvalues[k] * Wj0;
    }
    *Zi += dZi;
    ++Zi;
  }
}

inline void sage_embedding_lookup_ps_Wrowpow2_Wcollarge(
    int Xrow, int Wrow_mask, int Wcol, const int* Xrow_offsets,
    const uint64_t* Xcols, const float* Xvalues, const float* W, const float* b,
    float* Z) {
  int i, j, k_begin, k_end, k;
  const float *Wj, *Wjj, *Wjjj;
  float* Zi;

  if (b) {
    // Z = b
    Zi = Z;
    for (i = 0; i < Xrow; ++i) {
      sage_axpby_ps(Wcol, 1, b, 0, Zi);
      Zi += Wcol;
    }
  } else {
    sage_zero_ps(Xrow * Wcol, Z);
  }

  // Z = Z + X * W
  for (i = 0; i < Xrow; ++i) {
    k_begin = Xrow_offsets[i];
    k_end = Xrow_offsets[i + 1];
    Zi = Z + i * Wcol;
    sage_prefetch_range(Zi, Wcol);

    if (k_end - k_begin >= 2) {
      k = k_begin;
      j = (int)(Xcols[k] & Wrow_mask);
      Wj = W + j * Wcol;
      sage_prefetch_range(Wj, Wcol);
      j = (int)(Xcols[k + 1] & Wrow_mask);
      Wjj = W + j * Wcol;
      sage_prefetch_range(Wjj, Wcol);
      for (; k < k_end - 2; ++k) {
        j = (int)(Xcols[k + 2] & Wrow_mask);
        Wjjj = W + j * Wcol;
        sage_prefetch_range(Wjjj, Wcol);
        sage_axpy_ps(Wcol, Xvalues[k], Wj, Zi);
        Wj = Wjj;
        Wjj = Wjjj;
      }
      sage_axpy_ps(Wcol, Xvalues[k], Wj, Zi);
      sage_axpy_ps(Wcol, Xvalues[k + 1], Wjj, Zi);
    } else {
      for (k = k_begin; k < k_end; ++k) {
        j = (int)(Xcols[k] & Wrow_mask);
        Wj = W + j * Wcol;
        sage_axpy_ps(Wcol, Xvalues[k], Wj, Zi);
      }
    }
  }
}

inline void sage_embedding_lookup_ps_Wrowpow2_Wcolsmall(
    int Xrow, int Wrow_mask, int Wcol, const int* Xrow_offsets,
    const uint64_t* Xcols, const float* Xvalues, const float* W, const float* b,
    float* Z) {
  int i, j, k_begin, k_end, k, l;
  float Xij;
  const float* Wj;
  float* Zi;

  if (b) {
    // Z = b
    Zi = Z;
    for (i = 0; i < Xrow; ++i) {
      for (l = 0; l < Wcol; ++l) {
        Zi[l] = b[l];
      }
      Zi += Wcol;
    }
  } else {
    sage_zero_ps(Xrow * Wcol, Z);
  }

  // Z = Z + X * W
  for (i = 0; i < Xrow; ++i) {
    k_begin = Xrow_offsets[i];
    k_end = Xrow_offsets[i + 1];
    Zi = Z + i * Wcol;
    for (k = k_begin; k < k_end; ++k) {
      j = (int)(Xcols[k] & Wrow_mask);
      Xij = Xvalues[k];
      Wj = W + j * Wcol;
      for (l = 0; l < Wcol; ++l) {
        Zi[l] += Xij * Wj[l];
      }
    }
  }
}

inline void sage_embedding_lookup_ps_Wcol1(int Xrow, int Wrow, int Wcol,
                                           const int* Xrow_offsets,
                                           const uint64_t* Xcols,
                                           const float* Xvalues, const float* W,
                                           const float* b, float* Z) {
  int i, k_begin, k_end, k_end8, k;
  int j0, j1, j2, j3, j4, j5, j6, j7;
  float Wj0, Wj1, Wj2, Wj3, Wj4, Wj5, Wj6, Wj7;
  float* Zi;
  float dZi;
  (void)Wcol;

  if (b) {
    // Z = b
    Zi = Z;
    for (i = 0; i < (Xrow & -8); i += 8) {
      Zi[0] = b[0];
      Zi[1] = b[0];
      Zi[2] = b[0];
      Zi[3] = b[0];
      Zi[4] = b[0];
      Zi[5] = b[0];
      Zi[6] = b[0];
      Zi[7] = b[0];
      Zi += 8;
    }
    for (; i < Xrow; ++i) {
      *Zi++ = b[0];
    }
  } else {
    sage_zero_ps(Xrow * Wcol, Z);
  }

  // Z = Z + X * W
  Zi = Z;
  for (i = 0; i < Xrow; ++i) {
    k_begin = Xrow_offsets[i];
    k_end = Xrow_offsets[i + 1];
    k_end8 = k_begin + ((k_end - k_begin) & -8);
    k = k_begin;
    dZi = 0;
    for (; k < k_end8; k += 8) {
      j0 = (int)(Xcols[k] % Wrow);
      j1 = (int)(Xcols[k + 1] % Wrow);
      j2 = (int)(Xcols[k + 2] % Wrow);
      j3 = (int)(Xcols[k + 3] % Wrow);
      j4 = (int)(Xcols[k + 4] % Wrow);
      j5 = (int)(Xcols[k + 5] % Wrow);
      j6 = (int)(Xcols[k + 6] % Wrow);
      j7 = (int)(Xcols[k + 7] % Wrow);
      Wj0 = W[j0];
      Wj1 = W[j1];
      Wj2 = W[j2];
      Wj3 = W[j3];
      Wj4 = W[j4];
      Wj5 = W[j5];
      Wj6 = W[j6];
      Wj7 = W[j7];
      dZi += Xvalues[k] * Wj0;
      dZi += Xvalues[k + 1] * Wj1;
      dZi += Xvalues[k + 2] * Wj2;
      dZi += Xvalues[k + 3] * Wj3;
      dZi += Xvalues[k + 4] * Wj4;
      dZi += Xvalues[k + 5] * Wj5;
      dZi += Xvalues[k + 6] * Wj6;
      dZi += Xvalues[k + 7] * Wj7;
    }
    for (; k < k_end; ++k) {
      j0 = (int)(Xcols[k] % Wrow);
      Wj0 = W[j0];
      dZi += Xvalues[k] * Wj0;
    }
    *Zi += dZi;
    ++Zi;
  }
}

inline void sage_embedding_lookup_ps_Wcollarge(int Xrow, int Wrow, int Wcol,
                                               const int* Xrow_offsets,
                                               const uint64_t* Xcols,
                                               const float* Xvalues,
                                               const float* W, const float* b,
                                               float* Z) {
  int i, j, k_begin, k_end, k;
  const float *Wj, *Wjj, *Wjjj;
  float* Zi;

  if (b) {
    // Z = b
    Zi = Z;
    for (i = 0; i < Xrow; ++i) {
      sage_axpby_ps(Wcol, 1, b, 0, Zi);
      Zi += Wcol;
    }
  } else {
    sage_zero_ps(Xrow * Wcol, Z);
  }

  // Z = Z + X * W
  for (i = 0; i < Xrow; ++i) {
    k_begin = Xrow_offsets[i];
    k_end = Xrow_offsets[i + 1];
    Zi = Z + i * Wcol;
    sage_prefetch_range(Zi, Wcol);

    if (k_end - k_begin >= 2) {
      k = k_begin;
      j = (int)(Xcols[k] % Wrow);
      Wj = W + j * Wcol;
      sage_prefetch_range(Wj, Wcol);
      j = (int)(Xcols[k + 1] % Wrow);
      Wjj = W + j * Wcol;
      sage_prefetch_range(Wjj, Wcol);
      for (; k < k_end - 2; ++k) {
        j = (int)(Xcols[k + 2] % Wrow);
        Wjjj = W + j * Wcol;
        sage_prefetch_range(Wjjj, Wcol);
        sage_axpy_ps(Wcol, Xvalues[k], Wj, Zi);
        Wj = Wjj;
        Wjj = Wjjj;
      }
      sage_axpy_ps(Wcol, Xvalues[k], Wj, Zi);
      sage_axpy_ps(Wcol, Xvalues[k + 1], Wjj, Zi);
    } else {
      for (k = k_begin; k < k_end; ++k) {
        j = (int)(Xcols[k] % Wrow);
        Wj = W + j * Wcol;
        sage_axpy_ps(Wcol, Xvalues[k], Wj, Zi);
      }
    }
  }
}

inline void sage_embedding_lookup_ps_Wcolsmall(int Xrow, int Wrow, int Wcol,
                                               const int* Xrow_offsets,
                                               const uint64_t* Xcols,
                                               const float* Xvalues,
                                               const float* W, const float* b,
                                               float* Z) {
  int i, j, k_begin, k_end, k, l;
  float Xij;
  const float* Wj;
  float* Zi;

  if (b) {
    // Z = b
    Zi = Z;
    for (i = 0; i < Xrow; ++i) {
      for (l = 0; l < Wcol; ++l) {
        Zi[l] = b[l];
      }
      Zi += Wcol;
    }
  } else {
    sage_zero_ps(Xrow * Wcol, Z);
  }

  // Z = Z + X * W
  for (i = 0; i < Xrow; ++i) {
    k_begin = Xrow_offsets[i];
    k_end = Xrow_offsets[i + 1];
    Zi = Z + i * Wcol;
    for (k = k_begin; k < k_end; ++k) {
      j = (int)(Xcols[k] % Wrow);
      Xij = Xvalues[k];
      Wj = W + j * Wcol;
      for (l = 0; l < Wcol; ++l) {
        Zi[l] += Xij * Wj[l];
      }
    }
  }
}

inline void sage_embedding_lookup_ps(int Xrow, int Wrow, int Wcol,
                                     const int* Xrow_offsets,
                                     const uint64_t* Xcols,
                                     const float* Xvalues, const float* W,
                                     const float* b, float* Z) {
  if (__builtin_popcount(Wrow) == 1) {
    if (Wcol >= 16) {
      sage_embedding_lookup_ps_Wrowpow2_Wcollarge(
          Xrow, Wrow - 1, Wcol, Xrow_offsets, Xcols, Xvalues, W, b, Z);
    } else if (Wcol == 1) {
      sage_embedding_lookup_ps_Wrowpow2_Wcol1(
          Xrow, Wrow - 1, Wcol, Xrow_offsets, Xcols, Xvalues, W, b, Z);
    } else {
      sage_embedding_lookup_ps_Wrowpow2_Wcolsmall(
          Xrow, Wrow - 1, Wcol, Xrow_offsets, Xcols, Xvalues, W, b, Z);
    }
  } else {
    if (Wcol >= 16) {
      sage_embedding_lookup_ps_Wcollarge(Xrow, Wrow, Wcol, Xrow_offsets, Xcols,
                                         Xvalues, W, b, Z);
    } else if (Wcol == 1) {
      sage_embedding_lookup_ps_Wcol1(Xrow, Wrow, Wcol, Xrow_offsets, Xcols,
                                     Xvalues, W, b, Z);
    } else {
      sage_embedding_lookup_ps_Wcolsmall(Xrow, Wrow, Wcol, Xrow_offsets, Xcols,
                                         Xvalues, W, b, Z);
    }
  }
}

#endif  // SAGE2_LEGACY_H_
