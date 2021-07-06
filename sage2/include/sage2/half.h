// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_HALF_H_
#define SAGE2_HALF_H_

#include <sage2/macro.h>
#include <stdint.h>  // NOLINT

/************************************************************************/
/* sage2_half_t: IEEE 754 half precision floating-point number */
/************************************************************************/
// NOLINTNEXTLINE
typedef int16_t sage2_half_t;

/************************************************************************/
/* double scalar <-> half scalar conversion */
/* single scalar <-> half scalar conversion */
/************************************************************************/
SAGE2_C_API sage2_half_t sage2_d2h(double d);
SAGE2_C_API double sage2_h2d(sage2_half_t h);
SAGE2_C_API sage2_half_t sage2_s2h(float s);
SAGE2_C_API float sage2_h2s(sage2_half_t h);

/************************************************************************/
/* double vector <-> half vector conversion */
/* single vector <-> half vector conversion */
/************************************************************************/
SAGE2_C_API void sage2_pd2ph(uint64_t n, const double* pd, sage2_half_t* ph);
SAGE2_C_API void sage2_ph2pd(uint64_t n, const sage2_half_t* ph, double* pd);
SAGE2_C_API void sage2_ps2ph(uint64_t n, const float* ps, sage2_half_t* ph);
SAGE2_C_API void sage2_ph2ps(uint64_t n, const sage2_half_t* ph, float* ps);

#if defined __cplusplus
/************************************************************************/
/* double vector <-> half vector conversion for c++ */
/* single vector <-> half vector conversion for c++ */
/************************************************************************/
inline void sage2_half_convert(uint64_t n, const float* ps,
                               sage2_half_t* ph) SAGE2_NOEXCEPT {
  sage2_ps2ph(n, ps, ph);
}
inline void sage2_half_convert(uint64_t n, const sage2_half_t* ph,
                               float* ps) SAGE2_NOEXCEPT {
  sage2_ph2ps(n, ph, ps);
}
inline void sage2_half_convert(uint64_t n, const double* pd,
                               sage2_half_t* ph) SAGE2_NOEXCEPT {
  sage2_pd2ph(n, pd, ph);
}
inline void sage2_half_convert(uint64_t n, const sage2_half_t* ph,
                               double* pd) SAGE2_NOEXCEPT {
  sage2_ph2pd(n, ph, pd);
}
#endif

#endif  // SAGE2_HALF_H_
