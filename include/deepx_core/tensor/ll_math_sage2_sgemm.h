// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/sgemm.h>

namespace deepx_core {

/************************************************************************/
/* sage2 implementations */
/************************************************************************/
template <>
inline void LLMath<float>::gemm(int transX, int transY, int m, int n, int k,
                                float_t alpha, cptr_t X, int ldX, cptr_t Y,
                                int ldY, float_t beta, ptr_t Z,
                                int ldZ) noexcept {
  sage2_sgemm(101, transX ? 112 : 111, transY ? 112 : 111, m, n, k, alpha, X,
              ldX, Y, ldY, beta, Z, ldZ);
}

}  // namespace deepx_core
