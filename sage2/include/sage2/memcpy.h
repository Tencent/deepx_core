// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_MEMCPY_H_
#define SAGE2_MEMCPY_H_

#include <sage2/macro.h>
#include <stddef.h>  // NOLINT

/************************************************************************/
/* sage2_memcpy */
/* They are the same as memcpy. */
/************************************************************************/
SAGE2_C_API void* sage2_memcpy(void* dst, const void* src, size_t n);

#endif  // SAGE2_MEMCPY_H_
