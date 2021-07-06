// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_INTERNAL_MACRO_H_
#define SAGE2_INTERNAL_MACRO_H_

#include <sage2/macro.h>

// PRIVATE_C_FUNC is only a mark.
#define PRIVATE_C_FUNC SAGE2_EXTERN_C

#define ATTR_CTOR(n) __attribute__((constructor(n)))
#define ATTR_DTOR(n) __attribute__((destructor(n)))
#define ATTR_NOINLINE __attribute__((noinline))

#if defined __APPLE__ && defined __MACH__
#define ASM_FUNC(name) _##name
#else
#define ASM_FUNC(name) name
#endif

#endif  // SAGE2_INTERNAL_MACRO_H_
