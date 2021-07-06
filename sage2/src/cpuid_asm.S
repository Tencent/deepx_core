// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_cpuid)
ASM_FUNC(sage2_cpuid):
/************************************************************************/
movq %rbx, %r8
movl %edi, %eax
movl %esi, %ecx
movq %rdx, %r9
cpuid
movl %eax, (%r9)
movl %ebx, 4(%r9)
movl %ecx, 8(%r9)
movl %edx, 12(%r9)
movq %r8, %rbx
retq
