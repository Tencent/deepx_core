// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


#define N_REG           %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define I_REG           %r8
#define M_REG           %r9


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sqrt_ps)
ASM_FUNC(sage2_sqrt_ps):
/************************************************************************/
xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vsqrtps   (X_PTR,I_REG,4), %ymm0
vsqrtps 32(X_PTR,I_REG,4), %ymm1
vmovups %ymm0,   (Y_PTR,I_REG,4)
vmovups %ymm1, 32(Y_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vsqrtps (X_PTR,I_REG,4), %ymm0
vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0
vsqrtss %xmm0, %xmm0, %xmm0
vmovss %xmm0, (Y_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq
