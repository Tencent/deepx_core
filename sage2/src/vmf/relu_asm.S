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
.globl ASM_FUNC(sage2_relu_ps)
ASM_FUNC(sage2_relu_ps):
/************************************************************************/
vxorps %ymm0, %ymm0, %ymm0

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmaxps   (X_PTR,I_REG,4), %ymm0, %ymm1
vmaxps 32(X_PTR,I_REG,4), %ymm0, %ymm2
vmovups %ymm1,   (Y_PTR,I_REG,4)
vmovups %ymm2, 32(Y_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmaxps (X_PTR,I_REG,4), %ymm0, %ymm1
vmovups %ymm1, (Y_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmaxss (X_PTR,I_REG,4), %xmm0, %xmm1
vmovss %xmm1, (Y_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq
