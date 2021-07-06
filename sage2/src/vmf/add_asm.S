// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


#define N_REG           %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define Z_PTR           %rcx
#define I_REG           %r8
#define M_REG           %r9


/************************************************************************/
.text
.globl ASM_FUNC(sage2_add_ps)
ASM_FUNC(sage2_add_ps):
/************************************************************************/
xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmovups   (X_PTR,I_REG,4), %ymm0
vmovups 32(X_PTR,I_REG,4), %ymm1
vaddps   (Y_PTR,I_REG,4), %ymm0, %ymm0
vaddps 32(Y_PTR,I_REG,4), %ymm1, %ymm1
vmovups %ymm0,   (Z_PTR,I_REG,4)
vmovups %ymm1, 32(Z_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %ymm0
vaddps (Y_PTR,I_REG,4), %ymm0, %ymm0
vmovups %ymm0, (Z_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0
vaddss (Y_PTR,I_REG,4), %xmm0, %xmm0
vmovss %xmm0, (Z_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sub_ps)
ASM_FUNC(sage2_sub_ps):
/************************************************************************/
xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmovups   (X_PTR,I_REG,4), %ymm0
vmovups 32(X_PTR,I_REG,4), %ymm1
vsubps   (Y_PTR,I_REG,4), %ymm0, %ymm0
vsubps 32(Y_PTR,I_REG,4), %ymm1, %ymm1
vmovups %ymm0,   (Z_PTR,I_REG,4)
vmovups %ymm1, 32(Z_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %ymm0
vsubps (Y_PTR,I_REG,4), %ymm0, %ymm0
vmovups %ymm0, (Z_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0
vsubss (Y_PTR,I_REG,4), %xmm0, %xmm0
vmovss %xmm0, (Z_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_mul_ps)
ASM_FUNC(sage2_mul_ps):
/************************************************************************/
xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmovups   (X_PTR,I_REG,4), %ymm0
vmovups 32(X_PTR,I_REG,4), %ymm1
vmulps   (Y_PTR,I_REG,4), %ymm0, %ymm0
vmulps 32(Y_PTR,I_REG,4), %ymm1, %ymm1
vmovups %ymm0,   (Z_PTR,I_REG,4)
vmovups %ymm1, 32(Z_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %ymm0
vmulps (Y_PTR,I_REG,4), %ymm0, %ymm0
vmovups %ymm0, (Z_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0
vmulss (Y_PTR,I_REG,4), %xmm0, %xmm0
vmovss %xmm0, (Z_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_div_ps)
ASM_FUNC(sage2_div_ps):
/************************************************************************/
xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmovups   (X_PTR,I_REG,4), %ymm0
vmovups 32(X_PTR,I_REG,4), %ymm1
vdivps   (Y_PTR,I_REG,4), %ymm0, %ymm0
vdivps 32(Y_PTR,I_REG,4), %ymm1, %ymm1
vmovups %ymm0,   (Z_PTR,I_REG,4)
vmovups %ymm1, 32(Z_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %ymm0
vdivps (Y_PTR,I_REG,4), %ymm0, %ymm0
vmovups %ymm0, (Z_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0
vdivss (Y_PTR,I_REG,4), %xmm0, %xmm0
vmovss %xmm0, (Z_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq
