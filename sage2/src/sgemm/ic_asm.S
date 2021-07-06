// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"
#include "sgemm/offset.h"


#define CTX_PTR         %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define Z_PTR           %rcx
#define I_REG           %r8
#define K_REG_INT       %r9d
#define K_REG           %r9
#define KK_REG          %r10


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_ic)
ASM_FUNC(sage2_sgemm_ic):
/************************************************************************/
movl K_OFFSET(CTX_PTR), K_REG_INT
vxorps %ymm0, %ymm0, %ymm0
xorq I_REG, I_REG
movq K_REG, KK_REG
andq $-16, KK_REG
je 2f

vxorps %ymm1, %ymm1, %ymm1
1:
vmovups   (X_PTR,I_REG,4), %ymm2
vmovups 32(X_PTR,I_REG,4), %ymm3
vfmadd231ps   (Y_PTR,I_REG,4), %ymm2, %ymm0
vfmadd231ps 32(Y_PTR,I_REG,4), %ymm3, %ymm1
addq $16, I_REG
subq $16, KK_REG
jne 1b
vaddps %ymm0, %ymm1, %ymm0

2:
movq K_REG, KK_REG
andq $8, KK_REG
je 10f
vmovups (X_PTR,I_REG,4), %ymm1
vfmadd231ps (Y_PTR,I_REG,4), %ymm1, %ymm0
addq $8, I_REG

10:
vextractf128 $1, %ymm0, %xmm1
vaddps %xmm0, %xmm1, %xmm0
vhaddps %xmm1, %xmm0, %xmm0
vhaddps %xmm1, %xmm0, %xmm0
movq K_REG, KK_REG
andq $7, KK_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm2
vfmadd231ss (Y_PTR,I_REG,4), %xmm2, %xmm0
addq $1, I_REG
subq $1, KK_REG
jne 11b

12:
vmulss ALPHA_OFFSET(CTX_PTR), %xmm0, %xmm0
vmovss BETA_OFFSET(CTX_PTR), %xmm1
vfmadd231ss (Z_PTR), %xmm1, %xmm0
vmovss %xmm0, (Z_PTR)
vzeroupper
retq
