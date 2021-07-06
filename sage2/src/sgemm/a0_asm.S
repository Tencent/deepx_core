// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"
#include "sgemm/offset.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_a0_b0_Zc)
ASM_FUNC(sage2_sgemm_a0_b0_Zc):
/************************************************************************/
#define CTX_PTR         %rdi
#define Z_PTR           %rcx
#define N_REG_INT       %eax
#define N_REG           %rax            // m * n
#define I_REG           %r8
#define NN_REG          %r9

vxorps %ymm0, %ymm0, %ymm0
movl M_OFFSET(CTX_PTR), N_REG_INT
mull N_OFFSET(CTX_PTR)

xorq I_REG, I_REG
movq N_REG, NN_REG
andq $-16, NN_REG
je 2f

1:
vmovups %ymm0,   (Z_PTR,I_REG,4)
vmovups %ymm0, 32(Z_PTR,I_REG,4)
addq $16, I_REG
subq $16, NN_REG
jne 1b

2:
movq N_REG, NN_REG
andq $8, NN_REG
je 10f
vmovups %ymm0, (Z_PTR,I_REG,4)
addq $8, I_REG

10:
movq N_REG, NN_REG
andq $7, NN_REG
je 12f

11:
vmovss %xmm0, (Z_PTR,I_REG,4)
addq $1, I_REG
subq $1, NN_REG
jne 11b

12:
vzeroupper
retq

#undef CTX_PTR
#undef Z_PTR
#undef N_REG_INT
#undef N_REG
#undef I_REG
#undef NN_REG


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_a0_Zc)
ASM_FUNC(sage2_sgemm_a0_Zc):
/************************************************************************/
#define CTX_PTR         %rdi
#define Z_PTR           %rcx
#define N_REG_INT       %eax
#define N_REG           %rax            // m * n
#define I_REG           %r8
#define NN_REG          %r9

vbroadcastss BETA_OFFSET(CTX_PTR), %ymm0
movl M_OFFSET(CTX_PTR), N_REG_INT
mull N_OFFSET(CTX_PTR)

xorq I_REG, I_REG
movq N_REG, NN_REG
andq $-16, NN_REG
je 2f

1:
vmulps   (Z_PTR,I_REG,4), %ymm0, %ymm1
vmulps 32(Z_PTR,I_REG,4), %ymm0, %ymm2
vmovups %ymm1,   (Z_PTR,I_REG,4)
vmovups %ymm2, 32(Z_PTR,I_REG,4)
addq $16, I_REG
subq $16, NN_REG
jne 1b

2:
movq N_REG, NN_REG
andq $8, NN_REG
je 10f
vmulps (Z_PTR,I_REG,4), %ymm0, %ymm1
vmovups %ymm1, (Z_PTR,I_REG,4)
addq $8, I_REG

10:
movq N_REG, NN_REG
andq $7, NN_REG
je 12f

11:
vmulss (Z_PTR,I_REG,4), %xmm0, %xmm1
vmovss %xmm1, (Z_PTR,I_REG,4)
addq $1, I_REG
subq $1, NN_REG
jne 11b

12:
vzeroupper
retq

#undef CTX_PTR
#undef Z_PTR
#undef N_REG_INT
#undef N_REG
#undef I_REG
#undef NN_REG
