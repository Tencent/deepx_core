// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
GRAD_CLIP_20:
.float 20
GRAD_CLIP_NEG_20:
.float -20
/************************************************************************/


/************************************************************************/
.text
.globl ASM_FUNC(sage2_grad_clip_20_ss)
ASM_FUNC(sage2_grad_clip_20_ss):
/************************************************************************/
#define G_PTR           %rdi
vmovss (G_PTR), %xmm0
vminss GRAD_CLIP_20(%rip), %xmm0, %xmm0
vmaxss GRAD_CLIP_NEG_20(%rip), %xmm0, %xmm0
vmovss %xmm0, (G_PTR)

vzeroupper
retq
#undef G_PTR


/************************************************************************/
.text
.globl ASM_FUNC(sage2_grad_clip_20_ps)
ASM_FUNC(sage2_grad_clip_20_ps):
/************************************************************************/
#define N_REG           %rdi
#define G_PTR           %rsi
#define I_REG           %r8
#define M_REG           %r9
vbroadcastss GRAD_CLIP_20(%rip), %ymm14
vbroadcastss GRAD_CLIP_NEG_20(%rip), %ymm15

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vminps   (G_PTR,I_REG,4), %ymm14, %ymm0
vminps 32(G_PTR,I_REG,4), %ymm14, %ymm1
vmaxps %ymm15, %ymm0, %ymm0
vmaxps %ymm15, %ymm1, %ymm1
vmovups %ymm0,   (G_PTR,I_REG,4)
vmovups %ymm1, 32(G_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vminps (G_PTR,I_REG,4), %ymm14, %ymm0
vmaxps %ymm15, %ymm0, %ymm0
vmovups %ymm0, (G_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vminss (G_PTR,I_REG,4), %xmm14, %xmm0
vmaxss %xmm15, %xmm0, %xmm0
vmovss %xmm0, (G_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq
#undef N_REG
#undef G_PTR
#undef I_REG
#undef M_REG
