// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
ADA_GRAD_ONE:
.float 1
/************************************************************************/


#define CONFIG_PTR      %rdi
#define N_REG           %rsi
#define G_PTR           %rdx
#define W_PTR           %rcx
#define N_PTR           %r8
#define I_REG           %r10
#define M_REG           %r11


/************************************************************************/
.text
.globl ASM_FUNC(sage2_tf_ada_grad_v1_update_ps)
ASM_FUNC(sage2_tf_ada_grad_v1_update_ps):
/************************************************************************/
vbroadcastss ADA_GRAD_ONE(%rip), %ymm13
vbroadcastss (CONFIG_PTR), %ymm14     // ymm14: alpha

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 10f

1:
// n'
vmovups (G_PTR,I_REG,4), %ymm0        // ymm0: g
vmovups (N_PTR,I_REG,4), %ymm1        // ymm1: n
vfmadd231ps %ymm0, %ymm0, %ymm1       // ymm1: n' = n + g * g
vmovups %ymm1, (N_PTR,I_REG,4)        // n = n'
// w'
vsqrtps %ymm1, %ymm2                  // ymm2: sqrt(n')
vdivps %ymm2, %ymm13, %ymm2           // ymm2: 1 / sqrt(n')
vmulps %ymm14, %ymm2, %ymm2           // ymm2: 1 / sqrt(n') * alpha
vfnmadd213ps (W_PTR,I_REG,4), %ymm0, %ymm2 // ymm2: w' = w - g / sqrt(n') * alpha
vmovups %ymm2, (W_PTR,I_REG,4)        // w = w'
subq $-8, I_REG
subq $8, M_REG
jne 1b

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (G_PTR,I_REG,4), %xmm0
vmovss (N_PTR,I_REG,4), %xmm1
vfmadd231ss %xmm0, %xmm0, %xmm1
vmovss %xmm1, (N_PTR,I_REG,4)
vsqrtss %xmm1, %xmm1, %xmm2
vdivss %xmm2, %xmm13, %xmm2
vmulss %xmm14, %xmm2, %xmm2
vfnmadd213ss (W_PTR,I_REG,4), %xmm0, %xmm2
vmovss %xmm2, (W_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq
