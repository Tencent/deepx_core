// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_momentum_update_ss)
ASM_FUNC(sage2_momentum_update_ss):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define G_REG           %xmm0
#define W_PTR           %rsi
#define V_PTR           %rdx
vmovss  (CONFIG_PTR), %xmm14
vmovss 4(CONFIG_PTR), %xmm15

vfmadd231ss (V_PTR), %xmm14, %xmm0
vmovss %xmm0, (V_PTR)
vfnmadd213ss (W_PTR), %xmm15, %xmm0
vmovss %xmm0, (W_PTR)

vzeroupper
retq
#undef CONFIG_PTR
#undef G_REG
#undef W_PTR
#undef V_PTR


/************************************************************************/
.text
.globl ASM_FUNC(sage2_momentum_update_ps)
ASM_FUNC(sage2_momentum_update_ps):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define N_REG           %rsi
#define G_PTR           %rdx
#define W_PTR           %rcx
#define V_PTR           %r8
#define I_REG           %r10
#define M_REG           %r11
vbroadcastss  (CONFIG_PTR), %ymm14    // ymm14: rho
vbroadcastss 4(CONFIG_PTR), %ymm15    // ymm15: alpha

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 10f

1:
// v'
vmovups (G_PTR,I_REG,4), %ymm0        // ymm0: g
vfmadd231ps (V_PTR,I_REG,4), %ymm14, %ymm0 // ymm0: v' = rho * v + g
vmovups %ymm0, (V_PTR,I_REG,4)        // v = v'
// w'
vfnmadd213ps (W_PTR,I_REG,4), %ymm15, %ymm0 // ymm0: w - alpha * v'
vmovups %ymm0, (W_PTR,I_REG,4)        // w = w'
subq $-8, I_REG
subq $8, M_REG
jne 1b

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (G_PTR,I_REG,4), %xmm0
vfmadd231ss (V_PTR,I_REG,4), %xmm14, %xmm0
vmovss %xmm0, (V_PTR,I_REG,4)
vfnmadd213ss (W_PTR,I_REG,4), %xmm15, %xmm0
vmovss %xmm0, (W_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq
#undef CONFIG_PTR
#undef N_REG
#undef G_PTR
#undef W_PTR
#undef V_PTR
#undef I_REG
#undef M_REG
