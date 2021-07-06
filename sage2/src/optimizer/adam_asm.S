// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_adam_update_ss)
ASM_FUNC(sage2_adam_update_ss):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define G_REG           %xmm0
#define W_PTR           %rsi
#define M_PTR           %rdx
#define V_PTR           %rcx
vmovss   (CONFIG_PTR), %xmm10
vmovss  4(CONFIG_PTR), %xmm11
vmovss 32(CONFIG_PTR), %xmm15

vmulss G_REG, G_REG, %xmm1
vmulss (M_PTR), %xmm10, %xmm2
vmulss (V_PTR), %xmm11, %xmm3
vfmadd231ss 24(CONFIG_PTR), G_REG, %xmm2
vfmadd231ss 28(CONFIG_PTR), %xmm1, %xmm3
vmovss %xmm2, (M_PTR)
vmovss %xmm3, (V_PTR)
vsqrtss %xmm3, %xmm3, %xmm4
vaddss 12(CONFIG_PTR), %xmm4, %xmm4
vdivss %xmm4, %xmm2, %xmm4
vfnmadd213ss (W_PTR), %xmm15, %xmm4
vmovss %xmm4, (W_PTR)

vzeroupper
retq
#undef CONFIG_PTR
#undef G_REG
#undef W_PTR
#undef M_PTR
#undef V_PTR


/************************************************************************/
.text
.globl ASM_FUNC(sage2_adam_update_ps)
ASM_FUNC(sage2_adam_update_ps):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define N_REG           %rsi
#define G_PTR           %rdx
#define W_PTR           %rcx
#define M_PTR           %r8
#define V_PTR           %r9
#define I_REG           %r10
#define M_REG           %r11
vbroadcastss   (CONFIG_PTR), %ymm10   // ymm10: rho1
vbroadcastss  4(CONFIG_PTR), %ymm11   // ymm11: rho2
vbroadcastss 12(CONFIG_PTR), %ymm12   // ymm12: beta
vbroadcastss 24(CONFIG_PTR), %ymm13   // ymm13: 1 - rho1
vbroadcastss 28(CONFIG_PTR), %ymm14   // ymm14: 1 - rho2
vbroadcastss 32(CONFIG_PTR), %ymm15   // ymm15: a = sqrt(1 - rho2t) / (1 - rho1t) * alpha

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 10f

1:
// m' & v'
vmovups (G_PTR,I_REG,4), %ymm0        // ymm0: g
vmulps %ymm0, %ymm0, %ymm1            // ymm1: g * g
vmulps (M_PTR,I_REG,4), %ymm10, %ymm2 // ymm2: rho1 * m
vmulps (V_PTR,I_REG,4), %ymm11, %ymm3 // ymm3: rho2 * v
vfmadd231ps %ymm13, %ymm0, %ymm2      // ymm2: m' = rho1 * m + (1 - rho1) * g
vfmadd231ps %ymm14, %ymm1, %ymm3      // ymm3: v' = rho2 * v + (1 - rho2) * g * g
vmovups %ymm2, (M_PTR,I_REG,4)        // m = m'
vmovups %ymm3, (V_PTR,I_REG,4)        // v = v'
// w'
vsqrtps %ymm3, %ymm4                  // ymm4: sqrt(v')
vaddps %ymm12, %ymm4, %ymm4           // ymm4: sqrt(v') + beta
vdivps %ymm4, %ymm2, %ymm4            // ymm4: m' / (sqrt(v') + beta)
vfnmadd213ps (W_PTR,I_REG,4), %ymm15, %ymm4 // ymm4: w - a * m' / (sqrt(v') + beta)
vmovups %ymm4, (W_PTR,I_REG,4)        // w = w'
subq $-8, I_REG
subq $8, M_REG
jne 1b

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (G_PTR,I_REG,4), %xmm0
vmulss %xmm0, %xmm0, %xmm1
vmulss (M_PTR,I_REG,4), %xmm10, %xmm2
vmulss (V_PTR,I_REG,4), %xmm11, %xmm3
vfmadd231ss %xmm13, %xmm0, %xmm2
vfmadd231ss %xmm14, %xmm1, %xmm3
vmovss %xmm2, (M_PTR,I_REG,4)
vmovss %xmm3, (V_PTR,I_REG,4)
vsqrtss %xmm3, %xmm3, %xmm4
vaddss %xmm12, %xmm4, %xmm4
vdivss %xmm4, %xmm2, %xmm4
vfnmadd213ss (W_PTR,I_REG,4), %xmm15, %xmm4
vmovss %xmm4, (W_PTR,I_REG,4)
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
#undef M_PTR
#undef V_PTR
#undef I_REG
#undef M_REG
