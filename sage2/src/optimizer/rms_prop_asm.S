// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_rms_prop_update_ss)
ASM_FUNC(sage2_rms_prop_update_ss):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define G_REG           %xmm0
#define W_PTR           %rsi
#define V_PTR           %rdx
vmovss   (CONFIG_PTR), %xmm12
vmovss  4(CONFIG_PTR), %xmm13

vmulss G_REG, G_REG, %xmm1
vmulss (V_PTR), %xmm12, %xmm2
vfmadd132ss 12(CONFIG_PTR), %xmm2, %xmm1
vmovss %xmm1, (V_PTR)
vaddss 8(CONFIG_PTR), %xmm1, %xmm1
vsqrtss %xmm1, %xmm1, %xmm1
vdivss %xmm1, G_REG, %xmm0
vfnmadd213ss (W_PTR), %xmm13, %xmm0
vmovss %xmm0, (W_PTR)

vzeroupper
retq
#undef CONFIG_PTR
#undef G_REG
#undef W_PTR
#undef V_PTR


/************************************************************************/
.text
.globl ASM_FUNC(sage2_rms_prop_update_ps)
ASM_FUNC(sage2_rms_prop_update_ps):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define N_REG           %rsi
#define G_PTR           %rdx
#define W_PTR           %rcx
#define V_PTR           %r8
#define I_REG           %r10
#define M_REG           %r11
vbroadcastss   (CONFIG_PTR), %ymm12    // ymm12: rho
vbroadcastss  4(CONFIG_PTR), %ymm13    // ymm13: alpha
vbroadcastss  8(CONFIG_PTR), %ymm14    // ymm14: beta
vbroadcastss 12(CONFIG_PTR), %ymm15    // ymm15: 1 - rho

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 10f

1:
// v'
vmovups (G_PTR,I_REG,4), %ymm0        // ymm0: g
vmulps %ymm0, %ymm0, %ymm1            // ymm1: g * g
vmulps (V_PTR,I_REG,4), %ymm12, %ymm2 // ymm2: rho * v
vfmadd132ps %ymm15, %ymm2, %ymm1      // ymm1: v' = rho * v + (1 - rho) * g * g
vmovups %ymm1, (V_PTR,I_REG,4)        // v = v'
// w'
vaddps %ymm14, %ymm1, %ymm1           // ymm1: v' + beta
vsqrtps %ymm1, %ymm1                  // ymm1: sqrt(v' + beta)
vdivps %ymm1, %ymm0, %ymm0            // ymm0: g / sqrt(v' + beta)
vfnmadd213ps (W_PTR,I_REG,4), %ymm13, %ymm0 // ymm0: w' = w - g / sqrt(v' + beta) * alpha
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
vmulss %xmm0, %xmm0, %xmm1
vmulss (V_PTR,I_REG,4), %xmm12, %xmm2
vfmadd132ss %xmm15, %xmm2, %xmm1
vmovss %xmm1, (V_PTR,I_REG,4)
vaddss %xmm14, %xmm1, %xmm1
vsqrtss %xmm1, %xmm1, %xmm1
vdivss %xmm1, %xmm0, %xmm0
vfnmadd213ss (W_PTR,I_REG,4), %xmm13, %xmm0
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
