// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_ada_delta_update_ss)
ASM_FUNC(sage2_ada_delta_update_ss):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define G_REG           %xmm0
#define W_PTR           %rsi
#define N_PTR           %rdx
#define DELTAW_PTR      %rcx
vmovss   (CONFIG_PTR), %xmm12
vmovss  4(CONFIG_PTR), %xmm13
vmovss  8(CONFIG_PTR), %xmm14
vmovss 12(CONFIG_PTR), %xmm15

vmulss G_REG, G_REG, %xmm1
vmulss (N_PTR), %xmm12, %xmm2
vfmadd132ss %xmm15, %xmm2, %xmm1
vmovss %xmm1, (N_PTR)
vmovss (DELTAW_PTR), %xmm2
vaddss %xmm14, %xmm2, %xmm3
vsqrtss %xmm3, %xmm3, %xmm3
vaddss %xmm14, %xmm1, %xmm1
vsqrtss %xmm1, %xmm1, %xmm1
vdivss %xmm1, %xmm3, %xmm1
vmulss %xmm1, G_REG, %xmm0
vmulss %xmm0, %xmm0, %xmm1
vmulss %xmm12, %xmm2, %xmm2
vfmadd231ss %xmm15, %xmm1, %xmm2
vmovss %xmm2, (DELTAW_PTR)
vfnmadd213ss (W_PTR), %xmm13, %xmm0
vmovss %xmm0, (W_PTR)

vzeroupper
retq
#undef CONFIG_PTR
#undef G_REG
#undef W_PTR
#undef N_PTR
#undef DELTAW_PTR


/************************************************************************/
.text
.globl ASM_FUNC(sage2_ada_delta_update_ps)
ASM_FUNC(sage2_ada_delta_update_ps):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define N_REG           %rsi
#define G_PTR           %rdx
#define W_PTR           %rcx
#define N_PTR           %r8
#define DELTAW_PTR      %r9
#define I_REG           %r10
#define M_REG           %r11
vbroadcastss   (CONFIG_PTR), %ymm12   // ymm12: rho
vbroadcastss  4(CONFIG_PTR), %ymm13   // ymm13: alpha
vbroadcastss  8(CONFIG_PTR), %ymm14   // ymm14: beta
vbroadcastss 12(CONFIG_PTR), %ymm15   // ymm15: 1 - rho

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 10f

1:
// n'
vmovups (G_PTR,I_REG,4), %ymm0        // ymm0: g
vmulps %ymm0, %ymm0, %ymm1            // ymm1: g * g
vmulps (N_PTR,I_REG,4), %ymm12, %ymm2 // ymm2: rho * n
vfmadd132ps %ymm15, %ymm2, %ymm1      // ymm1: n' = rho * n + (1 - rho) * g * g
vmovups %ymm1, (N_PTR,I_REG,4)        // n = n'
// a & a * a
vmovups (DELTAW_PTR,I_REG,4), %ymm2   // ymm2: deltaw
vaddps %ymm14, %ymm2, %ymm3           // ymm3: deltaw + beta
vsqrtps %ymm3, %ymm3                  // ymm3: sqrt(deltaw + beta)
vaddps %ymm14, %ymm1, %ymm1           // ymm1: n' + beta
vsqrtps %ymm1, %ymm1                  // ymm1: sqrt(n' + beta)
vdivps %ymm1, %ymm3, %ymm1            // ymm1: sqrt(deltaw + beta) / sqrt(n' + beta)
vmulps %ymm1, %ymm0, %ymm0            // ymm0: a = sqrt(deltaw + beta) / sqrt(n' + beta) * g
vmulps %ymm0, %ymm0, %ymm1            // ymm1: a * a
// deltaw'
vmulps %ymm12, %ymm2, %ymm2           // ymm2: rho * deltaw
vfmadd231ps %ymm15, %ymm1, %ymm2      // ymm2: deltaw' = rho * deltaw + (1 - rho) * a * a
vmovups %ymm2, (DELTAW_PTR,I_REG,4)   // deltaw = deltaw'
// w'
vfnmadd213ps (W_PTR,I_REG,4), %ymm13, %ymm0 // ymm0: w' = w - alpha * a
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
vmulss (N_PTR,I_REG,4), %xmm12, %xmm2
vfmadd132ss %xmm15, %xmm2, %xmm1
vmovss %xmm1, (N_PTR,I_REG,4)
vmovss (DELTAW_PTR,I_REG,4), %xmm2
vaddss %xmm14, %xmm2, %xmm3
vsqrtss %xmm3, %xmm3, %xmm3
vaddss %xmm14, %xmm1, %xmm1
vsqrtss %xmm1, %xmm1, %xmm1
vdivss %xmm1, %xmm3, %xmm1
vmulss %xmm1, %xmm0, %xmm0
vmulss %xmm0, %xmm0, %xmm1
vmulss %xmm12, %xmm2, %xmm2
vfmadd231ss %xmm15, %xmm1, %xmm2
vmovss %xmm2, (DELTAW_PTR,I_REG,4)
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
#undef N_PTR
#undef DELTAW_PTR
#undef I_REG
#undef M_REG
