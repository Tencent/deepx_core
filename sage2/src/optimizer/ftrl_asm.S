// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
FTRL_ONE:
.float 1
FTRL_TWO:
.float 2
FTRL_ABS_MASK:
.int 0x7fffffff
/************************************************************************/


/************************************************************************/
.text
.globl ASM_FUNC(sage2_ftrl_update_ss)
ASM_FUNC(sage2_ftrl_update_ss):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define G_REG           %xmm0
#define W_PTR           %rsi
#define N_PTR           %rdx
#define Z_PTR           %rcx
vmovss FTRL_ABS_MASK(%rip), %xmm9
vmovss FTRL_ONE(%rip), %xmm10
vmovss FTRL_TWO(%rip), %xmm11
vmovss  8(CONFIG_PTR), %xmm13
vmovss 16(CONFIG_PTR), %xmm15

vmovss (N_PTR), %xmm1
vsqrtss %xmm1, %xmm1, %xmm2
vfmadd231ss G_REG, G_REG, %xmm1
vsqrtss %xmm1, %xmm1, %xmm3
vmovss %xmm1, (N_PTR)
vsubss %xmm3, %xmm2, %xmm1
vmulss %xmm15, %xmm1, %xmm1
vaddss (Z_PTR), G_REG, %xmm4
vfmadd231ss (W_PTR), %xmm1, %xmm4
vmovss %xmm4, (Z_PTR)
vxorps %xmm0, %xmm0, %xmm0
vcmpless %xmm4, %xmm0, %xmm0
vandps %xmm0, %xmm11, %xmm0
vsubss %xmm10, %xmm0, %xmm0
vandps %xmm4, %xmm9, %xmm1
vcmpless %xmm1, %xmm13, %xmm1
vandps %xmm1, %xmm10, %xmm1
vfmsub132ss %xmm13, %xmm4, %xmm0
vaddss 4(CONFIG_PTR), %xmm3, %xmm2
vmulss %xmm15, %xmm2, %xmm2
vaddss 8(CONFIG_PTR), %xmm2, %xmm2
vdivss %xmm2, %xmm0, %xmm0
vmulss %xmm1, %xmm0, %xmm0
vmovss %xmm0, (W_PTR)

vzeroupper
retq
#undef CONFIG_PTR
#undef G_REG
#undef W_PTR
#undef N_PTR
#undef Z_PTR


/************************************************************************/
.text
.globl ASM_FUNC(sage2_ftrl_update_ps)
ASM_FUNC(sage2_ftrl_update_ps):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define N_REG           %rsi
#define G_PTR           %rdx
#define W_PTR           %rcx
#define N_PTR           %r8
#define Z_PTR           %r9
#define I_REG           %r10
#define M_REG           %r11
vbroadcastss FTRL_ABS_MASK(%rip), %ymm9
vbroadcastss FTRL_ONE(%rip), %ymm10
vbroadcastss FTRL_TWO(%rip), %ymm11
vbroadcastss  4(CONFIG_PTR), %ymm12   // ymm12: beta
vbroadcastss  8(CONFIG_PTR), %ymm13   // ymm13: l1
vbroadcastss 12(CONFIG_PTR), %ymm14   // ymm14: l2
vbroadcastss 16(CONFIG_PTR), %ymm15   // ymm15: inv_alpha

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 10f

1:
// n'
vmovups (G_PTR,I_REG,4), %ymm0        // ymm0: g
vmovups (N_PTR,I_REG,4), %ymm1        // ymm1: n
vsqrtps %ymm1, %ymm2                  // ymm2: sqrt(n)
vfmadd231ps %ymm0, %ymm0, %ymm1       // ymm1: n' = n + g * g
vsqrtps %ymm1, %ymm3                  // ymm3: sqrt(n')
vmovups %ymm1, (N_PTR,I_REG,4)        // n = n'
// z'
vsubps %ymm3, %ymm2, %ymm1            // ymm1: sqrt(n) - sqrt(n')
vmulps %ymm15, %ymm1, %ymm1           // ymm1: sigma = (sqrt(n) - sqrt(n')) / alpha
vaddps (Z_PTR,I_REG,4), %ymm0, %ymm4  // ymm4: z + g
vfmadd231ps (W_PTR,I_REG,4), %ymm1, %ymm4 // ymm4: z' = z + g + sigma * w
vmovups %ymm4, (Z_PTR,I_REG,4)        // z = z'
// w' & w''
vxorps %ymm0, %ymm0, %ymm0            // ymm0: 0
vcmpleps %ymm4, %ymm0, %ymm0          // ymm0: (0 <= z') ? 0xffffffff : 0
vandps %ymm0, %ymm11, %ymm0           // ymm0: (0 <= z') ? 2 : 0
vsubps %ymm10, %ymm0, %ymm0           // ymm0: sign(z') = (0 <= z') ? 1 : -1
vandps %ymm4, %ymm9, %ymm1            // ymm1: abs(z')
vcmpleps %ymm1, %ymm13, %ymm1         // ymm1: (l1 <= abs(z')) ? 0xffffffff : 0
vandps %ymm1, %ymm10, %ymm1           // ymm1: (l1 <= abs(z')) ? 1 : 0
vfmsub132ps %ymm13, %ymm4, %ymm0      // ymm0: sign(z') * l1 - z'
vaddps %ymm12, %ymm3, %ymm2           // ymm2: sqrt(n') + beta
vmulps %ymm15, %ymm2, %ymm2           // ymm2: (sqrt(n') + beta) / alpha
vaddps %ymm14, %ymm2, %ymm2           // ymm2: (sqrt(n') + beta) / alpha + l2
vdivps %ymm2, %ymm0, %ymm0            // ymm0: w' = (sign(z') * l1 - z') / ((sqrt(n') + beta) / alpha + l2)
vmulps %ymm1, %ymm0, %ymm0            // ymm0: w'' = w' * ((l1 <= abs(z')) ? 1 : 0)
vmovups %ymm0, (W_PTR,I_REG,4)        // w = w''
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
vsqrtss %xmm1, %xmm1, %xmm2
vfmadd231ss %xmm0, %xmm0, %xmm1
vsqrtss %xmm1, %xmm1, %xmm3
vmovss %xmm1, (N_PTR,I_REG,4)
vsubss %xmm3, %xmm2, %xmm1
vmulss %xmm15, %xmm1, %xmm1
vaddss (Z_PTR,I_REG,4), %xmm0, %xmm4
vfmadd231ss (W_PTR,I_REG,4), %xmm1, %xmm4
vmovss %xmm4, (Z_PTR,I_REG,4)
vxorps %xmm0, %xmm0, %xmm0
vcmpless %xmm4, %xmm0, %xmm0
vandps %xmm0, %xmm11, %xmm0
vsubss %xmm10, %xmm0, %xmm0
vandps %xmm4, %xmm9, %xmm1
vcmpless %xmm1, %xmm13, %xmm1
vandps %xmm1, %xmm10, %xmm1
vfmsub132ss %xmm13, %xmm4, %xmm0
vaddss %xmm12, %xmm3, %xmm2
vmulss %xmm15, %xmm2, %xmm2
vaddss %xmm14, %xmm2, %xmm2
vdivss %xmm2, %xmm0, %xmm0
vmulss %xmm1, %xmm0, %xmm0
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
#undef Z_PTR
#undef I_REG
#undef M_REG
