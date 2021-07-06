// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
GFTRL_ONE:
.float 1
GFTRL_ABS_MASK:
.int 0x7fffffff
/************************************************************************/


/************************************************************************/
.text
.globl ASM_FUNC(sage2_gftrl_update_ss)
ASM_FUNC(sage2_gftrl_update_ss):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define G_REG           %xmm0
#define W_PTR           %rsi
#define N_PTR           %rdx
#define Z_PTR           %rcx
vmovss GFTRL_ABS_MASK(%rip), %xmm15

vmovss (N_PTR), %xmm13
vsqrtss %xmm13, %xmm13, %xmm1
vfmadd231ss %xmm0, %xmm0, %xmm13
vmovss %xmm13, (N_PTR)

vsqrtss %xmm13, %xmm13, %xmm2
vsubss %xmm2, %xmm1, %xmm1
vmulss 12(CONFIG_PTR), %xmm1, %xmm1
vaddss (Z_PTR), %xmm0, %xmm14
vfmadd231ss (W_PTR), %xmm1, %xmm14
vmovss %xmm14, (Z_PTR)

vandps %xmm14, %xmm15, %xmm15
vmovss 8(CONFIG_PTR), %xmm0
vcomiss %xmm15, %xmm0
jbe 30f

20:
vxorps %xmm0, %xmm0, %xmm0
vmovss %xmm0, (W_PTR)
vzeroupper
retq

30:
vdivss %xmm15, %xmm0, %xmm0
vsubss GFTRL_ONE(%rip), %xmm0, %xmm0
vmulss (CONFIG_PTR), %xmm0, %xmm0
vsqrtss %xmm13, %xmm13, %xmm1
vaddss 4(CONFIG_PTR), %xmm1, %xmm1
vmulss %xmm14, %xmm0, %xmm2
vdivss %xmm1, %xmm2, %xmm2
vmovss %xmm2, (W_PTR)
vzeroupper
retq
#undef CONFIG_PTR
#undef G_REG
#undef W_PTR
#undef N_PTR
#undef Z_PTR


/************************************************************************/
.text
.globl ASM_FUNC(sage2_gftrl_update_ps)
ASM_FUNC(sage2_gftrl_update_ps):
/************************************************************************/
#define CONFIG_PTR      %rdi
#define N_REG           %rsi
#define G_PTR           %rdx
#define W_PTR           %rcx
#define N_PTR           %r8
#define Z_PTR           %r9
#define I_REG           %r10
#define M_REG           %r11
vxorps %ymm13, %ymm13, %ymm13         // ymm13: norm2_z = 0
vbroadcastss  4(CONFIG_PTR), %ymm14   // ymm14: beta
vbroadcastss 12(CONFIG_PTR), %ymm15   // ymm15: inv_alpha

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 2f

1:
// n'
vmovups (G_PTR,I_REG,4), %ymm0        // ymm0: g
vmovups (N_PTR,I_REG,4), %ymm1        // ymm1: n
vsqrtps %ymm1, %ymm2                  // ymm2: sqrt_n = sqrt(n)
vfmadd231ps %ymm0, %ymm0, %ymm1       // ymm1: new_n = n + g * g
vsqrtps %ymm1, %ymm3                  // ymm3: sqrt_new_n = sqrt(new_n)
vmovups %ymm1, (N_PTR,I_REG,4)        // n = new_n
// z'
vsubps %ymm3, %ymm2, %ymm1            // ymm1: sqrt_n - sqrt_new_n
vmulps %ymm15, %ymm1, %ymm1           // ymm1: sigma = (sqrt_n - sqrt_new_n) * inv_alpha
vaddps (Z_PTR,I_REG,4), %ymm0, %ymm2  // ymm2: z + g
vfmadd231ps (W_PTR,I_REG,4), %ymm1, %ymm2 // ymm2: new_z = z + g + sigma * w
vmovups %ymm2, (Z_PTR,I_REG,4)        // z = new_z
vfmadd231ps %ymm2, %ymm2, %ymm13      // ymm13: norm2_z += new_z * new_z
subq $-8, I_REG
subq $8, M_REG
jne 1b

vextractf128 $1, %ymm13, %xmm0
vaddps %xmm13, %xmm0, %xmm13
vhaddps %xmm0, %xmm13, %xmm13
vhaddps %xmm0, %xmm13, %xmm13

2:
movq N_REG, M_REG
andq $7, M_REG
je 10f

3:
vmovss (G_PTR,I_REG,4), %xmm0
vmovss (N_PTR,I_REG,4), %xmm1
vsqrtss %xmm1, %xmm1, %xmm2
vfmadd231ss %xmm0, %xmm0, %xmm1
vsqrtss %xmm1, %xmm1, %xmm3
vmovss %xmm1, (N_PTR,I_REG,4)
vsubss %xmm3, %xmm2, %xmm1
vmulss %xmm15, %xmm1, %xmm1
vaddss (Z_PTR,I_REG,4), %xmm0, %xmm2
vfmadd231ss (W_PTR,I_REG,4), %xmm1, %xmm2
vmovss %xmm2, (Z_PTR,I_REG,4)
vfmadd231ss %xmm2, %xmm2, %xmm13
subq $-1, I_REG
subq $1, M_REG
jne 3b

/************************************************************************/

10:
vsqrtss %xmm13, %xmm13, %xmm13        // xmm13: norm2_z = sqrt(norm2_z)
vcvtsi2ss N_REG, %xmm0, %xmm0         // xmm0: _n
vsqrtss %xmm0, %xmm0, %xmm0           // xmm0: sqrt(_n)
vmulss 8(CONFIG_PTR), %xmm0, %xmm0    // xmm0: threshold = lambda * sqrt(_n)
vcomiss %xmm13, %xmm0                 // threshold cmp norm2_z
jbe 30f

/************************************************************************/

20:
// threshold > norm2_z
vxorps %ymm0, %ymm0, %ymm0

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 22f

21:
vmovups %ymm0, (W_PTR,I_REG,4)        // w = 0
subq $-8, I_REG
subq $8, M_REG
jne 21b

22:
movq N_REG, M_REG
andq $7, M_REG
je 24f

23:
vmovss %xmm0, (W_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 23b

24:
vzeroupper
retq

/************************************************************************/

30:
// threshold <= norm2_z
vdivss %xmm13, %xmm0, %xmm0           // xmm0: threshold / norm2_z
vsubss GFTRL_ONE(%rip), %xmm0, %xmm0  // xmm0: threshold / norm2_z - 1
vmulss (CONFIG_PTR), %xmm0, %xmm0     // xmm0: tmp = (threshold / norm2_z - 1) * alpha
vmovss %xmm0, -4(%rsp)
vbroadcastss -4(%rsp), %ymm0          // ymm0: tmp = (threshold / norm2_z - 1) * alpha

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 32f

31:
vsqrtps (N_PTR,I_REG,4), %ymm1        // ymm1: sqrt(new_n)
vaddps %ymm14, %ymm1, %ymm1           // ymm1: sqrt(new_n) + beta
vmulps (Z_PTR,I_REG,4), %ymm0, %ymm2  // ymm2: tmp * new_z
vdivps %ymm1, %ymm2, %ymm2            // ymm2: new_w = tmp * new_z / (sqrt(new_n) + beta)
vmovups %ymm2, (W_PTR,I_REG,4)        // w = new_w
subq $-8, I_REG
subq $8, M_REG
jne 31b

32:
movq N_REG, M_REG
andq $7, M_REG
je 34f

33:
vsqrtss (N_PTR,I_REG,4), %xmm1, %xmm1
vaddss %xmm14, %xmm1, %xmm1
vmulss (Z_PTR,I_REG,4), %xmm0, %xmm2
vdivss %xmm1, %xmm2, %xmm2
vmovss %xmm2, (W_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 33b

34:
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
