// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
EXP_HI:
.float  88.02969193111305
EXP_LO:
.float -88.02969193111305
EXP_INV_LOG2E:
.float 1.44269504088896341
EXP_P0_V2:
.float 1.3537703155e-02
EXP_P1_V2:
.float 5.2170695889e-02
EXP_P2_V2:
.float 2.4121210200e-01
EXP_P3_V2:
.float 6.9307905933e-01
EXP_P4_V2:
.float 1.0000001462e+00
EXP_ONE:
.float 1
/************************************************************************/


/************************************************************************/
.text
.globl ASM_FUNC(sage2_tanh_ss)
ASM_FUNC(sage2_tanh_ss):
/************************************************************************/
// tanh = (exp(2x) - 1) / (exp(2x) + 1)
vmovss EXP_ONE(%rip), %xmm15

vaddss %xmm0, %xmm0, %xmm0

vminss EXP_HI(%rip), %xmm0, %xmm0
vmaxss EXP_LO(%rip), %xmm0, %xmm0
vmulss EXP_INV_LOG2E(%rip), %xmm0, %xmm1
vroundss $1, %xmm1, %xmm1, %xmm2
vsubss %xmm2, %xmm1, %xmm1
vmovss EXP_P0_V2(%rip), %xmm3
vfmadd213ss EXP_P1_V2(%rip), %xmm1, %xmm3
vfmadd213ss EXP_P2_V2(%rip), %xmm1, %xmm3
vfmadd213ss EXP_P3_V2(%rip), %xmm1, %xmm3
vfmadd213ss EXP_P4_V2(%rip), %xmm1, %xmm3
vcvttps2dq %xmm2, %xmm2
vpslld $23, %xmm2, %xmm2
vpaddd %xmm15, %xmm2, %xmm2
vmulss %xmm2, %xmm3, %xmm0

vaddss %xmm15, %xmm0, %xmm1
vsubss %xmm15, %xmm0, %xmm2
vdivss %xmm1, %xmm2, %xmm0

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_tanh_ps)
ASM_FUNC(sage2_tanh_ps):
/************************************************************************/
#define N_REG           %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define I_REG           %r8
#define M_REG           %r9

vbroadcastss EXP_HI(%rip), %ymm8
vbroadcastss EXP_LO(%rip), %ymm9
vbroadcastss EXP_INV_LOG2E(%rip), %ymm10
vbroadcastss EXP_P1_V2(%rip), %ymm11
vbroadcastss EXP_P2_V2(%rip), %ymm12
vbroadcastss EXP_P3_V2(%rip), %ymm13
vbroadcastss EXP_P4_V2(%rip), %ymm14
vbroadcastss EXP_ONE(%rip), %ymm15

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 2f

1:
vmovups (X_PTR,I_REG,4), %ymm0

vaddps %ymm0, %ymm0, %ymm0

vminps %ymm8, %ymm0, %ymm0
vmaxps %ymm9, %ymm0, %ymm0
vmulps %ymm10, %ymm0, %ymm1
vroundps $1, %ymm1, %ymm2
vsubps %ymm2, %ymm1, %ymm1
vbroadcastss EXP_P0_V2(%rip), %ymm3
vfmadd213ps %ymm11, %ymm1, %ymm3
vfmadd213ps %ymm12, %ymm1, %ymm3
vfmadd213ps %ymm13, %ymm1, %ymm3
vfmadd213ps %ymm14, %ymm1, %ymm3
vcvttps2dq %ymm2, %ymm2
vpslld $23, %ymm2, %ymm2
vpaddd %ymm15, %ymm2, %ymm2
vmulps %ymm2, %ymm3, %ymm0

vaddps %ymm15, %ymm0, %ymm1
vsubps %ymm15, %ymm0, %ymm2
vdivps %ymm1, %ymm2, %ymm0

vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG
subq $8, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $4, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %xmm0

vaddps %xmm0, %xmm0, %xmm0

vminps %xmm8, %xmm0, %xmm0
vmaxps %xmm9, %xmm0, %xmm0
vmulps %xmm10, %xmm0, %xmm1
vroundps $1, %xmm1, %xmm2
vsubps %xmm2, %xmm1, %xmm1
vbroadcastss EXP_P0_V2(%rip), %xmm3
vfmadd213ps %xmm11, %xmm1, %xmm3
vfmadd213ps %xmm12, %xmm1, %xmm3
vfmadd213ps %xmm13, %xmm1, %xmm3
vfmadd213ps %xmm14, %xmm1, %xmm3
vcvttps2dq %xmm2, %xmm2
vpslld $23, %xmm2, %xmm2
vpaddd %xmm15, %xmm2, %xmm2
vmulps %xmm2, %xmm3, %xmm0

vaddps %xmm15, %xmm0, %xmm1
vsubps %xmm15, %xmm0, %xmm2
vdivps %xmm1, %xmm2, %xmm0

vmovups %xmm0, (Y_PTR,I_REG,4)
subq $-4, I_REG

10:
movq N_REG, M_REG
andq $3, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0

vaddss %xmm0, %xmm0, %xmm0

vminss %xmm0, %xmm8, %xmm0
vmaxss %xmm0, %xmm9, %xmm0
vmulss %xmm10, %xmm0, %xmm1
vroundss $1, %xmm1, %xmm1, %xmm2
vsubss %xmm2, %xmm1, %xmm1
vmovss EXP_P0_V2(%rip), %xmm3
vfmadd213ss %xmm11, %xmm1, %xmm3
vfmadd213ss %xmm12, %xmm1, %xmm3
vfmadd213ss %xmm13, %xmm1, %xmm3
vfmadd213ss %xmm14, %xmm1, %xmm3
vcvttps2dq %xmm2, %xmm2
vpslld $23, %xmm2, %xmm2
vpaddd %xmm15, %xmm2, %xmm2
vmulss %xmm2, %xmm3, %xmm0

vaddss %xmm15, %xmm0, %xmm1
vsubss %xmm15, %xmm0, %xmm2
vdivss %xmm1, %xmm2, %xmm0

vmovss %xmm0, (Y_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq
#undef N_REG
#undef X_PTR
#undef Y_PTR
#undef I_REG
#undef M_REG
