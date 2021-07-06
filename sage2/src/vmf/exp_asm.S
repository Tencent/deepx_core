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
EXP_C0_V1:
.float -0.693359375
EXP_C1_V1:
.float 2.12194440e-4
EXP_P0_V1:
.float 2.7565422393e-04
EXP_P1_V1:
.float 1.3038713518e-03
EXP_P2_V1:
.float 8.3795212816e-03
EXP_P3_V1:
.float 4.1653515712e-02
EXP_P4_V1:
.float 1.6666851064e-01
EXP_P5_V1:
.float 4.9999990238e-01
EXP_P0_V2:
.float 8.2886153525e-14
EXP_P1_V2:
.float 7.7822959126e-02
EXP_P2_V2:
.float 2.2586729288e-01
EXP_P3_V2:
.float 6.9617327373e-01
EXP_P4_V2:
.float 9.9986347636e-01
EXP_ONE:
.float 1
/************************************************************************/


/************************************************************************/
.text
.globl ASM_FUNC(sage2_exp_ss1)
ASM_FUNC(sage2_exp_ss1):
/************************************************************************/
vmovss EXP_ONE(%rip), %xmm15

vminss EXP_HI(%rip), %xmm0, %xmm0
vmaxss EXP_LO(%rip), %xmm0, %xmm0
vmulss EXP_INV_LOG2E(%rip), %xmm0, %xmm1
vroundss $1, %xmm1, %xmm1, %xmm1
vfmadd231ss EXP_C0_V1(%rip), %xmm1, %xmm0
vfmadd231ss EXP_C1_V1(%rip), %xmm1, %xmm0
vmulss %xmm0, %xmm0, %xmm2
vmovss EXP_P0_V1(%rip), %xmm3
vfmadd213ss EXP_P1_V1(%rip), %xmm0, %xmm3
vfmadd213ss EXP_P2_V1(%rip), %xmm0, %xmm3
vfmadd213ss EXP_P3_V1(%rip), %xmm0, %xmm3
vfmadd213ss EXP_P4_V1(%rip), %xmm0, %xmm3
vfmadd213ss EXP_P5_V1(%rip), %xmm0, %xmm3
vfmadd132ss %xmm2, %xmm0, %xmm3
vaddss %xmm15, %xmm3, %xmm3
vcvttps2dq %xmm1, %xmm1
vpslld $23, %xmm1, %xmm1
vpaddd %xmm15, %xmm1, %xmm1
vmulss %xmm1, %xmm3, %xmm0

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_exp_ps1)
ASM_FUNC(sage2_exp_ps1):
/************************************************************************/
#define N_REG           %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define I_REG           %r8
#define M_REG           %r9

vbroadcastss EXP_HI(%rip), %ymm5
vbroadcastss EXP_LO(%rip), %ymm6
vbroadcastss EXP_INV_LOG2E(%rip), %ymm7
vbroadcastss EXP_C0_V1(%rip), %ymm8
vbroadcastss EXP_C1_V1(%rip), %ymm9
vbroadcastss EXP_P1_V1(%rip), %ymm10
vbroadcastss EXP_P2_V1(%rip), %ymm11
vbroadcastss EXP_P3_V1(%rip), %ymm12
vbroadcastss EXP_P4_V1(%rip), %ymm13
vbroadcastss EXP_P5_V1(%rip), %ymm14
vbroadcastss EXP_ONE(%rip), %ymm15

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-8, M_REG
je 10f

1:
// ymm0: x
// ymm1: a, pow2n
// ymm2: b
// ymm3: c
vmovups (X_PTR,I_REG,4), %ymm0

// x = (x > EXP_HI) ? EXP_HI : x;
vminps %ymm5, %ymm0, %ymm0
// x = (x < EXP_LO) ? EXP_LO : x;
vmaxps %ymm6, %ymm0, %ymm0
// a = x * EXP_INV_LOG2E;
vmulps %ymm7, %ymm0, %ymm1
// a = floorf(a);
vroundps $1, %ymm1, %ymm1
// x = x + a * EXP_C0;
vfmadd231ps %ymm8, %ymm1, %ymm0
// x = x + a * EXP_C1;
vfmadd231ps %ymm9, %ymm1, %ymm0
// b = x * x;
vmulps %ymm0, %ymm0, %ymm2
// c = EXP_P0;
// c = c * x + EXP_P1;
// c = c * x + EXP_P2;
// c = c * x + EXP_P3;
// c = c * x + EXP_P4;
// c = c * x + EXP_P5;
vbroadcastss EXP_P0_V1(%rip), %ymm3
vfmadd213ps %ymm10, %ymm0, %ymm3
vfmadd213ps %ymm11, %ymm0, %ymm3
vfmadd213ps %ymm12, %ymm0, %ymm3
vfmadd213ps %ymm13, %ymm0, %ymm3
vfmadd213ps %ymm14, %ymm0, %ymm3
// c = c * b + x;
vfmadd132ps %ymm2,  %ymm0, %ymm3
// c = c + EXP_ONE.f;
vaddps %ymm15, %ymm3, %ymm3
// pow2n.u = (int)a;
vcvttps2dq %ymm1, %ymm1
// pow2n.u = pow2n.u << 23;
vpslld $23, %ymm1, %ymm1
// pow2n.u = pow2n.u + EXP_ONE.u;
vpaddd %ymm15, %ymm1, %ymm1
// c = c * pow2n.f;
vmulps %ymm1, %ymm3, %ymm0

vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG
subq $8, M_REG
jne 1b

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0

vminss %xmm5, %xmm0, %xmm0
vmaxss %xmm6, %xmm0, %xmm0
vmulss %xmm7, %xmm0, %xmm1
vroundss $1, %xmm1, %xmm1, %xmm1
vfmadd231ss %xmm8, %xmm1, %xmm0
vfmadd231ss %xmm9, %xmm1, %xmm0
vmulss %xmm0, %xmm0, %xmm2
vmovss EXP_P0_V1(%rip), %xmm3
vfmadd213ss %xmm10, %xmm0, %xmm3
vfmadd213ss %xmm11, %xmm0, %xmm3
vfmadd213ss %xmm12, %xmm0, %xmm3
vfmadd213ss %xmm13, %xmm0, %xmm3
vfmadd213ss %xmm14, %xmm0, %xmm3
vfmadd132ss %xmm2,  %xmm0, %xmm3
vaddss %xmm15, %xmm3, %xmm3
vcvttps2dq %xmm1, %xmm1
vpslld $23, %xmm1, %xmm1
vpaddd %xmm15, %xmm1, %xmm1
vmulss %xmm1, %xmm3, %xmm0

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


/************************************************************************/
.text
.globl ASM_FUNC(sage2_exp_ss)
.globl ASM_FUNC(sage2_exp_ss2)
ASM_FUNC(sage2_exp_ss):
ASM_FUNC(sage2_exp_ss2):
/************************************************************************/
vmovss EXP_ONE(%rip), %xmm15

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

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_exp_ps)
.globl ASM_FUNC(sage2_exp_ps2)
ASM_FUNC(sage2_exp_ps):
ASM_FUNC(sage2_exp_ps2):
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
// ymm0: x
// ymm1: a, b
// ymm2: fa, pow2n
// ymm3: c
vmovups (X_PTR,I_REG,4), %ymm0

// x = (x > EXP_HI) ? EXP_HI : x;
vminps %ymm8, %ymm0, %ymm0
// x = (x < EXP_LO) ? EXP_LO : x;
vmaxps %ymm9, %ymm0, %ymm0
// a = x * EXP_INV_LOG2E;
vmulps %ymm10, %ymm0, %ymm1
// fa = floorf(a);
vroundps $1, %ymm1, %ymm2
// b = a - fa;
vsubps %ymm2, %ymm1, %ymm1
// c = EXP_P0;
// c = c * x + EXP_P1;
// c = c * x + EXP_P2;
// c = c * x + EXP_P3;
// c = c * x + EXP_P4;
vbroadcastss EXP_P0_V2(%rip), %ymm3
vfmadd213ps %ymm11, %ymm1, %ymm3
vfmadd213ps %ymm12, %ymm1, %ymm3
vfmadd213ps %ymm13, %ymm1, %ymm3
vfmadd213ps %ymm14, %ymm1, %ymm3
// pow2n.u = (int)fa;
vcvttps2dq %ymm2, %ymm2
// pow2n.u = pow2n.u << 23;
vpslld $23, %ymm2, %ymm2
// pow2n.u = pow2n.u + EXP_ONE.u;
vpaddd %ymm15, %ymm2, %ymm2
// c = c * pow2n.f;
vmulps %ymm2, %ymm3, %ymm0

vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG
subq $8, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $4, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %xmm0

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

vmovups %xmm0, (Y_PTR,I_REG,4)
subq $-4, I_REG

10:
movq N_REG, M_REG
andq $3, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0

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


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sigmoid_ss)
ASM_FUNC(sage2_sigmoid_ss):
/************************************************************************/
// sigmoid = exp(x) / (exp(x) + 1)
vmovss EXP_ONE(%rip), %xmm15

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
vdivss %xmm1, %xmm0, %xmm0

vzeroupper
retq

/************************************************************************/
.text
.globl ASM_FUNC(sage2_sigmoid_ps)
ASM_FUNC(sage2_sigmoid_ps):
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
vdivps %ymm1, %ymm0, %ymm0

vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG
subq $8, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $4, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %xmm0

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
vdivps %xmm1, %xmm0, %xmm0

vmovups %xmm0, (Y_PTR,I_REG,4)
subq $-4, I_REG

10:
movq N_REG, M_REG
andq $3, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0

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
vdivss %xmm1, %xmm0, %xmm0

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
