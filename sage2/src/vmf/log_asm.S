// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
LOG_LO:
.float 1.0e-6
LOG_C0:
.float 0.7071067811865476
LOG_C1:
.float 0.6931471824645996
LOG_P0:
.float +7.0376836292e-2
LOG_P1:
.float -1.1514610310e-1
LOG_P2:
.float +1.1676998740e-1
LOG_P3:
.float -1.2420140846e-1
LOG_P4:
.float +1.4249322787e-1
LOG_P5:
.float -1.6668057665e-1
LOG_P6:
.float +2.0000714765e-1
LOG_P7:
.float -2.4999993993e-1
LOG_P8:
.float +3.3333331174e-1
LOG_INV_MANTISSA_MASK:
.int ~0x7f800000
LOG_EXP_OFFSET:
.int -127
LOG_ONE:
.float 1
LOG_0P5:
.float 0.5
/************************************************************************/


/************************************************************************/
.text
.globl ASM_FUNC(sage2_log_ss)
ASM_FUNC(sage2_log_ss):
/************************************************************************/
vmovss LOG_EXP_OFFSET(%rip), %xmm12
vmovss LOG_INV_MANTISSA_MASK(%rip), %xmm13
vmovss LOG_ONE(%rip), %xmm14
vmovss LOG_0P5(%rip), %xmm15

vmaxss LOG_LO(%rip), %xmm0, %xmm0
vpsrld $23, %xmm0, %xmm1
vpaddd %xmm12, %xmm1, %xmm1
vcvtdq2ps %xmm1, %xmm1
vaddss %xmm14, %xmm1, %xmm1
vandps %xmm13, %xmm0, %xmm0
vorps %xmm15, %xmm0, %xmm0
vcmpltss LOG_C0(%rip), %xmm0, %xmm2
vandps %xmm0, %xmm2, %xmm3
vaddss %xmm3, %xmm0, %xmm0
vsubss %xmm14, %xmm0, %xmm0
vandps %xmm14, %xmm2, %xmm3
vsubss %xmm3, %xmm1, %xmm1
vmovss LOG_P0(%rip), %xmm2
vfmadd213ss LOG_P1(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P2(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P3(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P4(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P5(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P6(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P7(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P8(%rip), %xmm0, %xmm2
vmulss %xmm0, %xmm0, %xmm3
vfmsub213ss %xmm15, %xmm0, %xmm2
vfmadd231ss LOG_C1(%rip), %xmm1, %xmm0
vfmadd231ss %xmm3, %xmm2, %xmm0

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_log_ps)
ASM_FUNC(sage2_log_ps):
/************************************************************************/
#define N_REG           %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define I_REG           %r8
#define M_REG           %r9

vbroadcastss LOG_EXP_OFFSET(%rip), %ymm12
vbroadcastss LOG_INV_MANTISSA_MASK(%rip), %ymm13
vbroadcastss LOG_ONE(%rip), %ymm14
vbroadcastss LOG_0P5(%rip), %ymm15

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
// ymm0, ymm4: x
// ymm1, ymm5: a, e
// ymm2, ymm6: mask, y
// ymm3, ymm7: dx, de, z
// ymm8: constants
vmovups   (X_PTR,I_REG,4), %ymm0
vmovups 32(X_PTR,I_REG,4), %ymm4

// x = (x < LOG_LO) ? LOG_LO : x;
vbroadcastss LOG_LO(%rip), %ymm8
vmaxps %ymm8, %ymm0, %ymm0
vmaxps %ymm8, %ymm4, %ymm4

// a.f = x;
// a.u = a.u >> 23;
vpsrld $23, %ymm0, %ymm1
vpsrld $23, %ymm4, %ymm5
// a.u = a.u + LOG_EXP_OFFSET;
vpaddd %ymm12, %ymm1, %ymm1
vpaddd %ymm12, %ymm5, %ymm5
// a.f = (float)(int)a.u;
vcvtdq2ps %ymm1, %ymm1
vcvtdq2ps %ymm5, %ymm5
// a.f = a.f + LOG_ONE;
// e = a.f;
vaddps %ymm14, %ymm1, %ymm1
vaddps %ymm14, %ymm5, %ymm5

// a.f = x;
// a.u = a.u & LOG_INV_MANTISSA_MASK;
vandps %ymm13, %ymm0, %ymm0
vandps %ymm13, %ymm4, %ymm4
// a.u = a.u | LOG_0P5.u;
// x = a.f;
vorps %ymm15, %ymm0, %ymm0
vorps %ymm15, %ymm4, %ymm4

// mask = (x < LOG_C0) ? 0xffffffff : 0
vbroadcastss LOG_C0(%rip), %ymm8
vcmpltps %ymm8, %ymm0, %ymm2
vcmpltps %ymm8, %ymm4, %ymm6

// dx = mask & x;
vandps %ymm0, %ymm2, %ymm3
vandps %ymm4, %ymm6, %ymm7
// x = x + dx - LOG_ONE;
vaddps %ymm3, %ymm0, %ymm0
vaddps %ymm7, %ymm4, %ymm4
vsubps %ymm14, %ymm0, %ymm0
vsubps %ymm14, %ymm4, %ymm4

// de = mask & LOG_ONE;
vandps %ymm14, %ymm2, %ymm3
vandps %ymm14, %ymm6, %ymm7
// e = e - de;
vsubps %ymm3, %ymm1, %ymm1
vsubps %ymm7, %ymm5, %ymm5

// y = LOG_P0;
// y = y * x + LOG_P1;
// y = y * x + LOG_P2;
// y = y * x + LOG_P3;
// y = y * x + LOG_P4;
// y = y * x + LOG_P5;
// y = y * x + LOG_P6;
// y = y * x + LOG_P7;
// y = y * x + LOG_P8;
vbroadcastss LOG_P0(%rip), %ymm2
vbroadcastss LOG_P0(%rip), %ymm6
vbroadcastss LOG_P1(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6
vbroadcastss LOG_P2(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6
vbroadcastss LOG_P3(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6
vbroadcastss LOG_P4(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6
vbroadcastss LOG_P5(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6
vbroadcastss LOG_P6(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6
vbroadcastss LOG_P7(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6
vbroadcastss LOG_P8(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vfmadd213ps %ymm8, %ymm4, %ymm6

// z = x * x;
vmulps %ymm0, %ymm0, %ymm3
vmulps %ymm4, %ymm4, %ymm7
// y = y * x - LOG_0P5;
vfmsub213ps %ymm15, %ymm0, %ymm2
vfmsub213ps %ymm15, %ymm4, %ymm6
// x = e * LOG_C1 + x;
vbroadcastss LOG_C1(%rip), %ymm8
vfmadd231ps %ymm8, %ymm1, %ymm0
vfmadd231ps %ymm8, %ymm5, %ymm4
// x = y * z + x;
vfmadd231ps %ymm3, %ymm2, %ymm0
vfmadd231ps %ymm7, %ymm6, %ymm4

vmovups %ymm0,   (Y_PTR,I_REG,4)
vmovups %ymm4, 32(Y_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmovups (X_PTR,I_REG,4), %ymm0
vbroadcastss LOG_LO(%rip), %ymm8
vmaxps %ymm8, %ymm0, %ymm0
vpsrld $23, %ymm0, %ymm1
vpaddd %ymm12, %ymm1, %ymm1
vcvtdq2ps %ymm1, %ymm1
vaddps %ymm14, %ymm1, %ymm1
vandps %ymm13, %ymm0, %ymm0
vorps %ymm15, %ymm0, %ymm0
vbroadcastss LOG_C0(%rip), %ymm8
vcmpltps %ymm8, %ymm0, %ymm2
vandps %ymm0, %ymm2, %ymm3
vaddps %ymm3, %ymm0, %ymm0
vsubps %ymm14, %ymm0, %ymm0
vandps %ymm14, %ymm2, %ymm3
vsubps %ymm3, %ymm1, %ymm1
vbroadcastss LOG_P0(%rip), %ymm2
vbroadcastss LOG_P1(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vbroadcastss LOG_P2(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vbroadcastss LOG_P3(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vbroadcastss LOG_P4(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vbroadcastss LOG_P5(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vbroadcastss LOG_P6(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vbroadcastss LOG_P7(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vbroadcastss LOG_P8(%rip), %ymm8
vfmadd213ps %ymm8, %ymm0, %ymm2
vmulps %ymm0, %ymm0, %ymm3
vfmsub213ps %ymm15, %ymm0, %ymm2
vbroadcastss LOG_C1(%rip), %ymm8
vfmadd231ps %ymm8, %ymm1, %ymm0
vfmadd231ps %ymm3, %ymm2, %ymm0
vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm0
vmaxss LOG_LO(%rip), %xmm0, %xmm0
vpsrld $23, %xmm0, %xmm1
vpaddd %xmm12, %xmm1, %xmm1
vcvtdq2ps %xmm1, %xmm1
vaddss %xmm14, %xmm1, %xmm1
vandps %xmm13, %xmm0, %xmm0
vorps %xmm15, %xmm0, %xmm0
vcmpltss LOG_C0(%rip), %xmm0, %xmm2
vandps %xmm0, %xmm2, %xmm3
vaddss %xmm3, %xmm0, %xmm0
vsubss %xmm14, %xmm0, %xmm0
vandps %xmm14, %xmm2, %xmm3
vsubss %xmm3, %xmm1, %xmm1
vmovss LOG_P0(%rip), %xmm2
vfmadd213ss LOG_P1(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P2(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P3(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P4(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P5(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P6(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P7(%rip), %xmm0, %xmm2
vfmadd213ss LOG_P8(%rip), %xmm0, %xmm2
vmulss %xmm0, %xmm0, %xmm3
vfmsub213ss %xmm15, %xmm0, %xmm2
vfmadd231ss LOG_C1(%rip), %xmm1, %xmm0
vfmadd231ss %xmm3, %xmm2, %xmm0
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
