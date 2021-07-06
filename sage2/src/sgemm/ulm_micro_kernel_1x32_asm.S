// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
SGEMM_ULM_ONE:
.float 1
/************************************************************************/


#define KC_REG          %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define Z_PTR           %rcx
#define Z_INC_ROW_REG   %r8
#define Z_INC_COL_REG   %r9
#define ALPHA_REG       %xmm0
#define BETA_REG        %xmm1
#define XY_PTR          %rdi
#define I_REG           %rsi
#define J_REG           %rdx


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_ulm_micro_kernel_1x32)
ASM_FUNC(sage2_sgemm_ulm_micro_kernel_1x32):
/************************************************************************/
// part 1
// XY(ymm8-ymm15) = 0
vxorps %ymm8,  %ymm8,  %ymm8
vxorps %ymm9,  %ymm9,  %ymm9
vxorps %ymm10, %ymm10, %ymm10
vxorps %ymm11, %ymm11, %ymm11

// part 2
// XY(ymm8-ymm15) += X(memory) * Y(memory)
1:
vbroadcastss (X_PTR), %ymm2
vfmadd231ps   (Y_PTR), %ymm2, %ymm8
vfmadd231ps 32(Y_PTR), %ymm2, %ymm9
vfmadd231ps 64(Y_PTR), %ymm2, %ymm10
vfmadd231ps 96(Y_PTR), %ymm2, %ymm11
addq $4, X_PTR                          // X += MR
addq $128, Y_PTR                        // Y += NR
subq $1, KC_REG
jne 1b

vcomiss SGEMM_ULM_ONE(%rip), ALPHA_REG  // if alpha == 1
je 2f

// part 3
// alpha != 1
// XY(ymm8-ymm15) *= alpha
vmovss ALPHA_REG, -4(%rsp)
vbroadcastss -4(%rsp), %ymm2
vmulps %ymm2, %ymm8,  %ymm8
vmulps %ymm2, %ymm9,  %ymm9
vmulps %ymm2, %ymm10, %ymm10
vmulps %ymm2, %ymm11, %ymm11

// part 4
2:
cmpq $1, Z_INC_COL_REG                  // if Z_inc_col == 1
jne 20f

10:
// part 4.1
// Z_inc_col == 1
vxorps %xmm2, %xmm2, %xmm2
vcomiss %xmm2, BETA_REG                 // if beta == 0
je 11f

// part 4.1.1
// Z_inc_col == 1
// beta != 0
// Z(ymm8-ymm15) = XY(ymm8-ymm15) + beta * Z(memory)
vmovss BETA_REG, -4(%rsp)
vbroadcastss -4(%rsp), %ymm2
vfmadd231ps   (Z_PTR), %ymm2, %ymm8
vfmadd231ps 32(Z_PTR), %ymm2, %ymm9
vfmadd231ps 64(Z_PTR), %ymm2, %ymm10
vfmadd231ps 96(Z_PTR), %ymm2, %ymm11

11:
// part 4.1.2
// Z_inc_col == 1
// Z(memory) = Z(ymm8-ymm15)
vmovups %ymm8,    (Z_PTR)
vmovups %ymm9,  32(Z_PTR)
vmovups %ymm10, 64(Z_PTR)
vmovups %ymm11, 96(Z_PTR)
vzeroupper
retq

20:
// part 4.2
// Z_inc_col != 1

// XY(stack) = XY(ymm8-ymm15)
subq $256, %rsp
movq %rsp, XY_PTR
vmovups %ymm8,     (XY_PTR)
vmovups %ymm9,   32(XY_PTR)
vmovups %ymm10,  64(XY_PTR)
vmovups %ymm11,  96(XY_PTR)

movq $1, I_REG                          // i = MR
vxorps %xmm2, %xmm2, %xmm2
vcomiss %xmm2, BETA_REG                 // if beta == 0
je 22f

// part 4.2.1
// Z_inc_col != 1
// beta != 0
// Z(memory) = XY(stack) + beta * Z(memory)
21:
#define HELPER_0()                              \
    xorq J_REG, J_REG;                          \
    vmovss (Z_PTR,J_REG,4), %xmm2;              \
    vfmadd213ss (XY_PTR), BETA_REG, %xmm2;      \
    vmovss %xmm2, (Z_PTR,J_REG,4);
#define HELPER_I(i)                             \
    addq Z_INC_COL_REG, J_REG;                  \
    vmovss (Z_PTR,J_REG,4), %xmm2;              \
    vfmadd213ss i*4(XY_PTR), BETA_REG, %xmm2;   \
    vmovss %xmm2, (Z_PTR,J_REG,4);
HELPER_0()
HELPER_I(1)
HELPER_I(2)
HELPER_I(3)
HELPER_I(4)
HELPER_I(5)
HELPER_I(6)
HELPER_I(7)
HELPER_I(8)
HELPER_I(9)
HELPER_I(10)
HELPER_I(11)
HELPER_I(12)
HELPER_I(13)
HELPER_I(14)
HELPER_I(15)
HELPER_I(16)
HELPER_I(17)
HELPER_I(18)
HELPER_I(19)
HELPER_I(20)
HELPER_I(21)
HELPER_I(22)
HELPER_I(23)
HELPER_I(24)
HELPER_I(25)
HELPER_I(26)
HELPER_I(27)
HELPER_I(28)
HELPER_I(29)
HELPER_I(30)
HELPER_I(31)
#undef HELPER_0
#undef HELPER_I
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
addq $128, XY_PTR                       // XY += NR
subq $1, I_REG
jne 21b
addq $256, %rsp
vzeroupper
retq

// part 4.2.2
// Z_inc_col != 1
// beta == 0
// Z(memory) = XY(stack)
22:
#define HELPER_0()                      \
    xorq J_REG, J_REG;                  \
    vmovss (XY_PTR), %xmm2;             \
    vmovss %xmm2, (Z_PTR,J_REG,4);
#define HELPER_I(i)                     \
    addq Z_INC_COL_REG, J_REG;          \
    vmovss i*4(XY_PTR), %xmm2;          \
    vmovss %xmm2, (Z_PTR,J_REG,4);
HELPER_0()
HELPER_I(1)
HELPER_I(2)
HELPER_I(3)
HELPER_I(4)
HELPER_I(5)
HELPER_I(6)
HELPER_I(7)
HELPER_I(8)
HELPER_I(9)
HELPER_I(10)
HELPER_I(11)
HELPER_I(12)
HELPER_I(13)
HELPER_I(14)
HELPER_I(15)
HELPER_I(16)
HELPER_I(17)
HELPER_I(18)
HELPER_I(19)
HELPER_I(20)
HELPER_I(21)
HELPER_I(22)
HELPER_I(23)
HELPER_I(24)
HELPER_I(25)
HELPER_I(26)
HELPER_I(27)
HELPER_I(28)
HELPER_I(29)
HELPER_I(30)
HELPER_I(31)
#undef HELPER_0
#undef HELPER_I
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
addq $128, XY_PTR                       // XY += NR
subq $1, I_REG
jne 22b
addq $256, %rsp
vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_ulm_micro_kernel_a1_Zrm_1x32)
ASM_FUNC(sage2_sgemm_ulm_micro_kernel_a1_Zrm_1x32):
/************************************************************************/
// part 1
// XY(ymm8-ymm15) = 0
vxorps %ymm8,  %ymm8,  %ymm8
vxorps %ymm9,  %ymm9,  %ymm9
vxorps %ymm10, %ymm10, %ymm10
vxorps %ymm11, %ymm11, %ymm11

// part 2
// XY(ymm8-ymm15) += X(memory) * Y(memory)
1:
vbroadcastss (X_PTR), %ymm2
vfmadd231ps   (Y_PTR), %ymm2, %ymm8
vfmadd231ps 32(Y_PTR), %ymm2, %ymm9
vfmadd231ps 64(Y_PTR), %ymm2, %ymm10
vfmadd231ps 96(Y_PTR), %ymm2, %ymm11
addq $4, X_PTR                          // X += MR
addq $128, Y_PTR                        // Y += NR
subq $1, KC_REG
jne 1b

// part 3
vxorps %xmm2, %xmm2, %xmm2
vcomiss %xmm2, BETA_REG                 // if beta == 0
je 10f

// part 3.1
// beta != 0
// Z(ymm8-ymm15) = XY(ymm8-ymm15) + beta * Z(memory)
vmovss BETA_REG, -4(%rsp)
vbroadcastss -4(%rsp), %ymm2
vfmadd231ps   (Z_PTR), %ymm2, %ymm8
vfmadd231ps 32(Z_PTR), %ymm2, %ymm9
vfmadd231ps 64(Z_PTR), %ymm2, %ymm10
vfmadd231ps 96(Z_PTR), %ymm2, %ymm11

10:
// part 3.2
// Z(memory) = Z(ymm8-ymm15)
vmovups %ymm8,    (Z_PTR)
vmovups %ymm9,  32(Z_PTR)
vmovups %ymm10, 64(Z_PTR)
vmovups %ymm11, 96(Z_PTR)
vzeroupper
retq
