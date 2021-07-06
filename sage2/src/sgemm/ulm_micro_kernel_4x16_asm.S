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
.globl ASM_FUNC(sage2_sgemm_ulm_micro_kernel_4x16)
ASM_FUNC(sage2_sgemm_ulm_micro_kernel_4x16):
/************************************************************************/
// part 1
// XY(ymm8-ymm15) = 0
vxorps %ymm8,  %ymm8,  %ymm8
vxorps %ymm9,  %ymm9,  %ymm9
vxorps %ymm10, %ymm10, %ymm10
vxorps %ymm11, %ymm11, %ymm11
vxorps %ymm12, %ymm12, %ymm12
vxorps %ymm13, %ymm13, %ymm13
vxorps %ymm14, %ymm14, %ymm14
vxorps %ymm15, %ymm15, %ymm15

// part 2
// XY(ymm8-ymm15) += X(memory) * Y(memory)
1:
vbroadcastss   (X_PTR), %ymm2           // X[0]
vbroadcastss  4(X_PTR), %ymm3           // X[1]
vbroadcastss  8(X_PTR), %ymm4           // X[2]
vbroadcastss 12(X_PTR), %ymm5           // X[3]
vmovups   (Y_PTR), %ymm6                // Y[0:7]
vmovups 32(Y_PTR), %ymm7                // Y[8:15]
vfmadd231ps %ymm2, %ymm6, %ymm8         // XY[0:7]   += X[0] * Y[0:7]
vfmadd231ps %ymm2, %ymm7, %ymm9         // XY[8:15]  += X[0] * Y[8:15]
vfmadd231ps %ymm3, %ymm6, %ymm10        // XY[16:23] += X[1] * Y[0:7]
vfmadd231ps %ymm3, %ymm7, %ymm11        // XY[24:31] += X[1] * Y[8:15]
vfmadd231ps %ymm4, %ymm6, %ymm12        // XY[32:39] += X[2] * Y[0:7]
vfmadd231ps %ymm4, %ymm7, %ymm13        // XY[40:47] += X[2] * Y[8:15]
vfmadd231ps %ymm5, %ymm6, %ymm14        // XY[48:55] += X[3] * Y[0:7]
vfmadd231ps %ymm5, %ymm7, %ymm15        // XY[56:63] += X[3] * Y[8:15]
addq $16, X_PTR                         // X += MR
addq $64, Y_PTR                         // Y += NR
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
vmulps %ymm2, %ymm12, %ymm12
vmulps %ymm2, %ymm13, %ymm13
vmulps %ymm2, %ymm14, %ymm14
vmulps %ymm2, %ymm15, %ymm15

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
movq Z_PTR, -8(%rsp)
vfmadd231ps   (Z_PTR), %ymm2, %ymm8
vfmadd231ps 32(Z_PTR), %ymm2, %ymm9
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vfmadd231ps   (Z_PTR), %ymm2, %ymm10
vfmadd231ps 32(Z_PTR), %ymm2, %ymm11
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vfmadd231ps   (Z_PTR), %ymm2, %ymm12
vfmadd231ps 32(Z_PTR), %ymm2, %ymm13
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vfmadd231ps   (Z_PTR), %ymm2, %ymm14
vfmadd231ps 32(Z_PTR), %ymm2, %ymm15
movq -8(%rsp), Z_PTR

11:
// part 4.1.2
// Z_inc_col == 1
// Z(memory) = Z(ymm8-ymm15)
vmovups %ymm8,    (Z_PTR)
vmovups %ymm9,  32(Z_PTR)
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vmovups %ymm10,   (Z_PTR)
vmovups %ymm11, 32(Z_PTR)
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vmovups %ymm12,   (Z_PTR)
vmovups %ymm13, 32(Z_PTR)
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vmovups %ymm14,   (Z_PTR)
vmovups %ymm15, 32(Z_PTR)
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
vmovups %ymm12, 128(XY_PTR)
vmovups %ymm13, 160(XY_PTR)
vmovups %ymm14, 192(XY_PTR)
vmovups %ymm15, 224(XY_PTR)

movq $4, I_REG                          // i = MR
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
#undef HELPER_0
#undef HELPER_I
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
addq $64, XY_PTR                        // XY += NR
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
#undef HELPER_0
#undef HELPER_I
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
addq $64, XY_PTR                        // XY += NR
subq $1, I_REG
jne 22b
addq $256, %rsp
vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_ulm_micro_kernel_a1_Zrm_4x16)
ASM_FUNC(sage2_sgemm_ulm_micro_kernel_a1_Zrm_4x16):
/************************************************************************/
// part 1
// XY(ymm8-ymm15) = 0
vxorps %ymm8,  %ymm8,  %ymm8
vxorps %ymm9,  %ymm9,  %ymm9
vxorps %ymm10, %ymm10, %ymm10
vxorps %ymm11, %ymm11, %ymm11
vxorps %ymm12, %ymm12, %ymm12
vxorps %ymm13, %ymm13, %ymm13
vxorps %ymm14, %ymm14, %ymm14
vxorps %ymm15, %ymm15, %ymm15

// part 2
// XY(ymm8-ymm15) += X(memory) * Y(memory)
1:
vbroadcastss   (X_PTR), %ymm2
vbroadcastss  4(X_PTR), %ymm3
vbroadcastss  8(X_PTR), %ymm4
vbroadcastss 12(X_PTR), %ymm5
vmovups   (Y_PTR), %ymm6
vmovups 32(Y_PTR), %ymm7
vfmadd231ps %ymm2, %ymm6, %ymm8
vfmadd231ps %ymm2, %ymm7, %ymm9
vfmadd231ps %ymm3, %ymm6, %ymm10
vfmadd231ps %ymm3, %ymm7, %ymm11
vfmadd231ps %ymm4, %ymm6, %ymm12
vfmadd231ps %ymm4, %ymm7, %ymm13
vfmadd231ps %ymm5, %ymm6, %ymm14
vfmadd231ps %ymm5, %ymm7, %ymm15
addq $16, X_PTR                         // X += MR
addq $64, Y_PTR                         // Y += NR
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
movq Z_PTR, -8(%rsp)
vfmadd231ps   (Z_PTR), %ymm2, %ymm8
vfmadd231ps 32(Z_PTR), %ymm2, %ymm9
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vfmadd231ps   (Z_PTR), %ymm2, %ymm10
vfmadd231ps 32(Z_PTR), %ymm2, %ymm11
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vfmadd231ps   (Z_PTR), %ymm2, %ymm12
vfmadd231ps 32(Z_PTR), %ymm2, %ymm13
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vfmadd231ps   (Z_PTR), %ymm2, %ymm14
vfmadd231ps 32(Z_PTR), %ymm2, %ymm15
movq -8(%rsp), Z_PTR

10:
// part 3.2
// Z(memory) = Z(ymm8-ymm15)
vmovups %ymm8,    (Z_PTR)
vmovups %ymm9,  32(Z_PTR)
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vmovups %ymm10,   (Z_PTR)
vmovups %ymm11, 32(Z_PTR)
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vmovups %ymm12,   (Z_PTR)
vmovups %ymm13, 32(Z_PTR)
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
vmovups %ymm14,   (Z_PTR)
vmovups %ymm15, 32(Z_PTR)
vzeroupper
retq
