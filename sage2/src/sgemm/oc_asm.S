// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"
#include "sgemm/offset.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_oc_Zrm)
ASM_FUNC(sage2_sgemm_oc_Zrm):
/************************************************************************/
#define CTX_PTR         %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define Z_PTR           %rcx
#define J_REG           %r8
#define M_REG_INT       %r9d
#define M_REG           %r9
#define N_REG_INT       %r10d
#define N_REG           %r10
#define NN_REG          %r11
#define Z_INC_ROW_REG_INT %edi
#define Z_INC_ROW_REG   %rdi
#define XI_REG_YMM      %ymm0
#define XI_REG_XMM      %xmm0
#define ALPHA_REG_YMM   %ymm1
#define BETA_REG_YMM    %ymm2
#define BETA_REG_XMM    %xmm2

movl M_OFFSET(CTX_PTR), M_REG_INT
movl N_OFFSET(CTX_PTR), N_REG_INT
vbroadcastss ALPHA_OFFSET(CTX_PTR), ALPHA_REG_YMM
vbroadcastss BETA_OFFSET(CTX_PTR), BETA_REG_YMM
movl Z_INC_ROW_OFFSET(CTX_PTR), Z_INC_ROW_REG_INT

1:
vbroadcastss (X_PTR), XI_REG_YMM
vmulps XI_REG_YMM, ALPHA_REG_YMM, XI_REG_YMM

xorq J_REG, J_REG
movq N_REG, NN_REG
andq $-8, NN_REG
je 11f

10:
vmulps (Z_PTR,J_REG,4), BETA_REG_YMM, %ymm3
vfmadd231ps (Y_PTR,J_REG,4), XI_REG_YMM, %ymm3
vmovups %ymm3, (Z_PTR,J_REG,4)
addq $8, J_REG
subq $8, NN_REG
jne 10b

11:
movq N_REG, NN_REG
andq $7, NN_REG
je 2f

12:
vmulss (Z_PTR,J_REG,4), BETA_REG_XMM, %xmm3
vfmadd231ss (Y_PTR,J_REG,4), XI_REG_XMM, %xmm3
vmovss %xmm3, (Z_PTR,J_REG,4)
addq $1, J_REG
subq $1, NN_REG
jne 12b

2:
addq $4, X_PTR
leaq (Z_PTR,Z_INC_ROW_REG,4), Z_PTR
subq $1, M_REG
jne 1b

vzeroupper
retq

#undef CTX_PTR
#undef X_PTR
#undef Y_PTR
#undef Z_PTR
#undef J_REG
#undef M_REG_INT
#undef M_REG
#undef N_REG_INT
#undef N_REG
#undef NN_REG
#undef Z_INC_ROW_REG_INT
#undef Z_INC_ROW_REG
#undef XI_REG_YMM
#undef XI_REG_XMM
#undef ALPHA_REG_YMM
#undef BETA_REG_YMM
#undef BETA_REG_XMM


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_oc_Zcm)
ASM_FUNC(sage2_sgemm_oc_Zcm):
/************************************************************************/
#define CTX_PTR         %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define Z_PTR           %rcx
#define I_REG           %r8
#define M_REG_INT       %r9d
#define M_REG           %r9
#define N_REG_INT       %r10d
#define N_REG           %r10
#define MM_REG          %r11
#define Z_INC_COL_REG_INT %edi
#define Z_INC_COL_REG   %rdi
#define YI_REG_YMM      %ymm0
#define YI_REG_XMM      %xmm0
#define ALPHA_REG_YMM   %ymm1
#define BETA_REG_YMM    %ymm2
#define BETA_REG_XMM    %xmm2

movl M_OFFSET(CTX_PTR), M_REG_INT
movl N_OFFSET(CTX_PTR), N_REG_INT
vbroadcastss ALPHA_OFFSET(CTX_PTR), ALPHA_REG_YMM
vbroadcastss BETA_OFFSET(CTX_PTR), BETA_REG_YMM
movl Z_INC_COL_OFFSET(CTX_PTR), Z_INC_COL_REG_INT

1:
vbroadcastss (Y_PTR), YI_REG_YMM
vmulps YI_REG_YMM, ALPHA_REG_YMM, YI_REG_YMM

xorq I_REG, I_REG
movq M_REG, MM_REG
andq $-8, MM_REG
je 11f

10:
vmulps (Z_PTR,I_REG,4), BETA_REG_YMM, %ymm3
vfmadd231ps (X_PTR,I_REG,4), YI_REG_YMM, %ymm3
vmovups %ymm3, (Z_PTR,I_REG,4)
addq $8, I_REG
subq $8, MM_REG
jne 10b

11:
movq M_REG, MM_REG
andq $7, MM_REG
je 2f

12:
vmulss (Z_PTR,I_REG,4), BETA_REG_XMM, %xmm3
vfmadd231ss (X_PTR,I_REG,4), YI_REG_XMM, %xmm3
vmovss %xmm3, (Z_PTR,I_REG,4)
addq $1, I_REG
subq $1, MM_REG
jne 12b

2:
addq $4, Y_PTR
leaq (Z_PTR,Z_INC_COL_REG,4), Z_PTR
subq $1, N_REG
jne 1b

vzeroupper
retq

#undef CTX_PTR
#undef X_PTR
#undef Y_PTR
#undef Z_PTR
#undef I_REG
#undef M_REG_INT
#undef M_REG
#undef N_REG_INT
#undef N_REG
#undef MM_REG
#undef Z_INC_COL_REG_INT
#undef Z_INC_COL_REG
#undef YI_REG_YMM
#undef YI_REG_XMM
#undef ALPHA_REG_YMM
#undef BETA_REG_YMM
#undef BETA_REG_XMM
