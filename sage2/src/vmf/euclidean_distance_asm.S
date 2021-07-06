// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


#define N_REG           %rdi
#define X_PTR           %rsi
#define Y_PTR           %rdx
#define I_REG           %r8
#define NN_REG          %r9


/************************************************************************/
.text
.globl ASM_FUNC(sage2_euclidean_distance_ps)
ASM_FUNC(sage2_euclidean_distance_ps):
/************************************************************************/
vxorps %ymm0, %ymm0, %ymm0
xorq I_REG, I_REG
movq N_REG, NN_REG
andq $-16, NN_REG
je 2f

vxorps %ymm1, %ymm1, %ymm1
1:
vmovups   (X_PTR,I_REG,4), %ymm2
vmovups 32(X_PTR,I_REG,4), %ymm3
vsubps   (Y_PTR,I_REG,4), %ymm2, %ymm2
vsubps 32(Y_PTR,I_REG,4), %ymm3, %ymm3
vfmadd231ps %ymm2, %ymm2, %ymm0
vfmadd231ps %ymm3, %ymm3, %ymm1
subq $-16, I_REG
subq $16, NN_REG
jne 1b
vaddps %ymm0, %ymm1, %ymm0

2:
movq N_REG, NN_REG
andq $8, NN_REG
je 10f
vmovups (X_PTR,I_REG,4), %ymm1
vsubps (Y_PTR,I_REG,4), %ymm1, %ymm1
vfmadd231ps %ymm1, %ymm1, %ymm0
subq $-8, I_REG

10:
vextractf128 $1, %ymm0, %xmm1
vaddps %xmm0, %xmm1, %xmm0
vhaddps %xmm1, %xmm0, %xmm0
vhaddps %xmm1, %xmm0, %xmm0
movq N_REG, NN_REG
andq $7, NN_REG
je 12f

11:
vmovss (X_PTR,I_REG,4), %xmm1
vsubss (Y_PTR,I_REG,4), %xmm1, %xmm1
vfmadd231ss %xmm1, %xmm1, %xmm0
subq $-1, I_REG
subq $1, NN_REG
jne 11b

12:
vsqrtss %xmm0, %xmm0, %xmm0
vzeroupper
retq
