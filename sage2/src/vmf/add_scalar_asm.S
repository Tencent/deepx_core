// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.data
SUB_SCALAR_SIGN_MASK:
.int 0x80000000
DIV_SCALAR_ONE:
.float 1
/************************************************************************/


#define N_REG           %rdi
#define X_PTR           %rsi
#define ALPHA_REG       %xmm0
#define Y_PTR           %rdx
#define I_REG           %r8
#define M_REG           %r9


/************************************************************************/
.text
.globl ASM_FUNC(sage2_add_scalar_ps)
ASM_FUNC(sage2_add_scalar_ps):
/************************************************************************/
vmovss ALPHA_REG, -4(%rsp)
vbroadcastss -4(%rsp), %ymm15

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vaddps   (X_PTR,I_REG,4), %ymm15, %ymm0
vaddps 32(X_PTR,I_REG,4), %ymm15, %ymm1
vmovups %ymm0,   (Y_PTR,I_REG,4)
vmovups %ymm1, 32(Y_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vaddps (X_PTR,I_REG,4), %ymm15, %ymm0
vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vaddss (X_PTR,I_REG,4), %xmm15, %xmm0
vmovss %xmm0, (Y_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sub_scalar_ps)
ASM_FUNC(sage2_sub_scalar_ps):
/************************************************************************/
vmovss SUB_SCALAR_SIGN_MASK(%rip), %xmm1
vxorps %xmm1, ALPHA_REG, ALPHA_REG
jmp ASM_FUNC(sage2_add_scalar_ps)


/************************************************************************/
.text
.globl ASM_FUNC(sage2_mul_scalar_ps)
ASM_FUNC(sage2_mul_scalar_ps):
/************************************************************************/
vmovss ALPHA_REG, -4(%rsp)
vbroadcastss -4(%rsp), %ymm15

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmulps   (X_PTR,I_REG,4), %ymm15, %ymm0
vmulps 32(X_PTR,I_REG,4), %ymm15, %ymm1
vmovups %ymm0,   (Y_PTR,I_REG,4)
vmovups %ymm1, 32(Y_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmulps (X_PTR,I_REG,4), %ymm15, %ymm0
vmovups %ymm0, (Y_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmulss (X_PTR,I_REG,4), %xmm15, %xmm0
vmovss %xmm0, (Y_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_div_scalar_ps)
ASM_FUNC(sage2_div_scalar_ps):
/************************************************************************/
vmovss DIV_SCALAR_ONE(%rip), %xmm1
vdivss ALPHA_REG, %xmm1, ALPHA_REG
jmp ASM_FUNC(sage2_mul_scalar_ps)
