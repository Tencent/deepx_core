// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


#define MC_REG          %rdi
#define KC_REG          %rsi
#define X_PTR           %rdx
#define PACKED_X_PTR    %rcx
#define X_INC_ROW_REG   %r8             // unused
#define X_INC_COL_REG   %r9
#define _X_PTR          %r8
#define MP_REG          %r10
#define _KC_REG         %r11
#define _MR_REG         %rdi
#define I_REG           %r10


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_ulm_pack_X_Xc_2)
ASM_FUNC(sage2_sgemm_ulm_pack_X_Xc_2):
/************************************************************************/
// part 1
movq MC_REG, MP_REG
sarq $1, MP_REG                         // mp = mc / MR
je 10f

1:
movq X_PTR, _X_PTR
movq KC_REG, _KC_REG

2:
vmovss  (_X_PTR), %xmm0
vmovss 4(_X_PTR), %xmm1
vmovss %xmm0,  (PACKED_X_PTR)
vmovss %xmm1, 4(PACKED_X_PTR)
addq $8, PACKED_X_PTR                   // packed_X += MR
leaq (_X_PTR,X_INC_COL_REG,4), _X_PTR
subq $1, _KC_REG
jne 2b

addq $8, X_PTR                          // X += MR
subq $1, MP_REG
jne 1b

// part 2
10:
andq $7, _MR_REG                        // _mr = mc % MR
je 20f

vxorps %xmm0, %xmm0, %xmm0

11:
vmovss %xmm0,  (PACKED_X_PTR)
vmovss %xmm0, 4(PACKED_X_PTR)
xorq I_REG, I_REG

12:
vmovss (X_PTR,I_REG,4), %xmm1
vmovss %xmm1, (PACKED_X_PTR,I_REG,4)
addq $1, I_REG
cmpq I_REG, _MR_REG
jne 12b

addq $8, PACKED_X_PTR                   // packed_X += MR
leaq (X_PTR,X_INC_COL_REG,4), X_PTR
subq $1, KC_REG
jne 11b

20:
vzeroupper
retq
