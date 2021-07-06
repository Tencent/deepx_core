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


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_ulm_pack_X_Xc_1)
ASM_FUNC(sage2_sgemm_ulm_pack_X_Xc_1):
/************************************************************************/
// part 1
movq MC_REG, MP_REG

1:
movq X_PTR, _X_PTR
movq KC_REG, _KC_REG

2:
vmovss (_X_PTR), %xmm0
vmovss %xmm0, (PACKED_X_PTR)
addq $4, PACKED_X_PTR                   // packed_X += MR
leaq (_X_PTR,X_INC_COL_REG,4), _X_PTR
subq $1, _KC_REG
jne 2b

addq $4, X_PTR                          // X += MR
subq $1, MP_REG
jne 1b

// part 2
10:
vzeroupper
retq
