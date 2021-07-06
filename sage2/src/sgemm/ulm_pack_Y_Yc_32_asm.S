// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


#define KC_REG          %rdi
#define NC_REG          %rsi
#define Y_PTR           %rdx
#define PACKED_Y_PTR    %rcx
#define Y_INC_ROW_REG   %r8
#define Y_INC_COL_REG   %r9             // unused
#define _Y_PTR          %r9
#define NP_REG          %r10
#define _KC_REG         %r11
#define _NR_REG         %rsi
#define J_REG           %r10


/************************************************************************/
.text
.globl ASM_FUNC(sage2_sgemm_ulm_pack_Y_Yc_32)
ASM_FUNC(sage2_sgemm_ulm_pack_Y_Yc_32):
/************************************************************************/
// part 1
movq NC_REG, NP_REG
sarq $5, NP_REG                         // np = nc / NR
je 10f

1:
movq Y_PTR, _Y_PTR
movq KC_REG, _KC_REG

2:
vmovups   (_Y_PTR), %ymm0
vmovups 32(_Y_PTR), %ymm1
vmovups 64(_Y_PTR), %ymm2
vmovups 96(_Y_PTR), %ymm3
vmovups %ymm0,   (PACKED_Y_PTR)
vmovups %ymm1, 32(PACKED_Y_PTR)
vmovups %ymm2, 64(PACKED_Y_PTR)
vmovups %ymm3, 96(PACKED_Y_PTR)
addq $128, PACKED_Y_PTR                 // packed_Y += NR
leaq (_Y_PTR,Y_INC_ROW_REG,4), _Y_PTR
subq $1, _KC_REG
jne 2b

addq $128, Y_PTR                        // Y += NR
subq $1, NP_REG
jne 1b

// part 2
10:
andq $31, _NR_REG                       // _nr = nc % NR
je 20f

vxorps %ymm0, %ymm0, %ymm0

11:
vmovups %ymm0,   (PACKED_Y_PTR)
vmovups %ymm0, 32(PACKED_Y_PTR)
xorq J_REG, J_REG

12:
vmovss (Y_PTR,J_REG,4), %xmm1
vmovss %xmm1, (PACKED_Y_PTR,J_REG,4)
addq $1, J_REG
cmpq J_REG, _NR_REG
jne 12b

addq $128, PACKED_Y_PTR                 // packed_Y += NR
leaq (Y_PTR,Y_INC_ROW_REG,4), Y_PTR
subq $1, KC_REG
jne 11b

20:
vzeroupper
retq
