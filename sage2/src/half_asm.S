// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "internal_macro.h"


/************************************************************************/
.text
.globl ASM_FUNC(sage2_d2h)
ASM_FUNC(sage2_d2h):
/************************************************************************/
                                      // xmm0: 0 d
vxorps %xmm15, %xmm15, %xmm15
vcvtsd2ss %xmm0, %xmm15, %xmm0        // xmm0: 0 0 0 f
vcvtps2ph $0, %xmm0, %xmm0            // xmm0: 0 0 0 0h
vmovd %xmm0, %eax

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_h2d)
ASM_FUNC(sage2_h2d):
/************************************************************************/
vmovd %eax, %xmm0                     // xmm0: 0 0 0 0h
vcvtph2ps %xmm0, %xmm0                // xmm0: 0 0 0 f
vcvtss2sd %xmm0, %xmm0, %xmm0         // xmm0: 0 d

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_s2h)
ASM_FUNC(sage2_s2h):
/************************************************************************/
                                      // xmm0: 0 0 0 f
vcvtps2ph $0, %xmm0, %xmm0            // xmm0: 0 0 0 0h
vmovd %xmm0, %eax

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_h2s)
ASM_FUNC(sage2_h2s):
/************************************************************************/
vmovd %eax, %xmm0                     // xmm0: 0 0 0 0h
vcvtph2ps %xmm0, %xmm0                // xmm0: 0 0 0 f

vzeroupper
retq


/************************************************************************/
.text
.globl ASM_FUNC(sage2_pd2ph)
ASM_FUNC(sage2_pd2ph):
/************************************************************************/
#define N_REG           %rdi
#define IN_PTR          %rsi
#define OUT_PTR         %rdx
#define I_REG           %r8
#define M_REG           %r9

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmovupd   (IN_PTR,I_REG,8), %ymm0     // ymm0: d d d d
vmovupd 32(IN_PTR,I_REG,8), %ymm1
vmovupd 64(IN_PTR,I_REG,8), %ymm2
vmovupd 96(IN_PTR,I_REG,8), %ymm3
vcvtpd2ps %ymm0, %xmm0                // xmm0: f f f f
vcvtpd2ps %ymm1, %xmm1
vcvtpd2ps %ymm2, %xmm2
vcvtpd2ps %ymm3, %xmm3
vcvtps2ph $0, %xmm0,   (OUT_PTR,I_REG,2) // xmm0: 0 0 hh hh
vcvtps2ph $0, %xmm1,  8(OUT_PTR,I_REG,2)
vcvtps2ph $0, %xmm2, 16(OUT_PTR,I_REG,2)
vcvtps2ph $0, %xmm3, 24(OUT_PTR,I_REG,2)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmovupd   (IN_PTR,I_REG,8), %ymm0
vmovupd 32(IN_PTR,I_REG,8), %ymm1
vcvtpd2ps %ymm0, %xmm0
vcvtpd2ps %ymm1, %xmm1
vcvtps2ph $0, %xmm0,  (OUT_PTR,I_REG,2)
vcvtps2ph $0, %xmm1, 8(OUT_PTR,I_REG,2)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

vxorps %xmm15, %xmm15, %xmm15
11:
vmovsd (IN_PTR,I_REG,8), %xmm0
vcvtsd2ss %xmm0, %xmm15, %xmm0
vcvtps2ph $0, %xmm0, %xmm0
vmovd %xmm0, %eax
movw %ax, (OUT_PTR,I_REG,2)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq

#undef N_REG
#undef IN_PTR
#undef OUT_PTR
#undef I_REG
#undef M_REG


/************************************************************************/
.text
.globl ASM_FUNC(sage2_ph2pd)
ASM_FUNC(sage2_ph2pd):
/************************************************************************/
#define N_REG           %rdi
#define IN_PTR          %rsi
#define OUT_PTR         %rdx
#define I_REG           %r8
#define M_REG           %r9

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vcvtph2ps   (IN_PTR,I_REG,2), %xmm0   // xmm0: f f f f
vcvtph2ps  8(IN_PTR,I_REG,2), %xmm1
vcvtph2ps 16(IN_PTR,I_REG,2), %xmm2
vcvtph2ps 24(IN_PTR,I_REG,2), %xmm3
vcvtps2pd %xmm0, %ymm0                // ymm0: d d d d
vcvtps2pd %xmm1, %ymm1
vcvtps2pd %xmm2, %ymm2
vcvtps2pd %xmm3, %ymm3
vmovupd %ymm0,   (OUT_PTR,I_REG,8)
vmovupd %ymm1, 32(OUT_PTR,I_REG,8)
vmovupd %ymm2, 64(OUT_PTR,I_REG,8)
vmovupd %ymm3, 96(OUT_PTR,I_REG,8)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vcvtph2ps  (IN_PTR,I_REG,2), %xmm0
vcvtph2ps 8(IN_PTR,I_REG,2), %xmm1
vcvtps2pd %xmm0, %ymm0
vcvtps2pd %xmm1, %ymm1
vmovupd %ymm0,   (OUT_PTR,I_REG,8)
vmovupd %ymm1, 32(OUT_PTR,I_REG,8)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

xorq %rax, %rax
11:
movw (IN_PTR,I_REG,2), %ax
vmovd %eax, %xmm0
vcvtph2ps %xmm0, %xmm0
vcvtss2sd %xmm0, %xmm0, %xmm0
vmovsd %xmm0, (OUT_PTR,I_REG,8)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq

#undef N_REG
#undef IN_PTR
#undef OUT_PTR
#undef I_REG
#undef M_REG


/************************************************************************/
.text
.globl ASM_FUNC(sage2_ps2ph)
ASM_FUNC(sage2_ps2ph):
/************************************************************************/
#define N_REG           %rdi
#define IN_PTR          %rsi
#define OUT_PTR         %rdx
#define I_REG           %r8
#define M_REG           %r9

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vmovups   (IN_PTR,I_REG,4), %ymm0     // ymm0: f f f f f f f f
vmovups 32(IN_PTR,I_REG,4), %ymm1
vcvtps2ph $0, %ymm0,   (OUT_PTR,I_REG,2) // xmm0: 0 0 hh hh
vcvtps2ph $0, %ymm1, 16(OUT_PTR,I_REG,2)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vmovups (IN_PTR,I_REG,4), %ymm0
vcvtps2ph $0, %ymm0, (OUT_PTR,I_REG,2)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

11:
vmovss (IN_PTR,I_REG,4), %xmm0
vcvtps2ph $0, %xmm0, %xmm0
vmovd %xmm0, %eax
movw %ax, (OUT_PTR,I_REG,2)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq

#undef N_REG
#undef IN_PTR
#undef OUT_PTR
#undef I_REG
#undef M_REG


/************************************************************************/
.text
.globl ASM_FUNC(sage2_ph2ps)
ASM_FUNC(sage2_ph2ps):
/************************************************************************/
#define N_REG           %rdi
#define IN_PTR          %rsi
#define OUT_PTR         %rdx
#define I_REG           %r8
#define M_REG           %r9

xorq I_REG, I_REG
movq N_REG, M_REG
andq $-16, M_REG
je 2f

1:
vcvtph2ps   (IN_PTR,I_REG,2), %ymm0   // ymm0: f f f f f f f f
vcvtph2ps 16(IN_PTR,I_REG,2), %ymm1
vmovups %ymm0,   (OUT_PTR,I_REG,4)
vmovups %ymm1, 32(OUT_PTR,I_REG,4)
subq $-16, I_REG
subq $16, M_REG
jne 1b

2:
movq N_REG, M_REG
andq $8, M_REG
je 10f
vcvtph2ps (IN_PTR,I_REG,2), %ymm0
vmovups %ymm0, (OUT_PTR,I_REG,4)
subq $-8, I_REG

10:
movq N_REG, M_REG
andq $7, M_REG
je 12f

xorq %rax, %rax
11:
movw (IN_PTR,I_REG,2), %ax
vmovd %eax, %xmm0
vcvtph2ps %xmm0, %xmm0
vmovss %xmm0, (OUT_PTR,I_REG,4)
subq $-1, I_REG
subq $1, M_REG
jne 11b

12:
vzeroupper
retq

#undef N_REG
#undef IN_PTR
#undef OUT_PTR
#undef I_REG
#undef M_REG
