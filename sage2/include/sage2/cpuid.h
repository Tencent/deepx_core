// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_CPUID_H_
#define SAGE2_CPUID_H_

#include <sage2/macro.h>
#include <stdint.h>  // NOLINT

/************************************************************************/
/* Run CPUID instruction. */
/************************************************************************/
// NOLINTNEXTLINE
typedef struct _sage2_cpu_info {
  uint32_t eax;
  uint32_t ebx;
  uint32_t ecx;
  uint32_t edx;
} sage2_cpu_info;
SAGE2_C_API void sage2_cpuid(uint32_t eax, uint32_t ecx,
                             sage2_cpu_info* cpu_info);

/************************************************************************/
/* CPU information. */
/************************************************************************/
enum SAGE2_CPU_VENDOR_TYPE {
  SAGE2_CPU_VENDOR_TYPE_NONE = 0,
  SAGE2_CPU_VENDOR_TYPE_INTEL,
  SAGE2_CPU_VENDOR_TYPE_AMD,
};
// They are read-only for users.
SAGE2_C_API const char* sage2_cpu_vendor;
SAGE2_C_API int sage2_cpu_vendor_type;
SAGE2_C_API const char* sage2_cpu_brand;
SAGE2_C_API int sage2_cpu_l1_dcache_bytes;
SAGE2_C_API int sage2_cpu_l1_icache_bytes;
SAGE2_C_API int sage2_cpu_l2_cache_bytes;
SAGE2_C_API int sage2_cpu_l3_cache_bytes;
SAGE2_C_API int sage2_cpu_mmx;
SAGE2_C_API int sage2_cpu_sse;
SAGE2_C_API int sage2_cpu_sse2;
SAGE2_C_API int sage2_cpu_sse3;
SAGE2_C_API int sage2_cpu_ssse3;
SAGE2_C_API int sage2_cpu_sse41;
SAGE2_C_API int sage2_cpu_sse42;
SAGE2_C_API int sage2_cpu_avx;
SAGE2_C_API int sage2_cpu_fma;
SAGE2_C_API int sage2_cpu_avx2;
SAGE2_C_API int sage2_cpu_avx512f;
SAGE2_C_API int sage2_cpu_avx512dq;
SAGE2_C_API int sage2_cpu_avx512_ifma;
SAGE2_C_API int sage2_cpu_avx512pf;
SAGE2_C_API int sage2_cpu_avx512er;
SAGE2_C_API int sage2_cpu_avx512cd;
SAGE2_C_API int sage2_cpu_avx512bw;
SAGE2_C_API int sage2_cpu_avx512vl;
SAGE2_C_API int sage2_cpu_avx512_vbmi;
SAGE2_C_API int sage2_cpu_avx512_vbmi2;
SAGE2_C_API int sage2_cpu_avx512_vnni;
SAGE2_C_API int sage2_cpu_avx512_bitalg;
SAGE2_C_API int sage2_cpu_avx512_vpopcntdq;
SAGE2_C_API int sage2_cpu_avx512_4vnniw;
SAGE2_C_API int sage2_cpu_avx512_4fmaps;

#endif  // SAGE2_CPUID_H_
