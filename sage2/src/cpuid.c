// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/cpuid.h>
#include <string.h>
#include "internal_macro.h"

#define BIT(u32, bit) ((int)((u32 & (1u << bit)) >> bit))
#define BITS(u32, lobit, hibit) \
  ((int)((u32 & (((1u << (hibit - lobit + 1)) - 1) << lobit)) >> lobit))

static char sage2_cpu_vendor_buf[64] = {0};
static char sage2_cpu_brand_buf[64] = {0};

const char* sage2_cpu_vendor = NULL;
int sage2_cpu_vendor_type = SAGE2_CPU_VENDOR_TYPE_NONE;
const char* sage2_cpu_brand = NULL;
int sage2_cpu_l1_dcache_bytes = 0;
int sage2_cpu_l1_icache_bytes = 0;
int sage2_cpu_l2_cache_bytes = 0;
int sage2_cpu_l3_cache_bytes = 0;
int sage2_cpu_mmx = 0;
int sage2_cpu_sse = 0;
int sage2_cpu_sse2 = 0;
int sage2_cpu_sse3 = 0;
int sage2_cpu_ssse3 = 0;
int sage2_cpu_sse41 = 0;
int sage2_cpu_sse42 = 0;
int sage2_cpu_avx = 0;
int sage2_cpu_fma = 0;
int sage2_cpu_avx2 = 0;
int sage2_cpu_avx512f = 0;
int sage2_cpu_avx512dq = 0;
int sage2_cpu_avx512_ifma = 0;
int sage2_cpu_avx512pf = 0;
int sage2_cpu_avx512er = 0;
int sage2_cpu_avx512cd = 0;
int sage2_cpu_avx512bw = 0;
int sage2_cpu_avx512vl = 0;
int sage2_cpu_avx512_vbmi = 0;
int sage2_cpu_avx512_vbmi2 = 0;
int sage2_cpu_avx512_vnni = 0;
int sage2_cpu_avx512_bitalg = 0;
int sage2_cpu_avx512_vpopcntdq = 0;
int sage2_cpu_avx512_4vnniw = 0;
int sage2_cpu_avx512_4fmaps = 0;

static void init_vendor() {
  sage2_cpu_info info;
  sage2_cpuid(0, 0, &info);
  memcpy(sage2_cpu_vendor_buf, &info.ebx, 4);
  memcpy(sage2_cpu_vendor_buf + 4, &info.edx, 4);
  memcpy(sage2_cpu_vendor_buf + 8, &info.ecx, 4);
  sage2_cpu_vendor_buf[12] = 0;
  if (strncmp(sage2_cpu_vendor_buf, "GenuineIntel", 12) == 0) {
    sage2_cpu_vendor = sage2_cpu_vendor_buf;
    sage2_cpu_vendor_type = SAGE2_CPU_VENDOR_TYPE_INTEL;
  } else if (strncmp(sage2_cpu_vendor_buf, "AuthenticAMD", 12) == 0) {
    sage2_cpu_vendor = sage2_cpu_vendor_buf;
    sage2_cpu_vendor_type = SAGE2_CPU_VENDOR_TYPE_AMD;
  }
}

static void init_brand() {
  sage2_cpu_info info802, info803, info804;
  sage2_cpuid(0x80000002, 0, &info802);
  sage2_cpuid(0x80000003, 0, &info803);
  sage2_cpuid(0x80000004, 0, &info804);
  memcpy(sage2_cpu_brand_buf, &info802.eax, 16);
  memcpy(sage2_cpu_brand_buf + 16, &info803.eax, 16);
  memcpy(sage2_cpu_brand_buf + 32, &info804.eax, 16);
  sage2_cpu_brand = sage2_cpu_brand_buf;
}

static void init_cache() {
  sage2_cpu_info info;
  int type, level, line, partition, way, set, bytes;
  int i;
  for (i = 0;; ++i) {
    sage2_cpuid(4, i, &info);
    type = BITS(info.eax, 0, 4);
    if (type == 0) {
      break;
    }

    level = BITS(info.eax, 5, 7);
    line = BITS(info.ebx, 0, 11) + 1;
    partition = BITS(info.ebx, 12, 21) + 1;
    way = BITS(info.ebx, 22, 31) + 1;
    set = info.ecx + 1;
    bytes = line * partition * way * set;
    if (type == 1 && level == 1) {
      sage2_cpu_l1_dcache_bytes = bytes;
    } else if (type == 2 && level == 1) {
      sage2_cpu_l1_icache_bytes = bytes;
    } else if (type == 3 && level == 2) {
      sage2_cpu_l2_cache_bytes = bytes;
    } else if (type == 3 && level == 3) {
      sage2_cpu_l3_cache_bytes = bytes;
    }
  }
}

static void init_simd() {
  sage2_cpu_info info1, info7;
  sage2_cpuid(1, 0, &info1);
  sage2_cpuid(7, 0, &info7);
  sage2_cpu_mmx = BIT(info1.edx, 23);
  sage2_cpu_sse = BIT(info1.edx, 25);
  sage2_cpu_sse2 = BIT(info1.edx, 26);
  sage2_cpu_sse3 = BIT(info1.ecx, 0);
  sage2_cpu_ssse3 = BIT(info1.ecx, 9);
  sage2_cpu_sse41 = BIT(info1.ecx, 19);
  sage2_cpu_sse42 = BIT(info1.ecx, 20);
  sage2_cpu_avx = BIT(info1.ecx, 28);
  sage2_cpu_fma = BIT(info1.ecx, 12);
  sage2_cpu_avx2 = BIT(info7.ebx, 5);
  sage2_cpu_avx512f = BIT(info7.ebx, 16);
  sage2_cpu_avx512dq = BIT(info7.ebx, 17);
  sage2_cpu_avx512_ifma = BIT(info7.ebx, 21);
  sage2_cpu_avx512pf = BIT(info7.ebx, 26);
  sage2_cpu_avx512er = BIT(info7.ebx, 27);
  sage2_cpu_avx512cd = BIT(info7.ebx, 28);
  sage2_cpu_avx512bw = BIT(info7.ebx, 30);
  sage2_cpu_avx512vl = BIT(info7.ebx, 31);
  sage2_cpu_avx512_vbmi = BIT(info7.ecx, 1);
  sage2_cpu_avx512_vbmi2 = BIT(info7.ecx, 6);
  sage2_cpu_avx512_vnni = BIT(info7.ecx, 11);
  sage2_cpu_avx512_bitalg = BIT(info7.ecx, 12);
  sage2_cpu_avx512_vpopcntdq = BIT(info7.ecx, 14);
  sage2_cpu_avx512_4vnniw = BIT(info7.edx, 2);
  sage2_cpu_avx512_4fmaps = BIT(info7.edx, 3);
}

ATTR_CTOR(101) static void init() {
  init_vendor();
  init_brand();
  init_cache();
  init_simd();
}
