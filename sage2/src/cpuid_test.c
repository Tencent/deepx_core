// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/cpuid.h>
#include <stdio.h>

const char* get_readable_bytes(char* buf, size_t size, int bytes) {
  if (bytes < 1024) {
    snprintf(buf, size, "%u B", bytes);
  } else if (bytes < 1024 * 1024) {
    snprintf(buf, size, "%u KB", bytes / 1024);
  } else if (bytes < 1024 * 1024 * 1024) {
    snprintf(buf, size, "%u MB", bytes / 1024 / 1024);
  } else {
    snprintf(buf, size, "%u GB", bytes / 1024 / 1024 / 1024);
  }
  return buf;
}

void dump_cpu(FILE* f) {
  char buf[64];
#define KEY "%-20s: "
  fprintf(f, KEY "%s\n", "Vendor", sage2_cpu_vendor);
  fprintf(f, KEY "%s\n", "Brand", sage2_cpu_brand);
  fprintf(f, KEY "%s\n", "L1 data cache",
          get_readable_bytes(buf, sizeof(buf), sage2_cpu_l1_dcache_bytes));
  fprintf(f, KEY "%s\n", "L1 instruction cache",
          get_readable_bytes(buf, sizeof(buf), sage2_cpu_l1_icache_bytes));
  fprintf(f, KEY "%s\n", "L2 cache",
          get_readable_bytes(buf, sizeof(buf), sage2_cpu_l2_cache_bytes));
  fprintf(f, KEY "%s\n", "L3 cache",
          get_readable_bytes(buf, sizeof(buf), sage2_cpu_l3_cache_bytes));
  fprintf(f, KEY "%d\n", "MMX", sage2_cpu_mmx);
  fprintf(f, KEY "%d\n", "SSE", sage2_cpu_sse);
  fprintf(f, KEY "%d\n", "SSE2", sage2_cpu_sse2);
  fprintf(f, KEY "%d\n", "SSE3", sage2_cpu_sse3);
  fprintf(f, KEY "%d\n", "SSSE3", sage2_cpu_ssse3);
  fprintf(f, KEY "%d\n", "SSE4.1", sage2_cpu_sse41);
  fprintf(f, KEY "%d\n", "SSE4.2", sage2_cpu_sse42);
  fprintf(f, KEY "%d\n", "AVX", sage2_cpu_avx);
  fprintf(f, KEY "%d\n", "FMA", sage2_cpu_fma);
  fprintf(f, KEY "%d\n", "AVX2", sage2_cpu_avx2);
  fprintf(f, KEY "%d\n", "AVX512F", sage2_cpu_avx512f);
  fprintf(f, KEY "%d\n", "AVX512DQ", sage2_cpu_avx512dq);
  fprintf(f, KEY "%d\n", "AVX512_IFMA", sage2_cpu_avx512_ifma);
  fprintf(f, KEY "%d\n", "AVX512PF", sage2_cpu_avx512pf);
  fprintf(f, KEY "%d\n", "AVX512ER", sage2_cpu_avx512er);
  fprintf(f, KEY "%d\n", "AVX512CD", sage2_cpu_avx512cd);
  fprintf(f, KEY "%d\n", "AVX512BW", sage2_cpu_avx512bw);
  fprintf(f, KEY "%d\n", "AVX512VL", sage2_cpu_avx512vl);
  fprintf(f, KEY "%d\n", "AVX512_VBMI", sage2_cpu_avx512_vbmi);
  fprintf(f, KEY "%d\n", "AVX512_VBMI2", sage2_cpu_avx512_vbmi2);
  fprintf(f, KEY "%d\n", "AVX512_VNNI", sage2_cpu_avx512_vnni);
  fprintf(f, KEY "%d\n", "AVX512_BITALG", sage2_cpu_avx512_bitalg);
  fprintf(f, KEY "%d\n", "AVX512_VPOPCNTDQ", sage2_cpu_avx512_vpopcntdq);
  fprintf(f, KEY "%d\n", "AVX512_4VNNIW", sage2_cpu_avx512_4vnniw);
  fprintf(f, KEY "%d\n", "AVX512_4FMAPS", sage2_cpu_avx512_4fmaps);
#undef KEY
}

int main() {
  dump_cpu(stdout);
  return 0;
}
