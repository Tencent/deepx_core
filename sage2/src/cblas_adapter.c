// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <dlfcn.h>
#include <sage2/cblas_adapter.h>
#include <stdio.h>
#include <string.h>
#include "internal_macro.h"

typedef struct _cblas_func_meta {
  const char* name;
  void** pptr;
} cblas_func_meta;

typedef struct _cblas_so_meta {
  const char* so;
  const char* vendor;
  sage2_cblas* cblas;
} cblas_so_meta;

static sage2_cblas cblas;
static sage2_cblas mkl;
static sage2_cblas openblas;

const sage2_cblas* sage2_cblas_cblas() { return &cblas; }
const sage2_cblas* sage2_cblas_mkl() { return &mkl; }
const sage2_cblas* sage2_cblas_openblas() { return &openblas; }

static const cblas_so_meta meta[] = {
    {"libcblas_sage2.so", "cblas", &cblas},
    {"libcblas.so", "cblas", &cblas},
#if defined __APPLE__ && defined __MACH__
    {"libcblas.dylib", "cblas", &cblas},
#endif
    {"libmkl_sage2.so", "mkl", &mkl},
    {"libopenblas_sage2.so", "openblas", &openblas},
    {"libopenblas.so", "openblas", &openblas},
#if defined __APPLE__ && defined __MACH__
    {"libopenblas.dylib", "openblas", &openblas},
#endif
    {NULL, NULL, NULL},
};

static int load_cblas(const char* so, const char* vendor, sage2_cblas* cblas) {
  const cblas_func_meta meta[] = {
      {"cblas_saxpy", (void**)&cblas->cblas_saxpy},
      {"cblas_sdot", (void**)&cblas->cblas_sdot},
      {"cblas_snrm2", (void**)&cblas->cblas_snrm2},
      {"cblas_sasum", (void**)&cblas->cblas_sasum},
      {"cblas_sscal", (void**)&cblas->cblas_sscal},
      {"cblas_sgemm", (void**)&cblas->cblas_sgemm},
      {"cblas_sgemm_batch", (void**)&cblas->cblas_sgemm_batch},
      {"cblas_saxpby", (void**)&cblas->cblas_saxpby},
      {"catlas_saxpby", (void**)&cblas->cblas_saxpby},
      {"mkl_cblas_jit_create_sgemm",
       (void**)&cblas->mkl_cblas_jit_create_sgemm},
      {"mkl_jit_get_sgemm_ptr", (void**)&cblas->mkl_jit_get_sgemm_ptr},
      {"mkl_jit_destroy", (void**)&cblas->mkl_jit_destroy},
      {"vsAdd", (void**)&cblas->vsAdd},
      {"vsSub", (void**)&cblas->vsSub},
      {"vsMul", (void**)&cblas->vsMul},
      {"vsDiv", (void**)&cblas->vsDiv},
      {"vsExp", (void**)&cblas->vsExp},
      {"vsLn", (void**)&cblas->vsLn},
      {"vsSqrt", (void**)&cblas->vsSqrt},
      {"vsTanh", (void**)&cblas->vsTanh},
      {NULL, NULL},
  };
  void* handle;
  int i;
  int n = 0;

  memset(cblas, 0, sizeof(sage2_cblas));
  handle = dlopen(so, RTLD_NOW);
  if (handle == NULL) {
    return -1;
  }

  cblas->handle = handle;
  cblas->so = so;
  cblas->vendor = vendor;
  for (i = 0;; ++i) {
    if (meta[i].name == NULL) {
      break;
    }
    if (*meta[i].pptr == NULL) {
      *meta[i].pptr = dlsym(handle, meta[i].name);
      if (*meta[i].pptr) {
        ++n;
      }
    }
  }
  fprintf(stderr, "Loaded %2d functions from %s(%s).\n", n, so, vendor);
  return 0;
}

static void unload_cblas(sage2_cblas* cblas) {
  if (cblas->handle) {
    (void)dlclose(cblas->handle);
  }
  memset(cblas, 0, sizeof(sage2_cblas));
}

ATTR_CTOR(102) static void init() {
  int i;
  for (i = 0;; ++i) {
    if (meta[i].cblas == NULL) {
      break;
    }
    if (meta[i].cblas->handle) {
      continue;
    }
    (void)load_cblas(meta[i].so, meta[i].vendor, meta[i].cblas);
  }
}

ATTR_DTOR(102) static void uninit() {
  unload_cblas(&cblas);
  unload_cblas(&mkl);
  unload_cblas(&openblas);
}
