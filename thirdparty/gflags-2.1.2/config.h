#if defined _WIN32

#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_STRTOLL 1
#define HAVE_STRTOQ 1

#define OS_WINDOWS 1
#define PATH_SEPARATOR '\\'
#include "windows_port.h"

#else  // _WIN32

#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_UNISTD_H 1
#define HAVE_FNMATCH_H 1
#define HAVE_STRTOLL 1
#define HAVE_PTHREAD 1
#define HAVE_RWLOCK 1

#define PATH_SEPARATOR '/'

#endif  // _WIN32

#if !defined __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif

#ifndef GFLAGS_DLL_DECL
#define GFLAGS_DLL_DECL
#endif

#ifndef GFLAGS_DLL_DEFINE_FLAG
#define GFLAGS_DLL_DEFINE_FLAG
#endif
