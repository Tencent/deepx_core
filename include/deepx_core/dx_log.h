// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <cstdio>
#include <cstdlib>
#include <stdexcept>  // std::invalid_argument, std::runtime_error
#if HAVE_WXG_LOG == 1
#include <iLog.h>
#endif

namespace deepx_core {

struct LogTime {
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
  int microsecond;
};

LogTime GetLogTime() noexcept;

}  // namespace deepx_core

#define _DX_LOG_DEFINE_LOG_TIME() auto __log_time = deepx_core::GetLogTime()
#define _DX_LOG_LOG_TIME_FORMAT "[%.4d%.2d%.2d %.2d:%.2d:%.2d.%.6d]"
#define _DX_LOG_LOG_TIME_ARGS                                         \
  __log_time.year, __log_time.month, __log_time.day, __log_time.hour, \
      __log_time.minute, __log_time.second, __log_time.microsecond

#define _DX_LOG_CONCAT(x, y) x y

#define _DX_LOG_IS_VA_ARGS_IMPL2(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
                                 count, ...)                              \
  count
#define _DX_LOG_IS_VA_ARGS_IMPL1(args) _DX_LOG_IS_VA_ARGS_IMPL2 args
#define _DX_LOG_IS_VA_ARGS(...) \
  _DX_LOG_IS_VA_ARGS_IMPL1((__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0))

// DXTHROW_INVALID_ARGUMENT
#define _DXTHROW_INVALID_ARGUMENT0(format)                                \
  do {                                                                    \
    char __buf[512];                                                      \
    _DX_LOG_DEFINE_LOG_TIME();                                            \
    snprintf(__buf, sizeof(__buf), _DX_LOG_LOG_TIME_FORMAT "%s: " format, \
             _DX_LOG_LOG_TIME_ARGS, __func__);                            \
    throw std::invalid_argument(&__buf[0]);                               \
  } while (0)
#define _DXTHROW_INVALID_ARGUMENT1(format, ...)                           \
  do {                                                                    \
    char __buf[512];                                                      \
    _DX_LOG_DEFINE_LOG_TIME();                                            \
    snprintf(__buf, sizeof(__buf), _DX_LOG_LOG_TIME_FORMAT "%s: " format, \
             _DX_LOG_LOG_TIME_ARGS, __func__, __VA_ARGS__);               \
    throw std::invalid_argument(&__buf[0]);                               \
  } while (0)
#define _DXTHROW_INVALID_ARGUMENT_IMPL2(is_var_args) \
  _DXTHROW_INVALID_ARGUMENT##is_var_args
#define _DXTHROW_INVALID_ARGUMENT_IMPL1(is_var_args) \
  _DXTHROW_INVALID_ARGUMENT_IMPL2(is_var_args)
#define _DXTHROW_INVALID_ARGUMENT(is_var_args) \
  _DXTHROW_INVALID_ARGUMENT_IMPL1(is_var_args)
#define DXTHROW_INVALID_ARGUMENT(...)                                        \
  _DX_LOG_CONCAT(_DXTHROW_INVALID_ARGUMENT(_DX_LOG_IS_VA_ARGS(__VA_ARGS__)), \
                 (__VA_ARGS__))

// DXTHROW_RUNTIME_ERROR
#define _DXTHROW_RUNTIME_ERROR0(format)                                   \
  do {                                                                    \
    char __buf[512];                                                      \
    _DX_LOG_DEFINE_LOG_TIME();                                            \
    snprintf(__buf, sizeof(__buf), _DX_LOG_LOG_TIME_FORMAT "%s: " format, \
             _DX_LOG_LOG_TIME_ARGS, __func__);                            \
    throw std::runtime_error(&__buf[0]);                                  \
  } while (0)
#define _DXTHROW_RUNTIME_ERROR1(format, ...)                              \
  do {                                                                    \
    char __buf[512];                                                      \
    _DX_LOG_DEFINE_LOG_TIME();                                            \
    snprintf(__buf, sizeof(__buf), _DX_LOG_LOG_TIME_FORMAT "%s: " format, \
             _DX_LOG_LOG_TIME_ARGS, __func__, __VA_ARGS__);               \
    throw std::runtime_error(&__buf[0]);                                  \
  } while (0)
#define _DXTHROW_RUNTIME_ERROR_IMPL2(is_var_args) \
  _DXTHROW_RUNTIME_ERROR##is_var_args
#define _DXTHROW_RUNTIME_ERROR_IMPL1(is_var_args) \
  _DXTHROW_RUNTIME_ERROR_IMPL2(is_var_args)
#define _DXTHROW_RUNTIME_ERROR(is_var_args) \
  _DXTHROW_RUNTIME_ERROR_IMPL1(is_var_args)
#define DXTHROW_RUNTIME_ERROR(...)                                        \
  _DX_LOG_CONCAT(_DXTHROW_RUNTIME_ERROR(_DX_LOG_IS_VA_ARGS(__VA_ARGS__)), \
                 (__VA_ARGS__))

// DXINFO
#if HAVE_WXG_LOG == 1
#define _DXINFO0(format)                                                      \
  do {                                                                        \
    Comm::LogErr("[INFO][%s:%d][%s]: " format, __FILE__, __LINE__, __func__); \
  } while (0)
#define _DXINFO1(format, ...)                                                \
  do {                                                                       \
    Comm::LogErr("[INFO][%s:%d][%s]: " format, __FILE__, __LINE__, __func__, \
                 __VA_ARGS__);                                               \
  } while (0)
#else
#define _DXINFO0(format)                                                       \
  do {                                                                         \
    _DX_LOG_DEFINE_LOG_TIME();                                                 \
    fprintf(stderr, _DX_LOG_LOG_TIME_FORMAT "[INFO][%s:%d][%s]: " format "\n", \
            _DX_LOG_LOG_TIME_ARGS, __FILE__, __LINE__, __func__);              \
  } while (0)
#define _DXINFO1(format, ...)                                                  \
  do {                                                                         \
    _DX_LOG_DEFINE_LOG_TIME();                                                 \
    fprintf(stderr, _DX_LOG_LOG_TIME_FORMAT "[INFO][%s:%d][%s]: " format "\n", \
            _DX_LOG_LOG_TIME_ARGS, __FILE__, __LINE__, __func__, __VA_ARGS__); \
  } while (0)
#endif
#define _DXINFO_IMPL2(is_var_args) _DXINFO##is_var_args
#define _DXINFO_IMPL1(is_var_args) _DXINFO_IMPL2(is_var_args)
#define _DXINFO(is_var_args) _DXINFO_IMPL1(is_var_args)
#define DXINFO(...) \
  _DX_LOG_CONCAT(_DXINFO(_DX_LOG_IS_VA_ARGS(__VA_ARGS__)), (__VA_ARGS__))

// DXERROR
#if HAVE_WXG_LOG == 1
#define _DXERROR0(format)                                                      \
  do {                                                                         \
    Comm::LogErr("[ERROR][%s:%d][%s]: " format, __FILE__, __LINE__, __func__); \
  } while (0)
#define _DXERROR1(format, ...)                                                \
  do {                                                                        \
    Comm::LogErr("[ERROR][%s:%d][%s]: " format, __FILE__, __LINE__, __func__, \
                 __VA_ARGS__);                                                \
  } while (0)
#else
#define _DXERROR0(format)                                               \
  do {                                                                  \
    _DX_LOG_DEFINE_LOG_TIME();                                          \
    fprintf(stderr,                                                     \
            _DX_LOG_LOG_TIME_FORMAT "[ERROR][%s:%d][%s]: " format "\n", \
            _DX_LOG_LOG_TIME_ARGS, __FILE__, __LINE__, __func__);       \
  } while (0)
#define _DXERROR1(format, ...)                                                 \
  do {                                                                         \
    _DX_LOG_DEFINE_LOG_TIME();                                                 \
    fprintf(stderr,                                                            \
            _DX_LOG_LOG_TIME_FORMAT "[ERROR][%s:%d][%s]: " format "\n",        \
            _DX_LOG_LOG_TIME_ARGS, __FILE__, __LINE__, __func__, __VA_ARGS__); \
  } while (0)
#endif
#define _DXERROR_IMPL2(is_var_args) _DXERROR##is_var_args
#define _DXERROR_IMPL1(is_var_args) _DXERROR_IMPL2(is_var_args)
#define _DXERROR(is_var_args) _DXERROR_IMPL1(is_var_args)
#define DXERROR(...) \
  _DX_LOG_CONCAT(_DXERROR(_DX_LOG_IS_VA_ARGS(__VA_ARGS__)), (__VA_ARGS__))

// DXASSERT
#if defined NDEBUG
#define DXASSERT(cond) ((void)0)
#else
#if HAVE_WXG_LOG == 1
#define DXASSERT(cond)                                                    \
  do {                                                                    \
    if (!(cond)) {                                                        \
      Comm::LogErr("[ASSERT][%s:%d][%s]: Assert failed: '%s'.", __FILE__, \
                   __LINE__, __func__, #cond);                            \
      abort();                                                            \
    }                                                                     \
  } while (0)
#else
#define DXASSERT(cond)                                                     \
  do {                                                                     \
    if (!(cond)) {                                                         \
      _DX_LOG_DEFINE_LOG_TIME();                                           \
      fprintf(stderr,                                                      \
              _DX_LOG_LOG_TIME_FORMAT                                      \
              "[ASSERT][%s:%d][%s]: Assert failed: '%s'.\n",               \
              _DX_LOG_LOG_TIME_ARGS, __FILE__, __LINE__, __func__, #cond); \
      abort();                                                             \
    }                                                                      \
  } while (0)
#endif
#endif

// DXCHECK
#if HAVE_WXG_LOG == 1
#define DXCHECK(cond)                                                   \
  do {                                                                  \
    if (!(cond)) {                                                      \
      Comm::LogErr("[CHECK][%s:%d][%s]: Check failed: '%s'.", __FILE__, \
                   __LINE__, __func__, #cond);                          \
      abort();                                                          \
    }                                                                   \
  } while (0)
#else
#define DXCHECK(cond)                                                      \
  do {                                                                     \
    if (!(cond)) {                                                         \
      _DX_LOG_DEFINE_LOG_TIME();                                           \
      fprintf(stderr,                                                      \
              _DX_LOG_LOG_TIME_FORMAT                                      \
              "[CHECK][%s:%d][%s]: Check failed: '%s'.\n",                 \
              _DX_LOG_LOG_TIME_ARGS, __FILE__, __LINE__, __func__, #cond); \
      abort();                                                             \
    }                                                                      \
  } while (0)
#endif

// DXCHECK_THROW
#define DXCHECK_THROW(cond)                                                 \
  do {                                                                      \
    if (!(cond)) {                                                          \
      char __buf[512];                                                      \
      _DX_LOG_DEFINE_LOG_TIME();                                            \
      snprintf(__buf, sizeof(__buf),                                        \
               _DX_LOG_LOG_TIME_FORMAT                                      \
               "[CHECK][%s:%d][%s]: Check failed: '%s'.",                   \
               _DX_LOG_LOG_TIME_ARGS, __FILE__, __LINE__, __func__, #cond); \
      throw std::runtime_error(&__buf[0]);                                  \
    }                                                                       \
  } while (0)
