// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <unistd.h>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/************************************************************************/
/* nan & inf */
/************************************************************************/
namespace detail {

// IEEE 754 quiet nan
constexpr uint32_t U32_QUIET_NAN = UINT32_C(0x7fc00000);
// non-standard quiet nan
constexpr uint32_t U32_QUIET_NAN1 = UINT32_C(0x7fc00001);
// non-standard quiet nan
constexpr uint32_t U32_QUIET_NAN2 = UINT32_C(0x7fc00002);
// IEEE 754 signaling nan
constexpr uint32_t U32_SIGNALING_NAN = UINT32_C(0x7f800001);
// IEEE 754 inf
constexpr uint32_t U32_INF = UINT32_C(0x7f800000);
// IEEE 754 -inf
constexpr uint32_t U32_NINF = UINT32_C(0xff800000);

inline float ReinterpretAsFloat(uint32_t u) noexcept {
  union {
    uint32_t u;
    float f;
  } a = {u};
  return a.f;
}

inline uint32_t ReinterpretAsU32(float f) noexcept {
  union {
    float f;
    uint32_t u;
  } a = {f};
  return a.u;
}

}  // namespace detail

#define FLOAT_QUIET_NAN (detail::ReinterpretAsFloat(detail::U32_QUIET_NAN))
#define FLOAT_SIGNALING_NAN \
  (detail::ReinterpretAsFloat(detail::U32_SIGNALING_NAN))
#define FLOAT_INF (detail::ReinterpretAsFloat(detail::U32_INF))
#define FLOAT_NINF (detail::ReinterpretAsFloat(detail::U32_NINF))

/************************************************************************/
/* Check functions */
/************************************************************************/
namespace detail {

template <typename T>
double RelativeError(T a, T b) noexcept {
  if (a == b) {
    return 0;
  }

  double abs_a = std::abs(a);
  double abs_b = std::abs(b);
  double error = std::abs(a - b) / (abs_a + abs_b);

#if !defined STRICT_ERROR
  // In non-strict mode,
  // scale down the error, if both a and b are too small.
  constexpr double EPS = 1e-3;
  if (abs_a < abs_b) {
    abs_a = abs_b;
  }
  if (abs_a < EPS) {
    error *= abs_a / EPS;
  }
#endif
  return error;
}

template <typename T>
double RelativeErrorThreshold() noexcept;

template <>
double RelativeErrorThreshold<float>() noexcept {
#if !defined STRICT_ERROR
  return 5e-3;
#else
  return 1e-3;
#endif
}

template <>
double RelativeErrorThreshold<double>() noexcept {
#if !defined STRICT_ERROR
  return 1e-3;
#else
  return 1e-6;
#endif
}

// scalar, is_floating_point, is_signed
template <typename T>
void CheckEqual(const char* file, int line, const char* func, T a, T b,
                std::false_type /*is_integral*/,
                std::true_type /*is_signed*/) noexcept {
  if (RelativeError(a, b) > RelativeErrorThreshold<T>()) {
    fprintf(stderr, "[%s:%d][%s] a(%.12f) != b(%.12f), error=%.6f.\n", file,
            line, func, (double)a, (double)b, RelativeError(a, b));
  }
}

// scalar, is_integral, !is_signed
template <typename T>
void CheckEqual(const char* file, int line, const char* func, T a, T b,
                std::true_type /*is_integral*/,
                std::false_type /*is_signed*/) noexcept {
  if (a != b) {
    fprintf(stderr, "[%s:%d][%s] a(%" PRIu64 ") != b(%" PRIu64 ").\n", file,
            line, func, (uint64_t)a, (uint64_t)b);
  }
}

// scalar, is_integral, is_signed
template <typename T>
void CheckEqual(const char* file, int line, const char* func, T a, T b,
                std::true_type /*is_integral*/,
                std::true_type /*is_signed*/) noexcept {
  if (a != b) {
    fprintf(stderr, "[%s:%d][%s] a(%" PRId64 ") != b(%" PRId64 ").\n", file,
            line, func, (int64_t)a, (int64_t)b);
  }
}

// vector, is_floating_point, is_signed
template <typename T>
void CheckEqual(const char* file, int line, const char* func,
                const std::vector<T>& a, const std::vector<T>& b,
                std::false_type /*is_integral*/,
                std::true_type /*is_signed*/) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    if (RelativeError(a[i], b[i]) > RelativeErrorThreshold<T>()) {
      fprintf(stderr,
              "[%s:%d][%s] a[%zu](%.12f) != b[%zu](%.12f), error=%.6f.\n", file,
              line, func, i, (double)a[i], i, (double)b[i],
              RelativeError(a[i], b[i]));
    }
  }
}

// vector, is_integral, !is_signed
template <typename T>
void CheckEqual(const char* file, int line, const char* func,
                const std::vector<T>& a, const std::vector<T>& b,
                std::true_type /*is_integral*/,
                std::false_type /*is_signed*/) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      fprintf(stderr,
              "[%s:%d][%s] a[%zu](%" PRIu64 ") != b[%zu](%" PRIu64 ").\n", file,
              line, func, i, (uint64_t)a[i], i, (uint64_t)b[i]);
    }
  }
}

// vector, is_integral, is_signed
template <typename T>
void CheckEqual(const char* file, int line, const char* func,
                const std::vector<T>& a, const std::vector<T>& b,
                std::true_type /*is_integral*/,
                std::true_type /*is_signed*/) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      fprintf(stderr,
              "[%s:%d][%s] a[%zu](%" PRId64 ") != b[%zu](%" PRId64 ").\n", file,
              line, func, i, (int64_t)a[i], i, (int64_t)b[i]);
    }
  }
}

template <typename T>
void CheckEqual(const char* file, int line, const char* func, T a,
                T b) noexcept {
  CheckEqual(file, line, func, a, b, typename std::is_integral<T>::type(),
             typename std::is_signed<T>::type());
}

template <typename T>
void CheckEqual(const char* file, int line, const char* func,
                const std::vector<T>& a, const std::vector<T>& b) noexcept {
  CheckEqual(file, line, func, a, b, typename std::is_integral<T>::type(),
             typename std::is_signed<T>::type());
}

template <typename T>
void CheckNotNanInf(const char* file, int line, const char* func, T a) noexcept;

template <>
inline void CheckNotNanInf<float>(const char* file, int line, const char* func,
                                  float a) noexcept {
  switch (ReinterpretAsU32(a)) {
    case U32_QUIET_NAN:
    case U32_QUIET_NAN1:
    case U32_QUIET_NAN2:
    case U32_SIGNALING_NAN:
      fprintf(stderr, "[%s:%d][%s] a is nan, which it should not be.\n", file,
              line, func);
      break;
    case U32_INF:
      fprintf(stderr, "[%s:%d][%s] a is inf, which it should not be.\n", file,
              line, func);
      break;
    case U32_NINF:
      fprintf(stderr, "[%s:%d][%s] a is -inf, which it should not be.\n", file,
              line, func);
      break;
  }
}

template <typename T>
void CheckNotNanInf(const char* file, int line, const char* func,
                    const std::vector<T>& a) noexcept;

template <>
inline void CheckNotNanInf<float>(const char* file, int line, const char* func,
                                  const std::vector<float>& a) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    switch (ReinterpretAsU32(a[i])) {
      case U32_QUIET_NAN:
      case U32_QUIET_NAN1:
      case U32_QUIET_NAN2:
      case U32_SIGNALING_NAN:
        fprintf(stderr, "[%s:%d][%s] a[%zu] is nan, which it should not be.\n",
                file, line, func, i);
        break;
      case U32_INF:
        fprintf(stderr, "[%s:%d][%s] a[%zu] is inf, which it should not be.\n",
                file, line, func, i);
        break;
      case U32_NINF:
        fprintf(stderr, "[%s:%d][%s] a[%zu] is -inf, which it should not be.\n",
                file, line, func, i);
        break;
    }
  }
}

template <typename T>
void CheckIsNanInf(const char* file, int line, const char* func, T a) noexcept;

template <>
inline void CheckIsNanInf<float>(const char* file, int line, const char* func,
                                 float a) noexcept {
  switch (ReinterpretAsU32(a)) {
    case U32_QUIET_NAN:
    case U32_QUIET_NAN1:
    case U32_QUIET_NAN2:
    case U32_SIGNALING_NAN:
    case U32_INF:
    case U32_NINF:
      break;
    default:
      fprintf(stderr,
              "[%s:%d][%s] a(%.12f) is not nan/inf/-inf, which it should be.\n",
              file, line, func, a);
      break;
  }
}

template <typename T>
void CheckIsNanInf(const char* file, int line, const char* func,
                   const std::vector<T>& a) noexcept;

template <>
inline void CheckIsNanInf<float>(const char* file, int line, const char* func,
                                 const std::vector<float>& a) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    switch (ReinterpretAsU32(a[i])) {
      case U32_QUIET_NAN:
      case U32_QUIET_NAN1:
      case U32_QUIET_NAN2:
      case U32_SIGNALING_NAN:
      case U32_INF:
      case U32_NINF:
        break;
      default:
        fprintf(
            stderr,
            "[%s:%d][%s] a[%zu] is not nan/inf/-inf, which it should not be.\n",
            file, line, func, i);
        break;
    }
  }
}

template <typename T>
void CheckIsNan(const char* file, int line, const char* func, T a) noexcept;

template <>
inline void CheckIsNan<float>(const char* file, int line, const char* func,
                              float a) noexcept {
  switch (ReinterpretAsU32(a)) {
    case U32_QUIET_NAN:
    case U32_QUIET_NAN1:
    case U32_QUIET_NAN2:
    case U32_SIGNALING_NAN:
      break;
    default:
      fprintf(stderr, "[%s:%d][%s] a(%.12f) is not nan, which it should be.\n",
              file, line, func, a);
      break;
  }
}

template <typename T>
void CheckIsNan(const char* file, int line, const char* func,
                const std::vector<T>& a) noexcept;

template <>
inline void CheckIsNan<float>(const char* file, int line, const char* func,
                              const std::vector<float>& a) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    switch (ReinterpretAsU32(a[i])) {
      case U32_QUIET_NAN:
      case U32_QUIET_NAN1:
      case U32_QUIET_NAN2:
      case U32_SIGNALING_NAN:
        break;
      default:
        fprintf(stderr,
                "[%s:%d][%s] a[%zu] is not nan, which it should not be.\n",
                file, line, func, i);
        break;
    }
  }
}

template <typename T>
void CheckIsInf(const char* file, int line, const char* func, T a) noexcept;

template <>
inline void CheckIsInf<float>(const char* file, int line, const char* func,
                              float a) noexcept {
  switch (ReinterpretAsU32(a)) {
    case U32_INF:
      break;
    default:
      fprintf(stderr, "[%s:%d][%s] a(%.12f) is not inf, which it should be.\n",
              file, line, func, a);
      break;
  }
}

template <typename T>
void CheckIsInf(const char* file, int line, const char* func,
                const std::vector<T>& a) noexcept;

template <>
inline void CheckIsInf<float>(const char* file, int line, const char* func,
                              const std::vector<float>& a) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    switch (ReinterpretAsU32(a[i])) {
      case U32_INF:
        break;
      default:
        fprintf(stderr,
                "[%s:%d][%s] a[%zu] is not inf, which it should not be.\n",
                file, line, func, i);
        break;
    }
  }
}

template <typename T>
void CheckIsNInf(const char* file, int line, const char* func, T a) noexcept;

template <>
inline void CheckIsNInf<float>(const char* file, int line, const char* func,
                               float a) noexcept {
  switch (ReinterpretAsU32(a)) {
    case U32_NINF:
      break;
    default:
      fprintf(stderr, "[%s:%d][%s] a(%.12f) is not -inf, which it should be\n",
              file, line, func, a);
      break;
  }
}

template <typename T>
void CheckIsNInf(const char* file, int line, const char* func,
                 const std::vector<T>& a) noexcept;

template <>
inline void CheckIsNInf<float>(const char* file, int line, const char* func,
                               const std::vector<float>& a) noexcept {
  for (size_t i = 0; i < a.size(); ++i) {
    switch (ReinterpretAsU32(a[i])) {
      case U32_NINF:
        break;
      default:
        fprintf(stderr,
                "[%s:%d][%s] a[%zu] is not -inf, which it should not be.\n",
                file, line, func, i);
        break;
    }
  }
}

}  // namespace detail

#define CHECK_EQUAL(a, b) detail::CheckEqual(__FILE__, __LINE__, __func__, a, b)
#define CHECK_NOT_NAN_INF(a) \
  detail::CheckNotNanInf(__FILE__, __LINE__, __func__, a)
#define CHECK_IS_NAN_INF(a) \
  detail::CheckIsNanInf(__FILE__, __LINE__, __func__, a)
#define CHECK_IS_NAN(a) detail::CheckIsNan(__FILE__, __LINE__, __func__, a)
#define CHECK_IS_INF(a) detail::CheckIsInf(__FILE__, __LINE__, __func__, a)
#define CHECK_IS_NINF(a) detail::CheckIsNInf(__FILE__, __LINE__, __func__, a)

/************************************************************************/
/* randn & rand */
/************************************************************************/
namespace detail {

// scalar, is_floating_point
template <class RandomEngine, typename T>
void rand(RandomEngine&& engine, T* a, T lo, T hi,
          std::false_type /*is_integral*/) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating-point type.");
  std::uniform_real_distribution<T> dist(lo, hi);
  *a = dist(engine);
}

// scalar, is_integral
template <class RandomEngine, typename T>
void rand(RandomEngine&& engine, T* a, T lo, T hi,
          std::true_type /*is_integral*/) noexcept {
  std::uniform_int_distribution<T> dist(lo, hi);
  *a = dist(engine);
}

// vector, is_floating_point
template <class RandomEngine, typename T>
void rand(RandomEngine&& engine, std::vector<T>* a, T lo, T hi,
          std::false_type /*is_integral*/) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating-point type.");
  std::uniform_real_distribution<T> dist(lo, hi);
  for (size_t i = 0; i < a->size(); ++i) {
    (*a)[i] = dist(engine);
  }
}

// vector, is_integral
template <class RandomEngine, typename T>
void rand(RandomEngine&& engine, std::vector<T>* a, T lo, T hi,
          std::true_type /*is_integral*/) noexcept {
  std::uniform_int_distribution<T> dist(lo, hi);
  for (size_t i = 0; i < a->size(); ++i) {
    (*a)[i] = dist(engine);
  }
}

}  // namespace detail

template <class RandomEngine, typename T>
void randn(RandomEngine&& engine, T* a) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating-point type.");
  std::normal_distribution<T> dist((T)0, (T)1);
  *a = dist(engine);
}

template <class RandomEngine, typename T1, typename T2>
void randn(RandomEngine&& engine, T1* a, T2 mean, T2 stddev) noexcept {
  static_assert(std::is_floating_point<T1>::value,
                "T1 must be a floating-point type.");
  std::normal_distribution<T1> dist((T1)mean, (T1)stddev);
  *a = dist(engine);
}

template <class RandomEngine, typename T>
void randn(RandomEngine&& engine, std::vector<T>* a) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating-point type.");
  std::normal_distribution<T> dist((T)0, (T)1);
  for (size_t i = 0; i < a->size(); ++i) {
    (*a)[i] = dist(engine);
  }
}

template <class RandomEngine, typename T1, typename T2>
void randn(RandomEngine&& engine, std::vector<T1>* a, T2 mean,
           T2 stddev) noexcept {
  static_assert(std::is_floating_point<T1>::value,
                "T1 must be a floating-point type.");
  std::normal_distribution<T1> dist((T1)mean, (T1)stddev);
  for (size_t i = 0; i < a->size(); ++i) {
    (*a)[i] = dist(engine);
  }
}

template <class RandomEngine, typename T>
void rand(RandomEngine&& engine, T* a) noexcept {
  return detail::rand(engine, a, std::numeric_limits<T>::min(),
                      std::numeric_limits<T>::max(),
                      typename std::is_integral<T>::type());
}

template <class RandomEngine, typename T1, typename T2>
void rand(RandomEngine&& engine, T1* a, T2 lo, T2 hi) noexcept {
  return detail::rand(engine, a, (T1)lo, (T1)hi,
                      typename std::is_integral<T1>::type());
}

template <class RandomEngine, typename T>
void rand(RandomEngine&& engine, std::vector<T>* a) noexcept {
  return detail::rand(engine, a, std::numeric_limits<T>::min(),
                      std::numeric_limits<T>::max(),
                      typename std::is_integral<T>::type());
}

template <class RandomEngine, typename T1, typename T2>
void rand(RandomEngine&& engine, std::vector<T1>* a, T2 lo, T2 hi) noexcept {
  return detail::rand(engine, a, (T1)lo, (T1)hi,
                      typename std::is_integral<T1>::type());
}

/************************************************************************/
/* Print functions */
/************************************************************************/
namespace detail {

inline void PrintHeader2(size_t N, const char* col) noexcept {
  for (size_t i = 0; i < N; ++i) {
    printf(" %8s", col);
  }
  printf("\n");
}

inline void PrintHeader3(const char* col) noexcept {
  printf(" %8s", col);
  printf("\n");
}

template <typename... Args>
void PrintHeader3(const char* col, Args&&... cols) noexcept {
  printf(" %8s", col);
  PrintHeader3(std::forward<Args>(cols)...);
}

template <size_t N>
size_t GetMaxIndex(const double (&gflops)[N]) noexcept {
  double max_gflops = 0;
  size_t max_index = 0;
  for (size_t i = 0; i < N; ++i) {
    if (gflops[i] > max_gflops) {
      max_gflops = gflops[i];
      max_index = i;
    }
  }
  return max_index;
}

template <size_t N>
void PrintContent(const double (&gflops)[N], size_t max_index) noexcept {
  static const int ENABLE_COLOR = isatty(fileno(stdout));
  for (size_t i = 0; i < N; ++i) {
    if (ENABLE_COLOR && i == max_index) {
      printf(" \033[0;35m%8.3f\033[0m", gflops[i]);
    } else {
      printf(" %8.3f", gflops[i]);
    }
  }
  printf("\n");
}

}  // namespace detail

inline void PrintHeader1(size_t prefix_size, size_t N,
                         const char* name) noexcept {
  size_t size = prefix_size + 9 * N;
  size_t name_size = strlen(name);
  size_t left_size = (size - name_size) / 2;
  size_t right_size = size - name_size - left_size;
  for (size_t i = 0; i < left_size; ++i) {
    printf("-");
  }
  printf("%s", name);
  for (size_t i = 0; i < right_size; ++i) {
    printf("-");
  }
  printf("\n");
}

inline void PrintHeader1(size_t N, const char* name) noexcept {
  size_t size = 8 + 9 * N;
  size_t name_size = strlen(name);
  size_t left_size = (size - name_size) / 2;
  size_t right_size = size - name_size - left_size;
  for (size_t i = 0; i < left_size; ++i) {
    printf("-");
  }
  printf("%s", name);
  for (size_t i = 0; i < right_size; ++i) {
    printf("-");
  }
  printf("\n");
}

inline void PrintHeader2(size_t N, const char* col) noexcept {
  printf("%8s", "n");
  detail::PrintHeader2(N, col);
}

inline void PrintHeader2(const char* prefix, size_t N,
                         const char* col) noexcept {
  printf("%s", prefix);
  detail::PrintHeader2(N, col);
}

template <typename... Args>
void PrintHeader3(Args&&... cols) noexcept {
  printf("%8s", "");
  detail::PrintHeader3(std::forward<Args>(cols)...);
}

template <typename... Args>
void PrintHeader3(size_t prefix_size, Args&&... cols) noexcept {
  for (size_t i = 0; i < prefix_size; ++i) {
    printf(" ");
  }
  detail::PrintHeader3(std::forward<Args>(cols)...);
}

template <size_t N>
void PrintContent(int n, const double (&gflops)[N]) noexcept {
  size_t max_index = detail::GetMaxIndex(gflops);
  printf("%8d", n);
  detail::PrintContent(gflops, max_index);
}

template <size_t N>
void PrintContent(const char* prefix, const double (&gflops)[N]) noexcept {
  size_t max_index = detail::GetMaxIndex(gflops);
  printf("%s", prefix);
  detail::PrintContent(gflops, max_index);
}

/************************************************************************/
/* Timing functions */
/************************************************************************/
inline std::chrono::steady_clock::time_point& Begin() noexcept {
  static thread_local std::chrono::steady_clock::time_point begin;
  return begin;
}

inline std::chrono::steady_clock::time_point& End() noexcept {
  static thread_local std::chrono::steady_clock::time_point end;
  return end;
}

inline void BeginTimer() noexcept {
  Begin() = std::chrono::steady_clock::now();
}

inline void EndTimer() noexcept { End() = std::chrono::steady_clock::now(); }

inline double GetSeconds() noexcept {
  auto duration = End() - Begin();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  return (double)ns.count() / 1e+9;
}

inline double GetNanoSeconds() noexcept {
  auto duration = End() - Begin();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  return (double)ns.count();
}

inline double GetGFLOPS(double flo) noexcept { return flo / GetNanoSeconds(); }

/************************************************************************/
/* Misc functions */
/************************************************************************/
inline std::initializer_list<int> GetN() noexcept {
  static std::initializer_list<int> il = {1,  2,  5,  8,   10,  16,  20, 24,
                                          32, 50, 64, 100, 128, 200, 256};
  return il;
}

inline std::initializer_list<int> GetLargeN() noexcept {
  static std::initializer_list<int> il = {1,   2,   5,    8,    10,  16,  20,
                                          24,  32,  50,   64,   100, 128, 200,
                                          256, 512, 1024, 2048, 4096};
  return il;
}

inline void Refer(const void* data, size_t size) noexcept {
  FILE* f = fopen("/dev/null", "w");
  if (f) {
    (void)fwrite(data, size, 1, f);
    (void)fclose(f);
  }
}

#define REFER(t) Refer(&t, sizeof(t))
