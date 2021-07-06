// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/half.h>
#include "benchmark.h"
#include "internal_macro.h"

ATTR_NOINLINE sage2_half_t sage2_d2h_ref(double d) {
  union {
    double d;
    uint64_t dbits;
  } conv;
  uint64_t d_exp, d_sig;
  uint16_t h_sgn, h_exp, h_sig, tmp;

  conv.d = d;
  h_sgn = (uint16_t)((conv.dbits & UINT64_C(0x8000000000000000)) >> 48);
  d_exp = conv.dbits & UINT64_C(0x7ff0000000000000);
  if (d_exp >= UINT64_C(0x40f0000000000000)) {
    if (d_exp == UINT64_C(0x7ff0000000000000)) {
      d_sig = conv.dbits & UINT64_C(0x000fffffffffffff);
      if (d_sig != 0) {
        tmp = (uint16_t)(0x7c00 + (d_sig >> 42));
        if (tmp == 0x7c00) {
          ++tmp;
        }
        return h_sgn + tmp;
      } else {
        return h_sgn + 0x7c00;
      }
    } else {
      return h_sgn + 0x7c00;
    }
  }

  if (d_exp <= UINT64_C(0x3f00000000000000)) {
    if (d_exp < UINT64_C(0x3e60000000000000)) {
      return h_sgn;
    }
    d_exp >>= 52;
    d_sig = UINT64_C(0x0010000000000000) +
            (conv.dbits & UINT64_C(0x000fffffffffffff));
    d_sig >>= (1009 - d_exp);
    if ((d_sig & UINT64_C(0x000007ffffffffff)) !=
        UINT64_C(0x0000020000000000)) {
      d_sig += UINT64_C(0x0000020000000000);
    }
    h_sig = (uint16_t)(d_sig >> 42);
    return h_sgn + h_sig;
  }

  h_exp = (uint16_t)((d_exp - UINT64_C(0x3f00000000000000)) >> 42);
  d_sig = conv.dbits & UINT64_C(0x000fffffffffffff);
  if ((d_sig & UINT64_C(0x000007ffffffffff)) != UINT64_C(0x0000020000000000)) {
    d_sig += UINT64_C(0x0000020000000000);
  }
  h_sig = (uint16_t)(d_sig >> 42);
  return h_sgn + h_exp + h_sig;
}

ATTR_NOINLINE double sage2_h2d_ref(sage2_half_t h) {
  union {
    double d;
    uint64_t dbits;
  } conv;
  uint16_t h_exp, h_sig;
  uint64_t d_sgn, d_exp, d_sig;

  h_exp = (h & 0x7c00);
  d_sgn = ((uint64_t)h & 0x8000) << 48;
  switch (h_exp) {
    case 0x0000:
      h_sig = h & 0x03ff;
      if (h_sig == 0) {
        conv.dbits = d_sgn;
        break;
      }
      h_sig <<= 1;
      while ((h_sig & 0x0400) == 0) {
        h_sig <<= 1;
        h_exp++;
      }
      d_exp = ((uint64_t)(1023 - 15 - h_exp)) << 52;
      d_sig = ((uint64_t)(h_sig & 0x03ff)) << 42;
      conv.dbits = d_sgn + d_exp + d_sig;
      break;
    case 0x7c00:
      conv.dbits = d_sgn + UINT64_C(0x7ff0000000000000) +
                   ((((uint64_t)h & 0x03ff)) << 42);
      break;
    default:
      conv.dbits = d_sgn + ((((uint64_t)h & 0x7fff) + 0xfc000) << 42);
      break;
  }
  return conv.d;
}

ATTR_NOINLINE sage2_half_t sage2_s2h_ref(float s) {
  union {
    float s;
    uint32_t sbits;
  } conv;
  uint32_t s_exp, s_sig;
  uint16_t h_sgn, h_exp, h_sig, tmp;

  conv.s = s;
  h_sgn = (uint16_t)((conv.sbits & 0x80000000) >> 16);
  s_exp = conv.sbits & 0x7f800000;
  if (s_exp >= 0x47800000) {
    if (s_exp == 0x7f800000) {
      s_sig = conv.sbits & 0x007fffff;
      if (s_sig != 0) {
        tmp = (uint16_t)(0x7c00 + (s_sig >> 13));
        if (tmp == 0x7c00) {
          ++tmp;
        }
        return h_sgn + tmp;
      } else {
        return h_sgn + 0x7c00;
      }
    } else {
      return h_sgn + 0x7c00;
    }
  }

  if (s_exp <= 0x38000000) {
    if (s_exp < 0x33000000) {
      return h_sgn;
    }
    s_exp >>= 23;
    s_sig = 0x00800000 + (conv.sbits & 0x007fffff);
    s_sig >>= (113 - s_exp);
    if ((s_sig & 0x00003fff) != 0x00001000) {
      s_sig += 0x00001000;
    }
    h_sig = (uint16_t)(s_sig >> 13);
    return h_sgn + h_sig;
  }

  h_exp = (uint16_t)((s_exp - 0x38000000) >> 13);
  s_sig = conv.sbits & 0x007fffff;
  if ((s_sig & 0x00003fff) != 0x00001000) {
    s_sig += 0x00001000;
  }
  h_sig = (uint16_t)(s_sig >> 13);
  return h_sgn + h_exp + h_sig;
}

ATTR_NOINLINE float sage2_h2s_ref(sage2_half_t h) {
  union {
    float s;
    uint32_t sbits;
  } conv;
  uint16_t h_exp, h_sig;
  uint32_t s_sgn, s_exp, s_sig;

  h_exp = h & 0x7c00;
  s_sgn = ((uint32_t)h & 0x8000) << 16;
  switch (h_exp) {
    case 0x0000:
      h_sig = h & 0x03ff;
      if (h_sig == 0) {
        conv.sbits = s_sgn;
        break;
      }
      h_sig <<= 1;
      while ((h_sig & 0x0400) == 0) {
        h_sig <<= 1;
        h_exp++;
      }
      s_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
      s_sig = ((uint32_t)(h_sig & 0x03ff)) << 13;
      conv.sbits = s_sgn + s_exp + s_sig;
      break;
    case 0x7c00:
      conv.sbits = s_sgn + 0x7f800000 + (((uint32_t)h & 0x03ff) << 13);
      break;
    default:
      conv.sbits = s_sgn + ((((uint32_t)h & 0x7fff) + 0x1c000) << 13);
      break;
  }
  return conv.s;
}

ATTR_NOINLINE void sage2_pd2ph_ref(uint64_t n, const double* pd,
                                   sage2_half_t* ph) {
  for (uint64_t i = 0; i < n; ++i) {
    ph[i] = sage2_d2h_ref(pd[i]);
  }
}

ATTR_NOINLINE void sage2_ph2pd_ref(uint64_t n, const sage2_half_t* ph,
                                   double* pd) {
  for (uint64_t i = 0; i < n; ++i) {
    pd[i] = sage2_h2d_ref(ph[i]);
  }
}

ATTR_NOINLINE void sage2_ps2ph_ref(uint64_t n, const float* ps,
                                   sage2_half_t* ph) {
  for (uint64_t i = 0; i < n; ++i) {
    ph[i] = sage2_s2h_ref(ps[i]);
  }
}

ATTR_NOINLINE void sage2_ph2ps_ref(uint64_t n, const sage2_half_t* ph,
                                   float* ps) {
  for (uint64_t i = 0; i < n; ++i) {
    ps[i] = sage2_h2s_ref(ph[i]);
  }
}

void Test_sage2_d2h_sage2_h2d() {
  std::default_random_engine engine;
  std::uniform_real_distribution<double> dist(-1, 1);
  double d;

  for (int i = 0; i < 10000; ++i) {
    d = dist(engine);
    CHECK_EQUAL(sage2_h2d_ref(sage2_d2h_ref(d)), sage2_h2d(sage2_d2h(d)));
  }
}

void Test_sage2_s2h_sage2_h2s() {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-1, 1);
  float s;

  for (int i = 0; i < 10000; ++i) {
    s = dist(engine);
    CHECK_EQUAL(sage2_h2s_ref(sage2_s2h_ref(s)), sage2_h2s(sage2_s2h(s)));
  }
}

void Test_sage2_pd2ph_sage2_ph2pd(int n) {
  std::default_random_engine engine;
  std::uniform_real_distribution<double> dist(-1, 1);
  std::vector<double> d(n);
  std::vector<sage2_half_t> h1(n), h2(n);
  std::vector<double> d1(n), d2(n);
  randn(engine, &d);

  sage2_pd2ph_ref(n, d.data(), h1.data());
  sage2_ph2pd_ref(n, h1.data(), d1.data());
  sage2_pd2ph(n, d.data(), h2.data());
  sage2_ph2pd(n, h2.data(), d2.data());
  CHECK_EQUAL(d1, d2);
}

void Test_sage2_ps2ph_sage2_ph2ps(int n) {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-1, 1);
  std::vector<float> s(n);
  std::vector<sage2_half_t> h1(n), h2(n);
  std::vector<float> s1(n), s2(n);
  randn(engine, &s);

  sage2_ps2ph_ref(n, s.data(), h1.data());
  sage2_ph2ps_ref(n, h1.data(), s1.data());
  sage2_ps2ph(n, s.data(), h2.data());
  sage2_ph2ps(n, h2.data(), s2.data());
  CHECK_EQUAL(s1, s2);
}

template <class Func>
double Benchmark_sage2_pd2ph(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<double> d(n);
  std::vector<sage2_half_t> h(n);
  randn(engine, &d);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, d.data(), h.data());
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark_sage2_pd2ph(int n) {
  int m = 100000000 / n;
  double mops[2] = {0};
  mops[0] = Benchmark_sage2_pd2ph(sage2_pd2ph_ref, n, m);
  mops[1] = Benchmark_sage2_pd2ph(sage2_pd2ph, n, m);
  PrintContent(n, mops);
}

template <class Func>
double Benchmark_sage2_ph2pd(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<sage2_half_t> h(n);
  std::vector<double> d(n);
  rand(engine, &h);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, h.data(), d.data());
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark_sage2_ph2pd(int n) {
  int m = 100000000 / n;
  double mops[2] = {0};
  mops[0] = Benchmark_sage2_ph2pd(sage2_ph2pd_ref, n, m);
  mops[1] = Benchmark_sage2_ph2pd(sage2_ph2pd, n, m);
  PrintContent(n, mops);
}

template <class Func>
double Benchmark_sage2_ps2ph(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<float> s(n);
  std::vector<sage2_half_t> h(n);
  randn(engine, &s);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, s.data(), h.data());
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark_sage2_ps2ph(int n) {
  int m = 100000000 / n;
  double mops[2] = {0};
  mops[0] = Benchmark_sage2_ps2ph(sage2_ps2ph_ref, n, m);
  mops[1] = Benchmark_sage2_ps2ph(sage2_ps2ph, n, m);
  PrintContent(n, mops);
}

template <class Func>
double Benchmark_sage2_ph2ps(Func&& func, int n, int m) {
  std::default_random_engine engine;
  std::vector<sage2_half_t> h(n);
  std::vector<float> s(n);
  rand(engine, &h);

  BeginTimer();
  for (int i = 0; i < m; ++i) {
    func(n, h.data(), s.data());
  }
  EndTimer();
  return 1e-6 * m / GetSeconds();
}

void Benchmark_sage2_ph2ps(int n) {
  int m = 100000000 / n;
  double mops[2] = {0};
  mops[0] = Benchmark_sage2_ph2ps(sage2_ph2ps_ref, n, m);
  mops[1] = Benchmark_sage2_ph2ps(sage2_ph2ps, n, m);
  PrintContent(n, mops);
}

int main(int argc, char** argv) {
  int action = 3;
  if (argc > 1) {
    action = std::stoi(argv[1]);
  }

  if (action & 1) {
    Test_sage2_d2h_sage2_h2d();
    Test_sage2_s2h_sage2_h2s();
    for (int n : GetN()) {
      Test_sage2_pd2ph_sage2_ph2pd(n);
    }
    for (int n : GetN()) {
      Test_sage2_ps2ph_sage2_ph2ps(n);
    }
  }

  if (action & 2) {
    PrintHeader1(2, "pd2ph");
    PrintHeader2(2, "Mop/s");
    PrintHeader3("ref", "opt");
    for (int n : GetLargeN()) {
      Benchmark_sage2_pd2ph(n);
    }
    PrintHeader1(2, "ph2pd");
    PrintHeader2(2, "Mop/s");
    PrintHeader3("ref", "opt");
    for (int n : GetLargeN()) {
      Benchmark_sage2_ph2pd(n);
    }
    PrintHeader1(2, "ps2ph");
    PrintHeader2(2, "Mop/s");
    PrintHeader3("ref", "opt");
    for (int n : GetLargeN()) {
      Benchmark_sage2_ps2ph(n);
    }
    PrintHeader1(2, "ph2ps");
    PrintHeader2(2, "Mop/s");
    PrintHeader3("ref", "opt");
    for (int n : GetLargeN()) {
      Benchmark_sage2_ph2ps(n);
    }
  }
  return 0;
}
