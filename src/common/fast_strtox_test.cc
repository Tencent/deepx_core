// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/fast_strtox.h>
#include <deepx_core/dx_gtest.h>
#include <cstdint>
#include <string>

namespace deepx_core {

class FastStrtoxTest : public testing::Test {
 protected:
  // 'std::isinf' always returns false with -ffast-math.
  // https://stackoverflow.com/questions/22931147/stdisinf-does-not-work-with-ffast-math-how-to-check-for-infinity
  static bool isinf(double d) noexcept {
    union {
      double d;
      uint64_t u;
    } a = {d};
    switch (a.u) {
      // IEEE 754 inf
      case UINT64_C(0x7ff0000000000000):
      // IEEE 754 -inf
      case UINT64_C(0xfff0000000000000):
        return true;
      default:
        return false;
    }
  }
};

TEST_F(FastStrtoxTest, fast_strtod) {
  std::string s;
  char* end;

  s = "9527";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 9527);
  EXPECT_EQ(end, s.data() + s.size());

  s = "9527a";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 9527);
  EXPECT_EQ(end, s.data() + 4);

  s = "9527 ";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 9527);
  EXPECT_EQ(end, s.data() + 4);

  s = "9527.";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 9527);
  EXPECT_EQ(end, s.data() + s.size());

  s = "-9527.";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), -9527);
  EXPECT_EQ(end, s.data() + s.size());

  s = "95.27";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 95.27);
  EXPECT_EQ(end, s.data() + s.size());

  s = "95.27E0";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 95.27);
  EXPECT_EQ(end, s.data() + s.size());

  s = "95.27E-0";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 95.27);
  EXPECT_EQ(end, s.data() + s.size());

  s = "+95.27E0";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 95.27);
  EXPECT_EQ(end, s.data() + s.size());

  s = "9.527E1";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 95.27);
  EXPECT_EQ(end, s.data() + s.size());

  s = "9.527E+1";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 95.27);
  EXPECT_EQ(end, s.data() + s.size());

  s = "9527e-2";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 95.27);
  EXPECT_EQ(end, s.data() + s.size());

  s = "-9.527e-3";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), -0.009527);
  EXPECT_EQ(end, s.data() + s.size());

  s = "95.27E-4";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 0.009527);
  EXPECT_EQ(end, s.data() + s.size());

  s = ".9527E-2";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 0.009527);
  EXPECT_EQ(end, s.data() + s.size());

  // very long string
  s = "3.14159265358979323846264338327950288419716939937510582097494";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 3.141592653);

  s = "314159265358979323846264338327950288419716939937510582097494";
  EXPECT_DOUBLE_NEAR(fast_strtod(s.data(), &end), 3.14159265358979e+59);

  s = "1e314159265358979323846264338327950288419716939937510582097494";
  EXPECT_TRUE(isinf(fast_strtod(s.data(), &end)));

  s = "-1e314159265358979323846264338327950288419716939937510582097494";
  EXPECT_TRUE(isinf(fast_strtod(s.data(), &end)));
}

TEST_F(FastStrtoxTest, fast_strtoi) {
  std::string s;
  char* end;

  s = "123";
  EXPECT_EQ(fast_strtoi<uint64_t>(s.data(), &end), 123u);
  EXPECT_EQ(end, s.data() + s.size());

  s = "-123";
  EXPECT_EQ(fast_strtoi<uint64_t>(s.data(), &end), 0u);
  EXPECT_EQ(end, s.data());

  s = "123456abc";
  EXPECT_EQ(fast_strtoi<uint32_t>(s.data(), &end), 123456u);
  EXPECT_EQ(end, s.data() + 6);

  s = "abc";
  EXPECT_EQ(fast_strtoi<uint32_t>(s.data(), &end), 0u);
  EXPECT_EQ(end, s.data());

  s = "123";
  EXPECT_EQ(fast_strtoi<int64_t>(s.data(), &end), 123);
  EXPECT_EQ(end, s.data() + s.size());

  s = "-123";
  EXPECT_EQ(fast_strtoi<int64_t>(s.data(), &end), -123);
  EXPECT_EQ(end, s.data() + s.size());

  s = "123456abc";
  EXPECT_EQ(fast_strtoi<int32_t>(s.data(), &end), 123456);
  EXPECT_EQ(end, s.data() + 6);

  s = "-123456abc";
  EXPECT_EQ(fast_strtoi<int32_t>(s.data(), &end), -123456);
  EXPECT_EQ(end, s.data() + 7);

  s = "abc";
  EXPECT_EQ(fast_strtoi<int64_t>(s.data(), &end), 0);
  EXPECT_EQ(end, s.data());
}

}  // namespace deepx_core
