// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_gtest.h>
#include <deepx_core/tensor/data_type.h>
#include <deepx_core/tensor/ll_math.h>
#include <vector>

namespace deepx_core {

class LLMathTest : public testing::Test, public DataTypeD {
 protected:
  using vectorf_t = std::vector<float_t>;
};

TEST_F(LLMathTest, zero) {
  vectorf_t x = {0, 1, 2};
  ll_math_t::zero(3, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 0, 0}));
}

TEST_F(LLMathTest, copy) {
  vectorf_t x = {0, 1, 2};
  vectorf_t y(3);
  ll_math_t::copy(3, x.data(), y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({0, 1, 2}));
}

TEST_F(LLMathTest, axpb) {
  vectorf_t x = {0, 1, 2};
  vectorf_t y(3);

  ll_math_t::axpb(3, 0, x.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({0, 0, 0}));

  ll_math_t::axpb(3, 1, x.data(), 1, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({1, 2, 3}));

  ll_math_t::axpb(3, 2, x.data(), 2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 4, 6}));
}

TEST_F(LLMathTest, axpy) {
  vectorf_t x = {0, 1, 2};
  vectorf_t y(3);

  ll_math_t::axpy(3, 0, x.data(), y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({0, 0, 0}));

  ll_math_t::axpy(3, 2, x.data(), y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({0, 2, 4}));

  ll_math_t::axpy(3, 2, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 3, 6}));
}

TEST_F(LLMathTest, axpby) {
  vectorf_t x = {0, 1, 2};
  vectorf_t y(3);

  ll_math_t::axpby(3, 0, x.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({0, 0, 0}));

  ll_math_t::axpby(3, 2, x.data(), 2, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({0, 2, 4}));

  ll_math_t::axpby(3, 2, x.data(), 2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 4, 8}));
}

TEST_F(LLMathTest, xypz) {
  vectorf_t x = {1, 2, 3};
  vectorf_t y = {3, 2, 1};
  vectorf_t z(3);

  ll_math_t::xypz(3, x.data(), y.data(), z.data());
  EXPECT_VECTOR_NEAR(z, vectorf_t({3, 4, 3}));

  ll_math_t::xypz(3, x.data(), x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 6, 12}));
}

TEST_F(LLMathTest, xypbz) {
  vectorf_t x = {1, 2, 3};
  vectorf_t y = {3, 2, 1};
  vectorf_t z(3);

  ll_math_t::xypbz(3, x.data(), y.data(), 0, z.data());
  EXPECT_VECTOR_NEAR(z, vectorf_t({3, 4, 3}));

  ll_math_t::xypbz(3, x.data(), y.data(), 2, z.data());
  EXPECT_VECTOR_NEAR(z, vectorf_t({9, 12, 9}));

  ll_math_t::xypbz(3, x.data(), x.data(), 2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({3, 8, 15}));
}

TEST_F(LLMathTest, xdypz) {
  vectorf_t x = {1, 2, 3};
  vectorf_t y = {1, 2, 3};
  vectorf_t z(3);

  ll_math_t::xdypz(3, x.data(), y.data(), z.data());
  EXPECT_VECTOR_NEAR(z, vectorf_t({1, 1, 1}));

  ll_math_t::xdypz(3, x.data(), x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 3, 4}));
}

TEST_F(LLMathTest, xdypbz) {
  vectorf_t x = {1, 2, 3};
  vectorf_t y = {1, 2, 3};
  vectorf_t z(3);

  ll_math_t::xdypbz(3, x.data(), y.data(), 0, z.data());
  EXPECT_VECTOR_NEAR(z, vectorf_t({1, 1, 1}));

  ll_math_t::xdypbz(3, x.data(), y.data(), 2, z.data());
  EXPECT_VECTOR_NEAR(z, vectorf_t({3, 3, 3}));

  ll_math_t::xdypbz(3, x.data(), x.data(), 2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({3, 5, 7}));
}

TEST_F(LLMathTest, add) {
  vectorf_t x = {1, 2, 3};
  vectorf_t y = {3, 2, 1};
  ll_math_t::add(3, x.data(), y.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({4, 4, 4}));
}

TEST_F(LLMathTest, add_scalar) {
  vectorf_t x;

  x = {0, 1, 2};
  ll_math_t::add_scalar(3, x.data(), 0, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 1, 2}));

  x = {0, 1, 2};
  ll_math_t::add_scalar(3, x.data(), 2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 3, 4}));
}

TEST_F(LLMathTest, add_row) {
  vectorf_t X;
  vectorf_t y = {1, 2};

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::add_row(3, 2, 1, X.data(), 0, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 1, 2, 3, 4, 5}));

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::add_row(3, 2, 2, X.data(), 2, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({2, 6, 6, 10, 10, 14}));
}

TEST_F(LLMathTest, add_col) {
  vectorf_t X;
  vectorf_t y = {1, 2};

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::add_col(2, 3, 1, X.data(), 0, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 1, 2, 3, 4, 5}));

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::add_col(2, 3, 2, X.data(), 2, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({2, 4, 6, 10, 12, 14}));
}

TEST_F(LLMathTest, sub) {
  vectorf_t x = {1, 2, 3};
  vectorf_t y = {3, 2, 1};
  ll_math_t::sub(3, x.data(), y.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-2, 0, 2}));
}

TEST_F(LLMathTest, sub_scalar) {
  vectorf_t x;

  x = {0, 1, 2};
  ll_math_t::sub_scalar(3, x.data(), 0, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 1, 2}));

  x = {0, 1, 2};
  ll_math_t::sub_scalar(3, x.data(), -2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 3, 4}));
}

TEST_F(LLMathTest, sub_row) {
  vectorf_t X;
  vectorf_t y = {1, 2};

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::sub_row(3, 2, 1, X.data(), 0, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 1, 2, 3, 4, 5}));

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::sub_row(3, 2, 2, X.data(), 2, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({-2, -2, 2, 2, 6, 6}));
}

TEST_F(LLMathTest, sub_col) {
  vectorf_t X;
  vectorf_t y = {1, 2};

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::sub_col(2, 3, 1, X.data(), 0, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 1, 2, 3, 4, 5}));

  X = {0, 1, 2, 3, 4, 5};
  ll_math_t::sub_col(2, 3, 2, X.data(), 2, y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({-2, 0, 2, 2, 4, 6}));
}

TEST_F(LLMathTest, mul) {
  vectorf_t x = {1, 2, 3};
  vectorf_t y = {3, 2, 1};
  ll_math_t::mul(3, x.data(), y.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({3, 4, 3}));
}

TEST_F(LLMathTest, mul_scalar) {
  vectorf_t x;

  x = {0, 1, 2};
  ll_math_t::mul_scalar(3, x.data(), 0, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 0, 0}));

  x = {0, 1, 2};
  ll_math_t::mul_scalar(3, x.data(), 2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 2, 4}));
}

TEST_F(LLMathTest, mul_row) {
  vectorf_t X = {0, 1, 2, 3, 4, 5};
  vectorf_t y = {1, 2};
  ll_math_t::mul_row(3, 2, X.data(), y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 2, 2, 6, 4, 10}));
}

TEST_F(LLMathTest, mul_col) {
  vectorf_t X = {0, 1, 2, 3, 4, 5};
  vectorf_t y = {1, 2};
  ll_math_t::mul_col(2, 3, X.data(), y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 1, 2, 6, 8, 10}));
}

TEST_F(LLMathTest, div) {
  vectorf_t x = {2, 4, 6};
  vectorf_t y = {1, 2, 3};
  ll_math_t::div(3, x.data(), y.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 2, 2}));
}

TEST_F(LLMathTest, div_scalar) {
  vectorf_t x;

  x = {0, 1, 2};
  ll_math_t::div_scalar(3, x.data(), 1, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 1, 2}));

  x = {0, 1, 2};
  ll_math_t::div_scalar(3, x.data(), 2, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0, 0.5, 1}));
}

TEST_F(LLMathTest, div_row) {
  vectorf_t X = {0, 1, 2, 3, 4, 5};
  vectorf_t y = {1, 0.5};
  ll_math_t::div_row(3, 2, X.data(), y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 2, 2, 6, 4, 10}));
}

TEST_F(LLMathTest, div_col) {
  vectorf_t X = {0, 1, 2, 3, 4, 5};
  vectorf_t y = {1, 0.5};
  ll_math_t::div_col(2, 3, X.data(), y.data(), X.data());
  EXPECT_VECTOR_NEAR(X, vectorf_t({0, 1, 2, 6, 8, 10}));
}

TEST_F(LLMathTest, inv) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::inv(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.9090909, 0.6666666, 0.5}));
}

TEST_F(LLMathTest, sqrt) {
  vectorf_t x = {1.1, 1.5, 2};
  ll_math_t::sqrt(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({1.0488088, 1.2247448, 1.4142135}));
}

TEST_F(LLMathTest, cbrt) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::cbrt(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-1.0322801, 1.1447142, 1.2599210}));
}

TEST_F(LLMathTest, square) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::square(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({1.21, 2.25, 4}));
}

TEST_F(LLMathTest, cubic) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::cubic(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-1.331, 3.375, 8}));
}

TEST_F(LLMathTest, pow) {
  vectorf_t x = {1.1, 1.5, 2};
  vectorf_t y = {1.5, 2, 3};
  ll_math_t::pow(3, x.data(), y.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({1.1536897, 2.25, 8}));
}

TEST_F(LLMathTest, pow_scalar) {
  vectorf_t x = {1.1, 1.5, 2};
  ll_math_t::pow_scalar(3, x.data(), 2.5, x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({1.2690587, 2.7556759, 5.6568542}));
}

TEST_F(LLMathTest, exp) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::exp(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.3328711, 4.4816891, 7.3890561}));
}

TEST_F(LLMathTest, expm1) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::expm1(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.6671289, 3.4816891, 6.3890561}));
}

TEST_F(LLMathTest, log) {
  vectorf_t x = {1.1, 1.5, 2};
  ll_math_t::log(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.0953101, 0.4054651, 0.6931471}));
}

TEST_F(LLMathTest, safe_log_1) {
  EXPECT_LT(ll_math_t::safe_log(-1), 0.0);
  EXPECT_LT(ll_math_t::safe_log(0), 0.0);
  EXPECT_DOUBLE_NEAR(ll_math_t::safe_log(1.1), 0.0953101);
  EXPECT_DOUBLE_NEAR(ll_math_t::safe_log(1.5), 0.4054651);
  EXPECT_DOUBLE_NEAR(ll_math_t::safe_log(2), 0.6931471);
}

TEST_F(LLMathTest, safe_log_2) {
  vectorf_t x = {1.1, 1.5, 2};
  ll_math_t::safe_log(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.0953101, 0.4054651, 0.6931471}));
}

TEST_F(LLMathTest, sigmoid_1) {
  EXPECT_DOUBLE_NEAR(ll_math_t::sigmoid(0), 0.5);
  EXPECT_DOUBLE_NEAR(ll_math_t::sigmoid(-999.999), 0.0);
  EXPECT_DOUBLE_NEAR(ll_math_t::sigmoid(999.999), 1.0);
  EXPECT_DOUBLE_NEAR(ll_math_t::sigmoid(-1.1), 0.2497398);
  EXPECT_DOUBLE_NEAR(ll_math_t::sigmoid(1.5), 0.8175744);
  EXPECT_DOUBLE_NEAR(ll_math_t::sigmoid(2), 0.8807970);
}

TEST_F(LLMathTest, sigmoid_2) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::sigmoid(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.2497398, 0.8175744, 0.8807970}));
}

TEST_F(LLMathTest, sin) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::sin(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.8912073, 0.9974949, 0.9092974}));
}

TEST_F(LLMathTest, asin) {
  vectorf_t x = {-0.5, 0.5, 0.7};
  ll_math_t::asin(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.5235987, 0.5235987, 0.7753974}));
}

TEST_F(LLMathTest, sinh) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::sinh(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-1.3356474, 2.1292794, 3.6268604}));
}

TEST_F(LLMathTest, asinh) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::asinh(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.9503469, 1.1947632, 1.4436354}));
}

TEST_F(LLMathTest, cos) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::cos(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.4535961, 0.0707372, -0.4161468}));
}

TEST_F(LLMathTest, acos) {
  vectorf_t x = {-0.5, 0.5, 0.7};
  ll_math_t::acos(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2.0943951, 1.0471975, 0.7953988}));
}

TEST_F(LLMathTest, cosh) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::cosh(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({1.6685185, 2.3524096, 3.7621956}));
}

TEST_F(LLMathTest, acosh) {
  vectorf_t x = {1.1, 1.5, 2};
  ll_math_t::acosh(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.4435682, 0.9624236, 1.3169578}));
}

TEST_F(LLMathTest, tan) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::tan(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-1.9647596, 14.1014199, -2.1850398}));
}

TEST_F(LLMathTest, atan) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::atan(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.8329812, 0.9827937, 1.1071487}));
}

TEST_F(LLMathTest, tanh) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::tanh(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.8004990, 0.9051482, 0.9640275}));
}

TEST_F(LLMathTest, atanh) {
  vectorf_t x = {-0.5, 0.5, 0.7};
  ll_math_t::atanh(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({-0.5493061, 0.5493061, 0.8673005}));
}

TEST_F(LLMathTest, abs) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::abs(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({1.1, 1.5, 2}));
}

TEST_F(LLMathTest, max_scalar) {
  vectorf_t x = {2, 3, 5};
  ll_math_t::max_scalar(3, 2.5, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2.5, 3, 5}));
}

TEST_F(LLMathTest, max_1) {
  vectorf_t x = {2, 3, 5};
  EXPECT_DOUBLE_NEAR(ll_math_t::max(3, x.data()), 5);
}

TEST_F(LLMathTest, max_2) {
  vectorf_t x = {2, 3, 5};
  vectorf_t y = {3, 2, 6};
  ll_math_t::max(3, x.data(), y.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({3, 3, 6}));
}

TEST_F(LLMathTest, min_scalar) {
  vectorf_t x = {2, 3, 5};
  ll_math_t::min_scalar(3, 2.5, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 2.5, 2.5}));
}

TEST_F(LLMathTest, min_1) {
  vectorf_t x = {2, 3, 5};
  EXPECT_DOUBLE_NEAR(ll_math_t::min(3, x.data()), 2);
}

TEST_F(LLMathTest, min_2) {
  vectorf_t x = {2, 3, 5};
  vectorf_t y = {3, 2, 6};
  ll_math_t::min(3, x.data(), y.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({2, 2, 5}));
}

TEST_F(LLMathTest, sum) {
  vectorf_t x = {2, 3, 5};
  EXPECT_DOUBLE_NEAR(ll_math_t::sum(3, x.data()), 10);
}

TEST_F(LLMathTest, sum_row) {
  vectorf_t X = {0, 1, 2, 3, 4, 5};
  vectorf_t y(2);

  ll_math_t::sum_row(3, 2, 1, X.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({6, 9}));

  ll_math_t::sum_row(3, 2, 1, X.data(), 1, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({12, 18}));
}

TEST_F(LLMathTest, sum_col) {
  vectorf_t X = {0, 1, 2, 3, 4, 5};
  vectorf_t y(3);

  ll_math_t::sum_col(3, 2, 1, X.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({1, 5, 9}));

  ll_math_t::sum_col(3, 2, 1, X.data(), 1, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({2, 10, 18}));
}

TEST_F(LLMathTest, dot) {
  vectorf_t x = {2, 3, 5};
  vectorf_t y = {1, 1, 1};
  EXPECT_DOUBLE_NEAR(ll_math_t::dot(3, x.data(), x.data()), 38);
  EXPECT_DOUBLE_NEAR(ll_math_t::dot(3, x.data(), y.data()), 10);
  EXPECT_DOUBLE_NEAR(ll_math_t::dot(3, y.data(), y.data()), 3);
}

TEST_F(LLMathTest, norm1) {
  vectorf_t x = {-2, 3, -5};
  EXPECT_DOUBLE_NEAR(ll_math_t::norm1(3, x.data()), 10);
}

TEST_F(LLMathTest, norm2) {
  vectorf_t x = {-2, 3, -5};
  EXPECT_DOUBLE_NEAR(ll_math_t::norm2(3, x.data()), 6.1644140);
}

TEST_F(LLMathTest, euclidean_distance) {
  vectorf_t x = {2, 3, 5};
  vectorf_t y = {1, 1, 1};
  EXPECT_DOUBLE_NEAR(ll_math_t::euclidean_distance(3, x.data(), x.data()), 0);
  EXPECT_DOUBLE_NEAR(ll_math_t::euclidean_distance(3, x.data(), y.data()),
                     4.5825757);
  EXPECT_DOUBLE_NEAR(ll_math_t::euclidean_distance(3, y.data(), y.data()), 0);
}

TEST_F(LLMathTest, softmax) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::softmax(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.0272764, 0.3672427, 0.6054809}));
}

TEST_F(LLMathTest, softmax2) {
  vectorf_t x = {-1.1, 1.5, 2};
  ll_math_t::softmax2(3, x.data(), x.data());
  EXPECT_VECTOR_NEAR(x, vectorf_t({0.0818292, 1.1017281, 1.8164425}));
}

TEST_F(LLMathTest, gemv) {
  vectorf_t A, x, y;

  A = {1, 2, 3, 666, 666, 666, 4, 5, 6, 666, 666, 666};
  x = {1, 1, 1};
  y.resize(2);
  ll_math_t::gemv(0, 2, 3, 1, A.data(), 6, x.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({6, 15}));

  A = {1, 2, 3, 666, 666, 666, 4, 5, 6, 666, 666, 666};
  x = {1, 1};
  y.resize(3);
  ll_math_t::gemv(1, 2, 3, 1, A.data(), 6, x.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({5, 7, 9}));

  A = {1, 2, 3, 4, 5, 6};
  x = {1, 1, 1};
  y.resize(2);
  ll_math_t::gemv(0, 2, 3, 0.5, A.data(), x.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({3, 7.5}));
  ll_math_t::gemv(0, 2, 3, 0.5, A.data(), x.data(), 1, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({6, 15}));

  A = {1, 2, 3, 4, 5, 6};
  x = {1, 1};
  y.resize(3);
  ll_math_t::gemv(1, 2, 3, 0.5, A.data(), x.data(), 0, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({2.5, 3.5, 4.5}));
  ll_math_t::gemv(1, 2, 3, 0.5, A.data(), x.data(), 1, y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({5, 7, 9}));

  A = {1, 2, 3, 4, 5, 6};
  x = {1, 1, 1};
  y.resize(2);
  ll_math_t::gemv(0, 2, 3, A.data(), x.data(), y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({6, 15}));

  A = {1, 2, 3, 4, 5, 6};
  x = {1, 1};
  y.resize(3);
  ll_math_t::gemv(1, 2, 3, A.data(), x.data(), y.data());
  EXPECT_VECTOR_NEAR(y, vectorf_t({5, 7, 9}));
}

TEST_F(LLMathTest, gemm_1) {
  int batch = 5;
  int in = 6;
  int out = 8;
  int n = 3;
  int ldX = in * n;
  int ldY = out;
  int ldZ = out * n;
  vectorf_t X(batch * in * n);
  vectorf_t Y(in * out);
  vectorf_t Z(batch * out * n);
  const float_t* pX;
  float_t* pZ;

  for (size_t i = 0; i < X.size(); ++i) {
    X[i] = (float_t)i;
  }
  for (size_t i = 0; i < Y.size(); ++i) {
    Y[i] = (float_t)i;
  }
  pX = &X[0];
  pZ = &Z[0];
  for (int i = 0; i < n; ++i) {
    ll_math_t::gemm(0, 0, batch, out, in, 1, pX, ldX, &Y[0], ldY, 0, pZ, ldZ);
    pX += in;
    pZ += out;
  }

  EXPECT_VECTOR_NEAR(
      Z,
      vectorf_t({440,   455,   470,   485,   500,   515,   530,   545,   1160,
                 1211,  1262,  1313,  1364,  1415,  1466,  1517,  1880,  1967,
                 2054,  2141,  2228,  2315,  2402,  2489,  2600,  2723,  2846,
                 2969,  3092,  3215,  3338,  3461,  3320,  3479,  3638,  3797,
                 3956,  4115,  4274,  4433,  4040,  4235,  4430,  4625,  4820,
                 5015,  5210,  5405,  4760,  4991,  5222,  5453,  5684,  5915,
                 6146,  6377,  5480,  5747,  6014,  6281,  6548,  6815,  7082,
                 7349,  6200,  6503,  6806,  7109,  7412,  7715,  8018,  8321,
                 6920,  7259,  7598,  7937,  8276,  8615,  8954,  9293,  7640,
                 8015,  8390,  8765,  9140,  9515,  9890,  10265, 8360,  8771,
                 9182,  9593,  10004, 10415, 10826, 11237, 9080,  9527,  9974,
                 10421, 10868, 11315, 11762, 12209, 9800,  10283, 10766, 11249,
                 11732, 12215, 12698, 13181, 10520, 11039, 11558, 12077, 12596,
                 13115, 13634, 14153}));
}

TEST_F(LLMathTest, gemm_2) {
  vectorf_t X = {1, 2, 3, 4, 5, 6};
  vectorf_t Y = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5};
  vectorf_t Z(10);

  ll_math_t::gemm(0, 0, 2, 5, 3, 0.5, X.data(), Y.data(), 0, Z.data());
  EXPECT_VECTOR_NEAR(
      Z, vectorf_t({8.5, 9.5, 11, 11.5, 12.5, 19, 21.5, 24.5, 26.5, 29}));

  ll_math_t::gemm(0, 0, 2, 5, 3, 0.5, X.data(), Y.data(), 1, Z.data());
  EXPECT_VECTOR_NEAR(Z, vectorf_t({17, 19, 22, 23, 25, 38, 43, 49, 53, 58}));

  ll_math_t::gemm(0, 0, 2, 5, 3, 0, X.data(), Y.data(), 0, Z.data());
  EXPECT_VECTOR_NEAR(Z, vectorf_t({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  ll_math_t::gemm(0, 0, 2, 5, 3, X.data(), Y.data(), Z.data());
  EXPECT_VECTOR_NEAR(Z, vectorf_t({17, 19, 22, 23, 25, 38, 43, 49, 53, 58}));

  ll_math_t::gemm(1, 0, 2, 5, 3, X.data(), Y.data(), Z.data());
  EXPECT_VECTOR_NEAR(Z, vectorf_t({27, 30, 35, 36, 39, 34, 38, 44, 46, 50}));

  ll_math_t::gemm(0, 1, 2, 5, 3, X.data(), Y.data(), Z.data());
  EXPECT_VECTOR_NEAR(Z, vectorf_t({6, 12, 18, 24, 30, 15, 30, 45, 60, 75}));

  ll_math_t::gemm(1, 1, 2, 5, 3, X.data(), Y.data(), Z.data());
  EXPECT_VECTOR_NEAR(Z, vectorf_t({9, 18, 27, 36, 45, 12, 24, 36, 48, 60}));
}

}  // namespace deepx_core
