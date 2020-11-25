// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_gtest.h>
#include <deepx_core/tensor/data_type.h>
#include <deepx_core/tensor/tensor.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace deepx_core {

class TensorTest : public testing::Test, public DataTypeD {
 protected:
  using vectorf_t = std::vector<float_t>;
  using vectors_t = std::vector<std::string>;

  static vectorf_t VF234;
  static vectorf_t VF2N34;
  static vectorf_t VF567;
  static vectorf_t VF234567;
  static vectors_t VS234;
  std::default_random_engine engine;
};

TensorTest::vectorf_t TensorTest::VF234{2, 3, 4};
TensorTest::vectorf_t TensorTest::VF2N34{2, -3, 4};
TensorTest::vectorf_t TensorTest::VF567{5, 6, 7};
TensorTest::vectorf_t TensorTest::VF234567{2, 3, 4, 5, 6, 7};
TensorTest::vectors_t TensorTest::VS234{"2", "3", "4"};

TEST_F(TensorTest, Copy) {
  tsr_t X(VF234);
  tsr_t Y = X;
  tsr_t Z;
  Z = X;
  EXPECT_FALSE(X.is_view());
  EXPECT_FALSE(Y.is_view());
  EXPECT_FALSE(Z.is_view());
  EXPECT_EQ(X, Y);
  EXPECT_EQ(X, Z);
}

TEST_F(TensorTest, Copy_view) {
  tsr_t X;
  X.view(Shape(3), VF234.data());
  tsr_t Y = X;
  tsr_t Z;
  Z = X;
  EXPECT_TRUE(X.is_view());
  EXPECT_TRUE(Y.is_view());
  EXPECT_TRUE(Z.is_view());
  EXPECT_EQ(X, Y);
  EXPECT_EQ(X, Z);
}

TEST_F(TensorTest, Move) {
  tsr_t X(VF234);
  EXPECT_FALSE(X.is_view());
  EXPECT_EQ(X, vectorf_t({2, 3, 4}));

  tsr_t Y(std::move(X));
  EXPECT_FALSE(Y.is_view());
  EXPECT_EQ(Y, vectorf_t({2, 3, 4}));

  tsr_t Z;
  Z = std::move(Y);
  EXPECT_FALSE(Z.is_view());
  EXPECT_EQ(Z, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, Move_view) {
  tsr_t X;
  X.view(Shape(3), VF234.data());
  EXPECT_TRUE(X.is_view());
  EXPECT_EQ(X, vectorf_t({2, 3, 4}));

  tsr_t Y(std::move(X));
  EXPECT_TRUE(Y.is_view());
  EXPECT_EQ(Y, vectorf_t({2, 3, 4}));

  tsr_t Z;
  Z = std::move(Y);
  EXPECT_TRUE(Z.is_view());
  EXPECT_EQ(Z, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, Construct_std_il) {
  tsr_t X{2, 3, 4};
  EXPECT_TRUE(X.same_shape(3));
  EXPECT_EQ(X, vectorf_t({2, 3, 4}));

  tsr_t Y;
  Y = {{2, 3, 4}, {5, 6, 7}};
  EXPECT_TRUE(Y.same_shape(2, 3));
  EXPECT_EQ(Y, vectorf_t({2, 3, 4, 5, 6, 7}));
}

TEST_F(TensorTest, Construct_std_vector) {
  tsr_t X(VF234);
  EXPECT_TRUE(X.same_shape(3));
  EXPECT_EQ(X, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, Construct_rvalue_std_vector) {
  vectorf_t VF234_COPY = VF234;
  tsr_t X(std::move(VF234_COPY));
  EXPECT_TRUE(X.same_shape(3));
  EXPECT_EQ(X, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, Construct_shape) {
  Shape shape(2, 3, 4);
  tsr_t X(shape);
  EXPECT_TRUE(X.same_shape(2, 3, 4));
  EXPECT_TRUE(X.same_shape(shape));
}

TEST_F(TensorTest, Construct_ii) {
  tsr_t X(VF234.begin(), VF234.end());
  EXPECT_TRUE(X.same_shape(3));
  EXPECT_EQ(X, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, assign) {
  tsr_t X;
  X.assign(VF234.begin(), VF234.end());
  EXPECT_TRUE(X.same_shape(3));
  EXPECT_EQ(X, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, set_data) {
  tsr_t X, Y;
  X.resize(3);
  Y.resize(3);
  EXPECT_EQ(X.set_data(VF234), VF234);
  EXPECT_EQ(X.set_data({5, 6, 7}), VF567);
  EXPECT_EQ(X.set_data(VF567.data(), VF567.size()), VF567);
  EXPECT_EQ(Y.set_data(X), VF567);
  EXPECT_EQ(X.set_data(VF234.begin(), VF234.end()), VF234);

  X.resize(2);
  Y.resize(3);
  EXPECT_ANY_THROW(X.set_data(VF234));
  EXPECT_ANY_THROW(X.set_data({5, 6, 7}));
  EXPECT_ANY_THROW(X.set_data(VF567.data(), VF567.size()));
  EXPECT_ANY_THROW(Y.set_data(X));
  EXPECT_ANY_THROW(X.set_data(VF234.begin(), VF234.end()));
}

TEST_F(TensorTest, ElementAccess) {
  tsr_t X(VF234);
  EXPECT_EQ(X.shape(), Shape(3));
  EXPECT_EQ(X.data(0), 2);
  EXPECT_EQ(X.data(1), 3);
  EXPECT_EQ(X.data(2), 4);
  EXPECT_EQ(X.front(), 2);
  EXPECT_EQ(X.back(), 4);

  const tsr_t& cX = X;
  EXPECT_EQ(cX.shape(), Shape(3));
  EXPECT_EQ(cX.data(0), 2);
  EXPECT_EQ(cX.data(1), 3);
  EXPECT_EQ(cX.data(2), 4);
  EXPECT_EQ(cX.front(), 2);
  EXPECT_EQ(cX.back(), 4);
}

TEST_F(TensorTest, ElementAccess_view) {
  tsr_t X;
  X.view(Shape(3), VF234.data());
  EXPECT_EQ(X.shape(), Shape(3));
  EXPECT_EQ(X.data(0), 2);
  EXPECT_EQ(X.data(1), 3);
  EXPECT_EQ(X.data(2), 4);
  EXPECT_EQ(X.front(), 2);
  EXPECT_EQ(X.back(), 4);

  const tsr_t& cX = X;
  EXPECT_EQ(cX.shape(), Shape(3));
  EXPECT_EQ(cX.data(0), 2);
  EXPECT_EQ(cX.data(1), 3);
  EXPECT_EQ(cX.data(2), 4);
  EXPECT_EQ(cX.front(), 2);
  EXPECT_EQ(cX.back(), 4);
}

TEST_F(TensorTest, iterator) {
  tsr_t X(VF234);
  float_t sum = 0;
  for (float_t value : X) {
    sum += value;
  }
  EXPECT_EQ(sum, 9);

  const tsr_t& cX = X;
  sum = 0;
  for (float_t value : cX) {
    sum += value;
  }
  EXPECT_EQ(sum, 9);
}

TEST_F(TensorTest, iterator_view) {
  tsr_t X;
  X.view(Shape(3), VF234.data());
  float_t sum = 0;
  for (float_t value : X) {
    sum += value;
  }
  EXPECT_EQ(sum, 9);

  const tsr_t& cX = X;
  sum = 0;
  for (float_t value : cX) {
    sum += value;
  }
  EXPECT_EQ(sum, 9);
}

TEST_F(TensorTest, get_view) {
  tsr_t X(VF234);
  tsr_t Y = X.get_view();
  tsr_t Z = Y.get_view();
  EXPECT_FALSE(X.is_view());
  EXPECT_TRUE(Y.is_view());
  EXPECT_TRUE(Z.is_view());
  EXPECT_EQ(X, Y);
  EXPECT_EQ(X, Z);
}

TEST_F(TensorTest, Subscript) {
  tsr_t X;
  X.resize(2, 3, 4, 5).arange();
  EXPECT_FALSE(X.is_view());

  const tsr_t X0 = X[0];
  EXPECT_TRUE(X0.is_view());
  EXPECT_TRUE(X0.same_shape(3, 4, 5));
  EXPECT_EQ(X0.data(0), 0);

  tsr_t X00 = X0[0];
  EXPECT_TRUE(X00.is_view());
  EXPECT_TRUE(X00.same_shape(4, 5));
  EXPECT_EQ(X00.data(0), 0);

  tsr_t X11 = X[1][1];
  EXPECT_TRUE(X11.is_view());
  EXPECT_TRUE(X11.same_shape(4, 5));
  EXPECT_EQ(X11.data(0), 80);

  EXPECT_ANY_THROW(X[3]);
  EXPECT_ANY_THROW(X[4]);
  EXPECT_ANY_THROW(X0[4]);
  EXPECT_ANY_THROW(X0[5]);
  EXPECT_ANY_THROW(X00[5]);
  EXPECT_ANY_THROW(X00[6]);
  EXPECT_ANY_THROW(X11[5]);
  EXPECT_ANY_THROW(X11[6]);
}

TEST_F(TensorTest, view) {
  tsr_t X;
  EXPECT_FALSE(X.is_view());
  X.view(Shape(6), VF234567.data());
  EXPECT_TRUE(X.is_view());
  X.view(Shape(2, 3), VF234567.data());
  EXPECT_TRUE(X.is_view());
  X.view(Shape(3, 2), VF234567.data());
  EXPECT_TRUE(X.is_view());
}

TEST_F(TensorTest, resize) {
  tsr_t X;
  EXPECT_TRUE(X.resize(2).same_shape(Shape(2)));
  EXPECT_TRUE(X.resize(2, 3).same_shape(Shape(2, 3)));
  EXPECT_TRUE(X.resize(2, 3, 4).same_shape(Shape(2, 3, 4)));
  EXPECT_TRUE(X.resize(2, 3, 4, 5).same_shape(Shape(2, 3, 4, 5)));
}

TEST_F(TensorTest, resize_view) {
  tsr_t X;
  X.view(Shape(3), VF234.data());
  EXPECT_ANY_THROW(X.resize(3));
}

TEST_F(TensorTest, reshape) {
  tsr_t X;
  X.resize(6).reshape(2, 3).reshape(3, 2).reshape(1, 6).reshape(6, 1);
}

TEST_F(TensorTest, expand_dim) {
  tsr_t X;
  X.resize(2, 3, 4).expand_dim(2).expand_dim(1).expand_dim(0);
  EXPECT_EQ(X.shape(), Shape(1, 2, 1, 3, 1, 4));
}

TEST_F(TensorTest, squeeze) {
  tsr_t X;
  X.resize(1, 2, 1, 3, 1, 4).squeeze(4).squeeze(2).squeeze(0);
  EXPECT_EQ(X.shape(), Shape(2, 3, 4));
}

TEST_F(TensorTest, reserve) {
  tsr_t X;
  X.reserve(3);
}

TEST_F(TensorTest, reserve_view) {
  tsr_t X;
  X.view(Shape(3), VF234.data());
  EXPECT_ANY_THROW(X.reserve(3));
}

TEST_F(TensorTest, clear) {
  tsr_t X(VF234);
  EXPECT_FALSE(X.empty());
  X.clear();
  EXPECT_TRUE(X.empty());
}

TEST_F(TensorTest, clear_view) {
  tsr_t X;
  X.view(Shape(3), VF234.data());
  EXPECT_FALSE(X.empty());
  X.clear();
  EXPECT_TRUE(X.empty());
}

TEST_F(TensorTest, swap) {
  tsr_t X(VF234);
  tsr_t Y(VF567);
  X.swap(Y);
  EXPECT_EQ(X, vectorf_t({5, 6, 7}));
  EXPECT_EQ(Y, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, swap_view) {
  tsr_t X;
  tsr_t Y;
  X.view(Shape(3), VF234.data());
  Y.view(Shape(3), VF567.data());
  X.swap(Y);
  EXPECT_EQ(X, vectorf_t({5, 6, 7}));
  EXPECT_EQ(Y, vectorf_t({2, 3, 4}));
}

TEST_F(TensorTest, sum) {
  tsr_t X(VF2N34);
  EXPECT_EQ(X.sum(), 3);
}

TEST_F(TensorTest, mean) {
  tsr_t X(VF2N34);
  EXPECT_EQ(X.mean(), 1);
}

TEST_F(TensorTest, asum) {
  tsr_t X(VF2N34);
  EXPECT_EQ(X.asum(), 9);
}

TEST_F(TensorTest, amean) {
  tsr_t X(VF2N34);
  EXPECT_EQ(X.amean(), 3);
}

TEST_F(TensorTest, std) {
  tsr_t X(VF2N34);
  EXPECT_DOUBLE_NEAR(X.std(), 2.9439202);
}

TEST_F(TensorTest, var) {
  tsr_t X(VF2N34);
  EXPECT_DOUBLE_NEAR(X.var(), 8.6666666);
}

TEST_F(TensorTest, arange) {
  tsr_t X;
  X.resize(3).arange();
  EXPECT_EQ(X, vectorf_t({0, 1, 2}));
}

TEST_F(TensorTest, constant) {
  tsr_t X;
  X.resize(3).constant(6);
  EXPECT_EQ(X, vectorf_t({6, 6, 6}));
}

TEST_F(TensorTest, zeros) {
  tsr_t X;
  X.resize(3).zeros();
  EXPECT_EQ(X, vectorf_t({0, 0, 0}));
}

TEST_F(TensorTest, ones) {
  tsr_t X;
  X.resize(3).ones();
  EXPECT_EQ(X, vectorf_t({1, 1, 1}));
}

TEST_F(TensorTest, rand) {
  tsr_t X;
  X.resize(10000).rand(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, randn) {
  tsr_t X;
  X.resize(10000).randn(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, rand_lecun) {
  tsr_t X;
  X.resize(100, 100).rand_lecun(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, randn_lecun) {
  tsr_t X;
  X.resize(100, 100).randn_lecun(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, rand_xavier) {
  tsr_t X;
  X.resize(100, 100).rand_xavier(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, randn_xavier) {
  tsr_t X;
  X.resize(100, 100).randn_xavier(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, rand_he) {
  tsr_t X;
  X.resize(100, 100).rand_he(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, randn_he) {
  tsr_t X;
  X.resize(100, 100).randn_he(engine);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, rand_int) {
  tsr_t X;
  X.resize(10000).rand_int(engine, 0, 100);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, rand_init) {
  tsr_t X;
  X.resize(100, 100);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_ZEROS);
  EXPECT_EQ(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_ONES);
  EXPECT_EQ(X.mean(), 1);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_CONSTANT, 6);
  EXPECT_EQ(X.mean(), 6);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RAND, -1, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RAND_LECUN, 0, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RANDN_LECUN, 0, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RAND_XAVIER, 0, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RANDN_XAVIER, 0, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RAND_HE, 0, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RANDN_HE, 0, 1);
  EXPECT_NE(X.mean(), 0);
  X.rand_init(engine, TENSOR_INITIALIZER_TYPE_RAND_INT, 0, 10);
  EXPECT_NE(X.mean(), 0);
}

TEST_F(TensorTest, Compare) {
  tsr_t X(VF234);
  tsr_t Y(VF567);
  EXPECT_TRUE(X != Y);
  EXPECT_FALSE(X == Y);
}

TEST_F(TensorTest, Compare_view) {
  tsr_t X;
  tsr_t Y;
  X.view(Shape(3), VF234.data());
  Y.view(Shape(3), VF567.data());
  EXPECT_TRUE(X != Y);
  EXPECT_FALSE(X == Y);
}

TEST_F(TensorTest, Compare_std_vector) {
  tsr_t X;
  tsr_t Y;
  X.view(Shape(3), VF234.data());
  Y.view(Shape(3), VF567.data());
  EXPECT_TRUE(X != VF567);
  EXPECT_FALSE(X == VF567);
  EXPECT_TRUE(VF567 != X);
  EXPECT_FALSE(VF567 == X);
  EXPECT_TRUE(VF234 != Y);
  EXPECT_FALSE(VF234 == Y);
  EXPECT_TRUE(Y != VF234);
  EXPECT_FALSE(Y == VF234);
}

TEST_F(TensorTest, Compare_std_vector_view) {
  tsr_t X;
  tsr_t Y;
  X.view(Shape(3), VF234.data());
  Y.view(Shape(3), VF567.data());
  EXPECT_TRUE(X != VF567);
  EXPECT_FALSE(X == VF567);
  EXPECT_TRUE(VF567 != X);
  EXPECT_FALSE(VF567 == X);
  EXPECT_TRUE(VF234 != Y);
  EXPECT_FALSE(VF234 == Y);
  EXPECT_TRUE(Y != VF234);
  EXPECT_FALSE(Y == VF234);
}

TEST_F(TensorTest, Compare_std_nullptr_t) {
  tsr_t X(VF234);
  tsr_t Y;
  EXPECT_TRUE(X != nullptr);
  EXPECT_FALSE(X == nullptr);
  EXPECT_FALSE(nullptr != Y);
  EXPECT_TRUE(nullptr == Y);
}

TEST_F(TensorTest, Compare_std_nullptr_t_view) {
  tsr_t X;
  tsr_t Y;
  X.view(Shape(3), VF234.data());
  EXPECT_TRUE(X != nullptr);
  EXPECT_FALSE(X == nullptr);
  EXPECT_FALSE(nullptr != Y);
  EXPECT_TRUE(nullptr == Y);
}

TEST_F(TensorTest, WriteRead) {
  tsr_t X(VF234), read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_X;
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
  EXPECT_FALSE(read_X.is_view());
}

TEST_F(TensorTest, WriteReadView) {
  tsr_t X(VF234), read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_X);
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
  EXPECT_TRUE(read_X.is_view());
}

TEST_F(TensorTest, WriteRead_tsrs_t) {
  tsrs_t X(VS234), read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  is >> read_X;
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
  EXPECT_FALSE(read_X.is_view());
}

TEST_F(TensorTest, WriteReadView_tsrs_t) {
  tsrs_t X(VS234), read_X;

  OutputStringStream os;
  InputStringStream is;

  os << X;
  ASSERT_TRUE(os);

  is.SetView(os.GetBuf());
  ReadView(is, read_X);
  ASSERT_TRUE(is);

  EXPECT_EQ(X, read_X);
  EXPECT_FALSE(read_X.is_view());
}

TEST_F(TensorTest, to_string_rank0_summaryN1) {
  tsr_t X;
  std::string s = to_string(X, -1);
  std::string expected_s = "()\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape2_summaryN1) {
  tsr_t X;
  X.resize(2).arange();
  std::string s = to_string(X, -1);
  std::string expected_s =
      "(2)\n"
      "[0 1]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape23_summaryN1) {
  tsr_t X;
  X.resize(2, 3).arange();
  X.data(0) = (float_t)0.001;
  X.data(4) = (float_t)0.000002;
  std::string s = to_string(X, -1);
  std::string expected_s =
      "(2,3)\n"
      "[[0.001     1     2]\n"
      " [    3 2e-06     5]]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape234_summaryN1) {
  tsr_t X;
  X.resize(2, 3, 4).arange();
  X.data(0) = (float_t)0.001;
  X.data(5) = (float_t)0.000002;
  X.data(10) = (float_t)123.456789;
  X.data(15) = (float_t)123456.789;
  X.data(16) = (float_t)123456789.0;
  X.data(21) = (float_t)123456789000.0;
  std::string s = to_string(X, -1);
  std::string expected_s =
      "(2,3,4)\n"
      "[[[      0.001           1           2           3]\n"
      "  [          4       2e-06           6           7]\n"
      "  [          8           9     123.457          11]]\n"
      " [[         12          13          14      123457]\n"
      "  [1.23457e+08          17          18          19]\n"
      "  [         20 1.23457e+11          22          23]]]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape2345_summaryN1) {
  tsr_t X;
  X.resize(2, 3, 4, 5).arange();
  std::string s = to_string(X, -1);
  std::string expected_s =
      "(2,3,4,5)\n"
      "[[[[  0   1   2   3   4]\n"
      "   [  5   6   7   8   9]\n"
      "   [ 10  11  12  13  14]\n"
      "   [ 15  16  17  18  19]]\n"
      "  [[ 20  21  22  23  24]\n"
      "   [ 25  26  27  28  29]\n"
      "   [ 30  31  32  33  34]\n"
      "   [ 35  36  37  38  39]]\n"
      "  [[ 40  41  42  43  44]\n"
      "   [ 45  46  47  48  49]\n"
      "   [ 50  51  52  53  54]\n"
      "   [ 55  56  57  58  59]]]\n"
      " [[[ 60  61  62  63  64]\n"
      "   [ 65  66  67  68  69]\n"
      "   [ 70  71  72  73  74]\n"
      "   [ 75  76  77  78  79]]\n"
      "  [[ 80  81  82  83  84]\n"
      "   [ 85  86  87  88  89]\n"
      "   [ 90  91  92  93  94]\n"
      "   [ 95  96  97  98  99]]\n"
      "  [[100 101 102 103 104]\n"
      "   [105 106 107 108 109]\n"
      "   [110 111 112 113 114]\n"
      "   [115 116 117 118 119]]]]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape2345_summary0) {
  tsr_t X;
  X.resize(2, 3, 4, 5).arange();
  std::string s = to_string(X, 0);
  std::string expected_s =
      "(2,3,4,5)\n"
      "[ ...\n"
      "]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape2345_summary1) {
  tsr_t X;
  X.resize(2, 3, 4, 5).arange();
  std::string s = to_string(X, 1);
  std::string expected_s =
      "(2,3,4,5)\n"
      "[[[[  0 ...   4]\n"
      "   ...\n"
      "   [ 15 ...  19]]\n"
      "  ...\n"
      "  [[ 40 ...  44]\n"
      "   ...\n"
      "   [ 55 ...  59]]]\n"
      " [[[ 60 ...  64]\n"
      "   ...\n"
      "   [ 75 ...  79]]\n"
      "  ...\n"
      "  [[100 ... 104]\n"
      "   ...\n"
      "   [115 ... 119]]]]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape2345_summary2) {
  tsr_t X;
  X.resize(2, 3, 4, 5).arange();
  std::string s = to_string(X, 2);
  std::string expected_s =
      "(2,3,4,5)\n"
      "[[[[  0   1 ...   3   4]\n"
      "   [  5   6 ...   8   9]\n"
      "   [ 10  11 ...  13  14]\n"
      "   [ 15  16 ...  18  19]]\n"
      "  [[ 20  21 ...  23  24]\n"
      "   [ 25  26 ...  28  29]\n"
      "   [ 30  31 ...  33  34]\n"
      "   [ 35  36 ...  38  39]]\n"
      "  [[ 40  41 ...  43  44]\n"
      "   [ 45  46 ...  48  49]\n"
      "   [ 50  51 ...  53  54]\n"
      "   [ 55  56 ...  58  59]]]\n"
      " [[[ 60  61 ...  63  64]\n"
      "   [ 65  66 ...  68  69]\n"
      "   [ 70  71 ...  73  74]\n"
      "   [ 75  76 ...  78  79]]\n"
      "  [[ 80  81 ...  83  84]\n"
      "   [ 85  86 ...  88  89]\n"
      "   [ 90  91 ...  93  94]\n"
      "   [ 95  96 ...  98  99]]\n"
      "  [[100 101 ... 103 104]\n"
      "   [105 106 ... 108 109]\n"
      "   [110 111 ... 113 114]\n"
      "   [115 116 ... 118 119]]]]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape2345_summary3) {
  tsr_t X;
  X.resize(2, 3, 4, 5).arange();
  std::string s = to_string(X, 3);
  std::string expected_s = to_string(X, -1);
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape888888_summary3) {
  tsr_t X;
  X.resize(88, 88, 88).arange();
  std::string s = to_string(X, 3);
  std::string expected_s =
      "(88,88,88)\n"
      "[[[     0      1      2 ...     85     86     87]\n"
      "  [    88     89     90 ...    173    174    175]\n"
      "  [   176    177    178 ...    261    262    263]\n"
      "  ...\n"
      "  [  7480   7481   7482 ...   7565   7566   7567]\n"
      "  [  7568   7569   7570 ...   7653   7654   7655]\n"
      "  [  7656   7657   7658 ...   7741   7742   7743]]\n"
      " [[  7744   7745   7746 ...   7829   7830   7831]\n"
      "  [  7832   7833   7834 ...   7917   7918   7919]\n"
      "  [  7920   7921   7922 ...   8005   8006   8007]\n"
      "  ...\n"
      "  [ 15224  15225  15226 ...  15309  15310  15311]\n"
      "  [ 15312  15313  15314 ...  15397  15398  15399]\n"
      "  [ 15400  15401  15402 ...  15485  15486  15487]]\n"
      " [[ 15488  15489  15490 ...  15573  15574  15575]\n"
      "  [ 15576  15577  15578 ...  15661  15662  15663]\n"
      "  [ 15664  15665  15666 ...  15749  15750  15751]\n"
      "  ...\n"
      "  [ 22968  22969  22970 ...  23053  23054  23055]\n"
      "  [ 23056  23057  23058 ...  23141  23142  23143]\n"
      "  [ 23144  23145  23146 ...  23229  23230  23231]]\n"
      " ...\n"
      " [[658240 658241 658242 ... 658325 658326 658327]\n"
      "  [658328 658329 658330 ... 658413 658414 658415]\n"
      "  [658416 658417 658418 ... 658501 658502 658503]\n"
      "  ...\n"
      "  [665720 665721 665722 ... 665805 665806 665807]\n"
      "  [665808 665809 665810 ... 665893 665894 665895]\n"
      "  [665896 665897 665898 ... 665981 665982 665983]]\n"
      " [[665984 665985 665986 ... 666069 666070 666071]\n"
      "  [666072 666073 666074 ... 666157 666158 666159]\n"
      "  [666160 666161 666162 ... 666245 666246 666247]\n"
      "  ...\n"
      "  [673464 673465 673466 ... 673549 673550 673551]\n"
      "  [673552 673553 673554 ... 673637 673638 673639]\n"
      "  [673640 673641 673642 ... 673725 673726 673727]]\n"
      " [[673728 673729 673730 ... 673813 673814 673815]\n"
      "  [673816 673817 673818 ... 673901 673902 673903]\n"
      "  [673904 673905 673906 ... 673989 673990 673991]\n"
      "  ...\n"
      "  [681208 681209 681210 ... 681293 681294 681295]\n"
      "  [681296 681297 681298 ... 681381 681382 681383]\n"
      "  [681384 681385 681386 ... 681469 681470 681471]]]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape2_summaryN1_tsr_t) {
  tsrs_t X;
  X.resize(2);
  for (int i = 0; i < X.total_dim(); ++i) {
    X.data(i) = std::to_string(i);
  }
  std::string s = to_string(X, -1);
  std::string expected_s =
      "(2)\n"
      "[\"0\" \"1\"]\n";
  EXPECT_EQ(s, expected_s);
}

TEST_F(TensorTest, to_string_shape23_summaryN1_tsr_t) {
  tsrs_t X;
  X.resize(2, 3);
  for (int i = 0; i < X.total_dim(); ++i) {
    X.data(i) = std::to_string(i);
  }
  std::string s = to_string(X, -1);
  std::string expected_s =
      "(2,3)\n"
      "[[\"0\" \"1\" \"2\"]\n"
      " [\"3\" \"4\" \"5\"]]\n";
  EXPECT_EQ(s, expected_s);
}

}  // namespace deepx_core
