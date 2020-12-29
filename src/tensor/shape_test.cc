// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/tensor/shape.h>
#include <gtest/gtest.h>
#include <vector>

namespace deepx_core {

class ShapeTest : public testing::Test {
 protected:
  // This alias is only to make the code more compact.
  const int ANY = SHAPE_DIM_ANY;
};

TEST_F(ShapeTest, Construct_std_il) {
  Shape shape{};
  EXPECT_EQ(shape.rank(), 0);
  EXPECT_EQ(shape.total_dim(), 0);
  EXPECT_TRUE(shape.empty());
  EXPECT_TRUE(shape.same_shape({}));

  shape = {2, 3, 4, 0};
  EXPECT_TRUE(shape.is_rank(4));
  EXPECT_EQ(shape.total_dim(), 0);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
  EXPECT_EQ(shape[3], 0);
  EXPECT_FALSE(shape.empty());
  EXPECT_TRUE(shape.same_shape({2, 3, 4, 0}));

  shape = {2, 3, 4};
  EXPECT_TRUE(shape.is_rank(3));
  EXPECT_EQ(shape.total_dim(), 24);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
  EXPECT_FALSE(shape.empty());
  EXPECT_TRUE(shape.same_shape({2, 3, 4}));
}

TEST_F(ShapeTest, Construct_std_vector) {
  Shape shape{std::vector<int>{}};
  EXPECT_TRUE(shape.same_shape(std::vector<int>{}));

  shape = std::vector<int>{2, 3, 4, 0};
  EXPECT_TRUE(shape.same_shape(std::vector<int>{2, 3, 4, 0}));

  shape = std::vector<int>{2, 3, 4};
  EXPECT_TRUE(shape.same_shape(std::vector<int>{2, 3, 4}));
}

TEST_F(ShapeTest, Construct_ii) {
  std::vector<int> vi{2, 3, 4};
  Shape shape(vi.begin(), vi.end());
  EXPECT_TRUE(shape.same_shape(2, 3, 4));
}

TEST_F(ShapeTest, Construct_int_rank1) {
  Shape shape(2);
  EXPECT_TRUE(shape.same_shape(2));
}

TEST_F(ShapeTest, Construct_int_rank2) {
  Shape shape(2, 3);
  EXPECT_TRUE(shape.same_shape(2, 3));
}

TEST_F(ShapeTest, Construct_int_rank3) {
  Shape shape(2, 3, 4);
  EXPECT_TRUE(shape.same_shape(2, 3, 4));
}

TEST_F(ShapeTest, Construct_va) {
  Shape shape(2, 3, 4, 5);
  EXPECT_TRUE(shape.same_shape(2, 3, 4, 5));
}

TEST_F(ShapeTest, assign) {
  std::vector<int> vi{2, 3, 4};
  Shape shape;
  shape.assign(vi.begin(), vi.end());
  EXPECT_EQ(shape, Shape(2, 3, 4));
}

TEST_F(ShapeTest, clear) {
  Shape shape(2, 3, 4);
  EXPECT_FALSE(shape.empty());
  shape.clear();
  EXPECT_TRUE(shape.empty());
}

TEST_F(ShapeTest, real_axis_1) {
  Shape shape(2, 3, 4);
  int axis;

  axis = 0;
  EXPECT_TRUE(shape.real_axis(&axis));
  EXPECT_EQ(axis, 0);

  axis = 1;
  EXPECT_TRUE(shape.real_axis(&axis));
  EXPECT_EQ(axis, 1);

  axis = 2;
  EXPECT_TRUE(shape.real_axis(&axis));
  EXPECT_EQ(axis, 2);

  axis = 3;
  EXPECT_FALSE(shape.real_axis(&axis));

  axis = -1;
  EXPECT_TRUE(shape.real_axis(&axis));
  EXPECT_EQ(axis, 2);

  axis = -2;
  EXPECT_TRUE(shape.real_axis(&axis));
  EXPECT_EQ(axis, 1);

  axis = -3;
  EXPECT_TRUE(shape.real_axis(&axis));
  EXPECT_EQ(axis, 0);

  axis = -4;
  EXPECT_FALSE(shape.real_axis(&axis));
}

TEST_F(ShapeTest, real_axis_2) {
  Shape shape(2, 3, 4);
  EXPECT_EQ(shape.real_axis(0), 0);
  EXPECT_EQ(shape.real_axis(1), 1);
  EXPECT_EQ(shape.real_axis(2), 2);
  EXPECT_EQ(shape.real_axis(3), SHAPE_INVALID_AXIS);
  EXPECT_EQ(shape.real_axis(-1), 2);
  EXPECT_EQ(shape.real_axis(-2), 1);
  EXPECT_EQ(shape.real_axis(-3), 0);
  EXPECT_EQ(shape.real_axis(-4), SHAPE_INVALID_AXIS);
}

TEST_F(ShapeTest, resize) {
  Shape shape;
  EXPECT_EQ(shape.resize(2), Shape(2));
  EXPECT_EQ(shape.resize(2, 3), Shape(2, 3));
  EXPECT_EQ(shape.resize(2, 3, 4), Shape(2, 3, 4));
  EXPECT_EQ(shape.resize(2, 3, 4, 5), Shape(2, 3, 4, 5));
}

TEST_F(ShapeTest, reshape) {
  Shape shape;
  // 'this' or 'other' is empty
  EXPECT_ANY_THROW(shape.resize().reshape(6));
  EXPECT_ANY_THROW(shape.resize(6).reshape());
  EXPECT_EQ(shape.resize().reshape(), Shape());

  // 'this' or 'other' has two any
  EXPECT_ANY_THROW(shape.resize(ANY, ANY).reshape(6));
  EXPECT_ANY_THROW(shape.resize(6).reshape(ANY, ANY));
  EXPECT_ANY_THROW(shape.resize(ANY, ANY).reshape(ANY, ANY));
  EXPECT_ANY_THROW(shape.resize(6, ANY, ANY).reshape(6, ANY, ANY));

  // without any
  // not match
  EXPECT_ANY_THROW(shape.resize(6).reshape(5));
  EXPECT_ANY_THROW(shape.resize(2, 3).reshape(5));
  EXPECT_ANY_THROW(shape.resize(3, 2).reshape(5));

  // without any
  // match
  EXPECT_EQ(shape.resize(6).reshape(2, 3), Shape(2, 3));
  EXPECT_EQ(shape.resize(6).reshape(3, 2), Shape(3, 2));
  EXPECT_EQ(shape.resize(6).reshape(1, 2, 3), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(2, 3).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(3, 2).reshape(1, 6), Shape(1, 6));

  // with one side any
  // not match
  EXPECT_ANY_THROW(shape.resize(ANY, 3).reshape(5));
  EXPECT_ANY_THROW(shape.resize(ANY, 0).reshape(5));
  EXPECT_ANY_THROW(shape.resize(3, ANY).reshape(5));
  EXPECT_ANY_THROW(shape.resize().reshape(ANY, 3));
  EXPECT_ANY_THROW(shape.resize(5).reshape(ANY, 3));
  EXPECT_ANY_THROW(shape.resize(5).reshape(ANY, 0));
  EXPECT_ANY_THROW(shape.resize(5).reshape(3, ANY));

  // with one side any
  // match
  EXPECT_EQ(shape.resize(ANY, 3).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(2, ANY).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(ANY, 2).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(3, ANY).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(ANY, 2, 3).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(1, ANY, 3).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(1, 2, ANY).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(ANY, 6).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(1, ANY).reshape(6), Shape(6));
  EXPECT_EQ(shape.resize(6).reshape(ANY, 3), Shape(2, 3));
  EXPECT_EQ(shape.resize(6).reshape(2, ANY), Shape(2, 3));
  EXPECT_EQ(shape.resize(6).reshape(ANY, 2), Shape(3, 2));
  EXPECT_EQ(shape.resize(6).reshape(3, ANY), Shape(3, 2));
  EXPECT_EQ(shape.resize(6).reshape(ANY, 2, 3), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(6).reshape(1, ANY, 3), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(6).reshape(1, 2, ANY), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(6).reshape(ANY, 6), Shape(1, 6));
  EXPECT_EQ(shape.resize(6).reshape(1, ANY), Shape(1, 6));

  // with two side any
  EXPECT_EQ(shape.resize(ANY, 6).reshape(5, ANY), Shape(5, ANY));
  EXPECT_EQ(shape.resize(5, ANY).reshape(ANY, 6), Shape(ANY, 6));
}

TEST_F(ShapeTest, reshape_nothrow) {
  // Test cases are the same as the above test.
  Shape shape;
  EXPECT_TRUE(shape.resize().reshape_nothrow(6).empty());
  EXPECT_TRUE(shape.resize(6).reshape_nothrow().empty());
  EXPECT_EQ(shape.resize().reshape_nothrow(), Shape());

  EXPECT_TRUE(shape.resize(ANY, ANY).reshape_nothrow(6).empty());
  EXPECT_TRUE(shape.resize(6).reshape_nothrow(ANY, ANY).empty());
  EXPECT_TRUE(shape.resize(ANY, ANY).reshape_nothrow(ANY, ANY).empty());
  EXPECT_TRUE(shape.resize(6, ANY, ANY).reshape_nothrow(6, ANY, ANY).empty());

  EXPECT_TRUE(shape.resize(6).reshape_nothrow(5).empty());
  EXPECT_TRUE(shape.resize(2, 3).reshape_nothrow(5).empty());
  EXPECT_TRUE(shape.resize(3, 2).reshape_nothrow(5).empty());

  EXPECT_EQ(shape.resize(6).reshape_nothrow(2, 3), Shape(2, 3));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(3, 2), Shape(3, 2));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(1, 2, 3), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(2, 3).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(3, 2).reshape_nothrow(1, 6), Shape(1, 6));

  EXPECT_TRUE(shape.resize(ANY, 3).reshape_nothrow(5).empty());
  EXPECT_TRUE(shape.resize(ANY, 0).reshape_nothrow(5).empty());
  EXPECT_TRUE(shape.resize(3, ANY).reshape_nothrow(5).empty());
  EXPECT_TRUE(shape.resize().reshape_nothrow(ANY, 3).empty());
  EXPECT_TRUE(shape.resize(5).reshape_nothrow(ANY, 3).empty());
  EXPECT_TRUE(shape.resize(5).reshape_nothrow(ANY, 0).empty());
  EXPECT_TRUE(shape.resize(5).reshape_nothrow(3, ANY).empty());

  EXPECT_EQ(shape.resize(ANY, 3).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(2, ANY).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(ANY, 2).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(3, ANY).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(ANY, 2, 3).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(1, ANY, 3).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(1, 2, ANY).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(ANY, 6).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(1, ANY).reshape_nothrow(6), Shape(6));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(ANY, 3), Shape(2, 3));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(2, ANY), Shape(2, 3));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(ANY, 2), Shape(3, 2));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(3, ANY), Shape(3, 2));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(ANY, 2, 3), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(1, ANY, 3), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(1, 2, ANY), Shape(1, 2, 3));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(ANY, 6), Shape(1, 6));
  EXPECT_EQ(shape.resize(6).reshape_nothrow(1, ANY), Shape(1, 6));

  EXPECT_EQ(shape.resize(ANY, 6).reshape_nothrow(5, ANY), Shape(5, ANY));
  EXPECT_EQ(shape.resize(5, ANY).reshape_nothrow(ANY, 6), Shape(ANY, 6));
}

TEST_F(ShapeTest, expand_dim) {
  Shape shape;
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(0), Shape(1, 2, 3, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(1), Shape(2, 1, 3, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(2), Shape(2, 3, 1, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(3), Shape(2, 3, 4, 1));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(-1), Shape(2, 3, 4, 1));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(-2), Shape(2, 3, 1, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(-3), Shape(2, 1, 3, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim(-4), Shape(1, 2, 3, 4));

  EXPECT_ANY_THROW(shape.resize(2, 3, 4, 5, 6, 7, 8, 9).expand_dim(0));
  EXPECT_ANY_THROW(shape.resize(2, 3, 4).expand_dim(4));
  EXPECT_ANY_THROW(shape.resize(2, 3, 4).expand_dim(-5));
}

TEST_F(ShapeTest, expand_dim_nothrow) {
  // Test cases are the same as the above test.
  Shape shape;
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(0), Shape(1, 2, 3, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(1), Shape(2, 1, 3, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(2), Shape(2, 3, 1, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(3), Shape(2, 3, 4, 1));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(-1), Shape(2, 3, 4, 1));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(-2), Shape(2, 3, 1, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(-3), Shape(2, 1, 3, 4));
  EXPECT_EQ(shape.resize(2, 3, 4).expand_dim_nothrow(-4), Shape(1, 2, 3, 4));

  EXPECT_TRUE(
      shape.resize(2, 3, 4, 5, 6, 7, 8, 9).expand_dim_nothrow(0).empty());
  EXPECT_TRUE(shape.resize(2, 3, 4).expand_dim_nothrow(4).empty());
  EXPECT_TRUE(shape.resize(2, 3, 4).expand_dim_nothrow(-5).empty());
}

TEST_F(ShapeTest, squeeze) {
  Shape shape;
  EXPECT_EQ(shape.resize(1, 3, 4).squeeze(0), Shape(3, 4));
  EXPECT_EQ(shape.resize(2, 1, 4).squeeze(1), Shape(2, 4));
  EXPECT_EQ(shape.resize(2, 3, 1).squeeze(2), Shape(2, 3));
  EXPECT_EQ(shape.resize(2, 3, 1).squeeze(-1), Shape(2, 3));
  EXPECT_EQ(shape.resize(2, 1, 4).squeeze(-2), Shape(2, 4));
  EXPECT_EQ(shape.resize(1, 3, 4).squeeze(-3), Shape(3, 4));

  EXPECT_ANY_THROW(shape.resize(1, 1, 1).squeeze(3));
  EXPECT_ANY_THROW(shape.resize(1, 1, 1).squeeze(-4));
  EXPECT_ANY_THROW(shape.resize(2, 1, 1).squeeze(0));
  EXPECT_ANY_THROW(shape.resize(1, 2, 1).squeeze(1));
  EXPECT_ANY_THROW(shape.resize(1, 1, 2).squeeze(2));
}

TEST_F(ShapeTest, squeeze_nothrow) {
  // Test cases are the same as the above test.
  Shape shape;
  EXPECT_EQ(shape.resize(1, 3, 4).squeeze_nothrow(0), Shape(3, 4));
  EXPECT_EQ(shape.resize(2, 1, 4).squeeze_nothrow(1), Shape(2, 4));
  EXPECT_EQ(shape.resize(2, 3, 1).squeeze_nothrow(2), Shape(2, 3));
  EXPECT_EQ(shape.resize(2, 3, 1).squeeze_nothrow(-1), Shape(2, 3));
  EXPECT_EQ(shape.resize(2, 1, 4).squeeze_nothrow(-2), Shape(2, 4));
  EXPECT_EQ(shape.resize(1, 3, 4).squeeze_nothrow(-3), Shape(3, 4));

  EXPECT_TRUE(shape.resize(1, 1, 1).squeeze_nothrow(3).empty());
  EXPECT_TRUE(shape.resize(1, 1, 1).squeeze_nothrow(-4).empty());
  EXPECT_TRUE(shape.resize(2, 1, 1).squeeze_nothrow(0).empty());
  EXPECT_TRUE(shape.resize(1, 2, 1).squeeze_nothrow(1).empty());
  EXPECT_TRUE(shape.resize(1, 1, 2).squeeze_nothrow(2).empty());
}

TEST_F(ShapeTest, iterator) {
  Shape shape(2, 3, 4);
  int total_dim = 1;
  for (int dim : shape) {
    total_dim *= dim;
  }
  EXPECT_EQ(total_dim, shape.total_dim());

  const Shape& cshape = shape;
  total_dim = 1;
  for (int dim : cshape) {
    total_dim *= dim;
  }
  EXPECT_EQ(total_dim, shape.total_dim());
}

TEST_F(ShapeTest, to_string) {
  Shape shape;
  EXPECT_EQ(to_string(shape), "()");
  EXPECT_EQ(to_string(shape.resize(1)), "(1)");
  EXPECT_EQ(to_string(shape.resize(1, 2)), "(1,2)");
  EXPECT_EQ(to_string(shape.resize(1, 2, 3)), "(1,2,3)");
  EXPECT_EQ(to_string(shape.resize(ANY)), "(-1)");
  EXPECT_EQ(to_string(shape.resize(ANY, 2)), "(-1,2)");
  EXPECT_EQ(to_string(shape.resize(ANY, 2, 3)), "(-1,2,3)");
  EXPECT_EQ(to_string(shape.resize(ANY)), "(-1)");
  EXPECT_EQ(to_string(shape.resize(2, ANY)), "(2,-1)");
  EXPECT_EQ(to_string(shape.resize(2, 3, ANY)), "(2,3,-1)");
  EXPECT_EQ(to_string(shape.resize(-2)), "(-2)");
  EXPECT_EQ(to_string(shape.resize(2, -3)), "(2,-3)");
  EXPECT_EQ(to_string(shape.resize(2, 3, -4)), "(2,3,-4)");
}

TEST_F(ShapeTest, Assert) {
  Shape a, b, c, d;
  a.resize(2, 3, 4);
  b.resize(2, 3, 4);
  c.resize(2, 3, 4);
  d.resize(2, 3, 4);
  DXASSERT_SAME_SHAPE(a, b);
  DXASSERT_SAME_SHAPE(a, b, c);
  DXASSERT_SAME_SHAPE(a, b, c, d);
  DXASSERT_SAME_RANK(a, b);
  DXASSERT_RANK(a, 3);
  b.resize(2);
  DXASSERT_RANK1(b);
  c.resize(2, 3);
  DXASSERT_RANK2(c);
  DXASSERT_RANK3(d);
  DXASSERT_SAME_TOTAL_DIM(a, d);
  DXASSERT_TOTAL_DIM(a, 24);
}

}  // namespace deepx_core
