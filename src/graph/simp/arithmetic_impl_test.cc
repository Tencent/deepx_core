// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include "arithmetic_impl.h"
#include <string>
#include "simp_test.h"

namespace deepx_core {

class ArithmeticSimpStageTest : public SimpStageTestBase {
 protected:
  Shape shape;

 protected:
  void SetUp() override {
    simp_name = "arithmetic_simp";
    shape.resize(2, 3);
  }
};

/*
 *             ReduceMean
 *                 |
 *               Add6
 *             /     \
 *          AddN4    Add5                 ReduceMean
 *      /    |    \   |   \                   |
 *   Add1   AddN2   Add3  I7  ->   _________AddN*_____________
 *   /  \   / | \    /  \          | | /    | |    \|    \|  |
 *  I1   I2  I3  I4 I5   I6       I1 I2    I3 I4    I5    I6 I7
 */
TEST_F(ArithmeticSimpStageTest, RewriteGroupedAddStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", shape, TENSOR_TYPE_TSR);
  auto* I5 = new InstanceNode("I5", shape, TENSOR_TYPE_TSR);
  auto* I6 = new InstanceNode("I6", shape, TENSOR_TYPE_TSR);
  auto* I7 = new InstanceNode("I7", shape, TENSOR_TYPE_TSR);
  auto* add1 = Add("Add1", I1, I2);
  auto* add_n2 = AddN("AddN2", {I2, I3, I4});
  auto* add3 = Add("Add3", I5, I6);
  auto* add_n4 = AddN("AddN4", {add1, add_n2, add3});
  auto* add5 = Add("Add5", add3, I7);
  auto* add6 = Add("Add6", add_n4, add5);
  auto* reduce_mean = ReduceMean("ReduceMean", add6, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteGroupedAddStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(9, item.node_size());

  std::string new_add_name = ScopedName("Add6", 2);
  AssertTypeEQ(new_add_name, typeid(AddNNode));

  AssertNodesDeleted({add1, add_n2, add3, add_n4, add5});

  AssertInputsEQ(new_add_name,
                 {I1->name(), I2->name(), I2->name(), I3->name(), I4->name(),
                  I5->name(), I5->name(), I6->name(), I6->name(), I7->name()});
  AssertInputsEQ(reduce_mean->name(), {new_add_name});
}

/*
 *                                                ReduceMean
 *                                                    |
 *              ReduceMean                         *BAdd6
 *                  |                             /       \
 *                BAdd6                      *BAdd3        *BAdd5
 *              /       \                     /  \          /  \
 *        BAdd3           BAdd5    ->    *BAdd2   I3   *BAdd4   I6
 *       /     \         /    \          /   |          /  \
 *   BAdd1     BAdd2   BAdd4   |        /  *BAdd1     I7   I4
 *   /  \      /  \    /  \    |       I2   /  \
 *  I1   I2   I3   I4 I5   I6  I7          I1   I5
 */
TEST_F(ArithmeticSimpStageTest, RewriteGroupedBroadcastStage_BroadcastAdd) {
  auto* I1 = new InstanceNode("I1", Shape(1, 3), TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", Shape(4, 1), TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", Shape(5, 1, 3), TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", Shape(5, 4, 1), TENSOR_TYPE_TSR);
  auto* I5 = new InstanceNode("I5", Shape(4, 3), TENSOR_TYPE_TSR);
  auto* I6 = new InstanceNode("I6", Shape(5, 4, 3), TENSOR_TYPE_TSR);
  auto* I7 = new InstanceNode("I7", Shape(5, 1, 1), TENSOR_TYPE_TSR);
  auto* b_add1 = BroadcastAdd("BAdd1", I1, I2);
  auto* b_add2 = BroadcastAdd("BAdd2", I3, b_add1);
  auto* b_add3 = BroadcastAdd("BAdd3", I4, b_add2);
  auto* b_add4 = BroadcastAdd("BAdd4", I5, b_add3);
  auto* b_add5 = BroadcastAdd("BAdd5", I6, b_add4);
  auto* b_add6 = BroadcastAdd("BAdd6", I7, b_add5);
  auto* reduce_mean = ReduceMean("ReduceMean", b_add6, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteGroupedBroadcastStage(simp_name, &ctx));
  int old_node_size = item.node_size();
  SimplifyTwice();
  ASSERT_EQ(old_node_size, item.node_size());

  std::string new_b_add1_name = ScopedName("BAdd6_0");
  AssertTypeEQ(new_b_add1_name, typeid(BroadcastAddNode));

  std::string new_b_add2_name = ScopedName("BAdd6_1");
  AssertTypeEQ(new_b_add2_name, typeid(BroadcastAddNode));

  std::string new_b_add3_name = ScopedName("BAdd6_2");
  AssertTypeEQ(new_b_add3_name, typeid(BroadcastAddNode));

  std::string new_b_add4_name = ScopedName("BAdd6_3");
  AssertTypeEQ(new_b_add4_name, typeid(BroadcastAddNode));

  std::string new_b_add5_name = ScopedName("BAdd6_4");
  AssertTypeEQ(new_b_add5_name, typeid(BroadcastAddNode));

  std::string new_b_add6_name = ScopedName("BAdd6_5");
  AssertTypeEQ(new_b_add6_name, typeid(BroadcastAddNode));

  AssertNodesDeleted({b_add1, b_add2, b_add3, b_add4, b_add5, b_add6});

  AssertInputsEQ(new_b_add1_name, {I1->name(), I5->name()});
  AssertInputsEQ(new_b_add2_name, {I2->name(), new_b_add1_name});
  AssertInputsEQ(new_b_add3_name, {new_b_add2_name, I3->name()});
  AssertInputsEQ(new_b_add4_name, {I7->name(), I4->name()});
  AssertInputsEQ(new_b_add5_name, {new_b_add4_name, I6->name()});
  AssertInputsEQ(new_b_add6_name, {new_b_add3_name, new_b_add5_name});
  AssertInputsEQ(reduce_mean->name(), {new_b_add6_name});
}

/*
 *                                      ReduceMean
 *                                          |
 *        ReduceMean                      BMul4
 *            |                           /  \
 *          BMul4                   *BMul3    \
 *           /  \          ->        /   \     \
 *      BMul3    \              *BMul2   I3    |
 *      /   \     \              /  \          |
 *   BMul1  BMul2   BAdd       I2   *BMul1   BAdd
 *   /  \   /   \   /  \            /  \     /  \
 *  I1    I2    I3 I4  I5          I1    I2 I4  I5
 */
TEST_F(ArithmeticSimpStageTest,
       RewriteGroupedBroadcastStage_BroadcastMul_mix_BroadcastAdd) {
  auto* I1 = new InstanceNode("I1", Shape(1, 3), TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", Shape(4, 1), TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", Shape(5, 4, 1), TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", Shape(5, 4, 1), TENSOR_TYPE_TSR);
  auto* I5 = new InstanceNode("I5", Shape(4, 3), TENSOR_TYPE_TSR);
  auto* b_mul1 = BroadcastMul("BMul1", I1, I2);
  auto* b_mul2 = BroadcastMul("BMul2", I2, I3);
  auto* b_mul3 = BroadcastMul("BMul3", b_mul1, b_mul2);
  auto* b_add = BroadcastAdd("BAdd", I4, I5);
  auto* b_mul4 = BroadcastMul("BMul4", b_mul3, b_add);
  auto* reduce_mean = ReduceMean("ReduceMean", b_mul4, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteGroupedBroadcastStage(simp_name, &ctx));
  int old_node_size = item.node_size();
  SimplifyTwice();
  ASSERT_EQ(old_node_size, item.node_size());

  std::string new_b_mul1_name = ScopedName("BMul3_0");
  AssertTypeEQ(new_b_mul1_name, typeid(BroadcastMulNode));

  std::string new_b_mul2_name = ScopedName("BMul3_1");
  AssertTypeEQ(new_b_mul2_name, typeid(BroadcastMulNode));

  std::string new_b_mul3_name = ScopedName("BMul3_2");
  AssertTypeEQ(new_b_mul3_name, typeid(BroadcastMulNode));

  AssertNodesDeleted({b_mul1, b_mul2, b_mul3});

  AssertInputsEQ(new_b_mul1_name, {I1->name(), I2->name()});
  AssertInputsEQ(new_b_mul2_name, {I2->name(), new_b_mul1_name});
  AssertInputsEQ(new_b_mul3_name, {new_b_mul2_name, I3->name()});
  AssertInputsEQ(b_mul4->name(), {new_b_mul3_name, b_add->name()});
  AssertInputsEQ(reduce_mean->name(), {b_mul4->name()});
}

/*
 *  ReduceMean1 ReduceMean2 AddN              ReduceMean1 ReduceMean2  AddN
 *     |           |       / | \                 |          |         / / |
 *    Pow5         |   Pow6  |  \                |          | *OnesLike/  |
 *    |  \         |   /  \  |   \               |          |   /     /   |
 *    | OnesLike  Pow4  ZerosLike \      ->      |   *Reciprocal ZerosLike|
 *    |     |    /   \      |      \             |       /         |      |
 *    |    Pow1     C4(-1) Pow2   Pow3           |   *Cubic       Pow2 *Sqrt
 *    |  /     \          /   \   /  \           |  /             /  \  /
 *    I1        C1(3)   X2   C2(mix)  C3(0.5)     I1            X2   C2(mix)
 */
TEST_F(ArithmeticSimpStageTest, RewritePowStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* C1 = Constant("C1", shape, 3);
  auto* C2 = Constant("C2", shape, {2, 1, 2, 2, 2, 2});
  auto* C3 = Constant("C3", shape, {0.5, 0.5, 0.5, 0.5, 0.5, 0.5});
  auto* C4 = Constant("C4", shape, -1);
  auto* pow1 = Pow("Pow1", I1, C1);
  auto* ones_like = OnesLike("OnesLike", pow1);
  auto* pow2 = Pow("Pow2", I2, C2);
  auto* pow3 = Pow("Pow3", C2, C3);
  auto* pow4 = Pow("Pow4", pow1, C4);
  auto* pow5 = Pow("Pow5", I1, ones_like);
  auto* zeros_like = ZerosLike("ZerosLike", pow2);
  auto* pow6 = Pow("Pow6", pow4, zeros_like);
  auto* reduce_mean1 = ReduceMean("ReduceMean1", pow5, 0, 1);
  auto* reduce_mean2 = ReduceMean("ReduceMean2", pow4, 0, 1);
  auto* add_n = AddN("AddN", {pow6, zeros_like, pow3});
  ASSERT_TRUE(graph.Compile({reduce_mean1, reduce_mean2, add_n}, 1));
  item.FromGraph(graph);

  stage.reset(new RewritePowStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(12, item.node_size());

  std::string new_cubic_name = ScopedName("Pow1");
  AssertTypeEQ(new_cubic_name, typeid(CubicNode));

  std::string new_square_name = ScopedName("Pow2");
  ASSERT_EQ(item.find_node(new_square_name), null_node);

  std::string new_sqrt_name = ScopedName("Pow3");
  AssertTypeEQ(new_sqrt_name, typeid(SqrtNode));

  std::string new_reciprocal_name = ScopedName("Pow4");
  AssertTypeEQ(new_reciprocal_name, typeid(InvNode));

  std::string new_ones_like_name = ScopedName("Pow6");
  AssertTypeEQ(new_ones_like_name, typeid(OnesLikeNode));

  AssertNodesDeleted({pow5, pow1, ones_like, C4, pow4, C3, pow6, pow3});

  AssertInputsEQ(reduce_mean1->name(), {I1->name()});
  AssertInputsEQ(reduce_mean2->name(), {new_reciprocal_name});
  AssertInputsEQ(new_reciprocal_name, {new_cubic_name});
  AssertInputsEQ(new_cubic_name, {I1->name()});
  AssertInputsEQ(add_n->name(),
                 {new_ones_like_name, zeros_like->name(), new_sqrt_name});
  AssertInputsEQ(new_ones_like_name, {new_reciprocal_name});
  AssertInputsEQ(zeros_like->name(), {pow2->name()});
  AssertInputsEQ(pow2->name(), {I2->name(), C2->name()});
  AssertInputsEQ(new_sqrt_name, {C2->name()});
}

/*
 *  Sigmoid      Add        ReLU          Sigmoid   Add         ReLU
 *    \      /    |          |               |     /   \          |
 *     ArgMax  ReduceMax   MaxPool1D  ->     |    /    Sqrt      Exp
 *       |        |          |               |   /       |        |
 *      Inv      Sqrt       Exp              |  /    ReduceMax MaxPool1D
 *       |         \        /                | /         |       /
 *    Negate        \    Reciprocal        *ArgMax       |   Reciprocal
 *       |           \   /                   |            \   /
 *       I1            I2                    I1             I2
 */
TEST_F(ArithmeticSimpStageTest, RewriteMaxOrMinOfMonotonicStage) {
  shape.resize(2, 2, 3);
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* negate = Negate("Negate", I1);
  auto* inv = Inv("Inv", negate);
  auto* argmax = ArgMax("ArgMax", inv);
  auto* sigmoid = Sigmoid("Sigmoid", argmax);
  auto* sqrt = Sqrt("Sqrt", I2);
  auto* reduce_max = ReduceMax("ReduceMax", sqrt, 0, 1);
  auto* add = Add("Add", argmax, reduce_max);
  auto* reciprocal = Reciprocal("Reciprocal", I2);
  auto* exp = Exp("Exp", reciprocal);
  auto* max_pool_1d = MaxPool1d("MaxPool1d", exp, 1, 1, 1, 1, 0, 1);
  auto* relu = Relu("Relu", max_pool_1d);
  ASSERT_TRUE(graph.Compile({sigmoid, add, relu}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteMaxOrMinOfMonotonicStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(11, item.node_size());

  std::string new_argmax_name = ScopedName("ArgMax", 2);
  AssertTypeEQ(new_argmax_name, typeid(ArgMaxNode));

  AssertNodesDeleted({negate, inv});

  AssertInputsEQ(sigmoid->name(), {new_argmax_name});
  AssertInputsEQ(new_argmax_name, {I1->name()});
  AssertInputsEQ(add->name(), {new_argmax_name, sqrt->name()});
  AssertInputsEQ(sqrt->name(), {reduce_max->name()});
  AssertInputsEQ(reduce_max->name(), {I2->name()});
  AssertInputsEQ(relu->name(), {exp->name()});
  AssertInputsEQ(exp->name(), {max_pool_1d->name()});
  AssertInputsEQ(max_pool_1d->name(), {reciprocal->name()});
  AssertInputsEQ(reciprocal->name(), {I2->name()});
}

/*
 *  Sigmoid     ReLU                   Sigmoid          ReLU
 *     |         |                        |              |
 *   AddN1     AddN2       ->      *BroadcastMul1      *AddN
 *    / \    /  |||  \                 /    \        /    \  \_________
 *   /  |   /   |||   Add             /      \      /      \          Add
 *   \  |  /   / | \  / \      *Constant1(2)  \    / *BroadcastMul2   / \
 *    \ | /    \ | /  \ /                      \  /     /       \     \ /
 *      I1       I2    I3                       I1    I2 *Constant2(3) I3
 */
TEST_F(ArithmeticSimpStageTest, RewriteAggregatableAddNStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* add_n1 = AddN("AddN1", {I1, I1});
  auto* sigmoid = Sigmoid("Sigmoid", add_n1);
  auto* add = Add("Add", I3, I3);
  auto* addn2 = AddN("AddN2", {I2, I1, I2, I2, add});
  auto* relu = Relu("Relu", addn2);
  ASSERT_TRUE(graph.Compile({sigmoid, relu}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteAggregatableAddNStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(11, item.node_size());

  std::string new_constant1_name = ScopedName("I1_constant");
  AssertTypeEQ(new_constant1_name, typeid(ConstantNode));

  std::string new_broadcastmul1_name = ScopedName("I1_broadcastmul");
  AssertTypeEQ(new_broadcastmul1_name, typeid(BroadcastMulNode));

  std::string new_addn_name = ScopedName("AddN2");
  AssertTypeEQ(new_addn_name, typeid(AddNNode));

  std::string new_constant2_name = ScopedName("I2_constant");
  AssertTypeEQ(new_constant2_name, typeid(ConstantNode));

  std::string new_broadcastmul2_name = ScopedName("I2_broadcastmul");
  AssertTypeEQ(new_broadcastmul2_name, typeid(BroadcastMulNode));

  AssertNodesDeleted({add_n1, addn2});

  AssertInputsEQ(new_broadcastmul1_name, {I1->name(), new_constant1_name});
  AssertInputsEQ(sigmoid->name(), {new_broadcastmul1_name});
  AssertInputsEQ(new_broadcastmul2_name, {I2->name(), new_constant2_name});
  AssertInputsEQ(add->name(), {I3->name(), I3->name()});
  AssertInputsEQ(new_addn_name,
                 {add->name(), I1->name(), new_broadcastmul2_name});
  AssertInputsEQ(relu->name(), {new_addn_name});
}

/*
 *   ReduceMean         ReduceMean
 *       |                  |
 *      Mul3               Mul3
 *     /    \      ->     /    \
 *   Mul1   Mul2       *Square  Mul2
 *   /  \   / \          |      / \
 *  I1  I1 I2  I3        I1    I2  I3
 */
TEST_F(ArithmeticSimpStageTest, RewriteSquareMulStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* mul1 = Mul("Mul1", I1, I1);
  auto* mul2 = Mul("Mul2", I2, I3);
  auto* mul3 = Mul("Mul3", mul1, mul2);
  auto* reduce_mean = ReduceMean("ReduceMean", mul3, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteSquareMulStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(7, item.node_size());

  std::string new_square_name = ScopedName("I1");
  AssertTypeEQ(new_square_name, typeid(SquareNode));

  AssertNodesDeleted({mul1});

  AssertInputsEQ(new_square_name, {I1->name()});
  AssertInputsEQ(mul2->name(), {I2->name(), I3->name()});
  AssertInputsEQ(mul3->name(), {new_square_name, mul2->name()});
}

/*
 *   Sigmoid  AddN                    Sigmoid   AddN
 *      |   /  |   \__                   |    /  |   \
 *      Mul2  Mul4    Mul6 ReLU          |   / *Mul4   Mul6  ReLU
 *    /  |   /   \    |  \ /           *Cubic1/  \     | \   |
 *  Mul1 | Mul3 Square|  Mul5     ->     |   /  *Cubic2|  Mul5
 *  \ | / /  \  /     |   | \            |  /     |    |   / \
 *     I1     I2      I3  I3  I3         I1       I2   I3 I3 I3
 */
TEST_F(ArithmeticSimpStageTest, RewriteCubicMulStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* mul1 = Mul("Mul1", I1, I1);
  auto* mul2 = Mul("Mul2", mul1, I1);
  auto* sigmoid = Sigmoid("Sigmoid", mul2);
  auto* square = Square("Square", I2);
  auto* mul3 = Mul("Mul3", I1, I2);
  auto* mul4 = Mul("Mul4", mul3, square);
  auto* mul5 = Mul("Mul5", I3, I3);
  auto* mul6 = Mul("Mul6", I3, mul5);
  auto* relu = Relu("Relu", mul5);
  auto* add_n = AddN("AddN", {mul2, mul4, mul6});
  ASSERT_TRUE(graph.Compile({sigmoid, add_n, relu}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteCubicMulStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(11, item.node_size());

  std::string new_cubic1_name = ScopedName("I1_cubic");
  AssertTypeEQ(new_cubic1_name, typeid(CubicNode));

  std::string new_cubic2_name = ScopedName("I2_cubic");
  AssertTypeEQ(new_cubic2_name, typeid(CubicNode));

  std::string new_mul4_name = ScopedName("I1_mul");
  AssertTypeEQ(new_mul4_name, typeid(MulNode));

  AssertNodesDeleted({mul1, mul2, mul3, square, mul4});

  AssertInputsEQ(sigmoid->name(), {new_cubic1_name});
  AssertInputsEQ(new_mul4_name, {I1->name(), new_cubic2_name});
  AssertInputsEQ(new_cubic2_name, {I2->name()});
  AssertInputsEQ(add_n->name(), {new_cubic1_name, new_mul4_name, mul6->name()});
  AssertInputsEQ(mul5->name(), {I3->name(), I3->name()});
  AssertInputsEQ(relu->name(), {mul5->name()});
  AssertInputsEQ(mul6->name(), {I3->name(), mul5->name()});
}

/*
 *         Sigmoid                 Sigmoid
 *           |                       |
 *          Mul                     Mul
 *        /    \                  /     \
 *      Add     Sub        ->   *Sub     *Add
 *     /   \    /  \            /  \    /  |
 *  Negate1 |  /  Negate2      |   I1  /   |
 *    |     | /     |           \     /    |
 *   I1     I2      I3             I2      I3
 */
TEST_F(ArithmeticSimpStageTest, RewriteNegateStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* negate1 = Negate("Negate1", I1);
  auto* add = Add("Add", negate1, I2);
  auto* negate2 = Negate("Negate2", I3);
  auto* sub = Sub("Sub", I2, negate2);
  auto* mul = Mul("Mul", add, sub);
  auto* sigmoid = Sigmoid("Sigmoid", mul);
  ASSERT_TRUE(graph.Compile({sigmoid}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteNegateStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(7, item.node_size());

  std::string new_sub_name = ScopedName("Add");
  AssertTypeEQ(new_sub_name, typeid(SubNode));

  std::string new_add_name = ScopedName("Sub");
  AssertTypeEQ(new_add_name, typeid(AddNode));

  AssertNodesDeleted({negate1, negate2, add, sub});

  AssertInputsEQ(sigmoid->name(), {mul->name()});
  AssertInputsEQ(mul->name(), {new_sub_name, new_add_name});
  AssertInputsEQ(new_sub_name, {I2->name(), I1->name()});
  AssertInputsEQ(new_add_name, {I2->name(), I3->name()});
}

/*
 *            Sigmoid               Sigmoid
 *               |                     |
 *              Add                   Add
 *            /      \              /     \
 *          Div      Mul     ->  *Div1  *Div2
 *         /   \     /  \        /  \    / |
 *  Reciprocal Inv1 /  Inv2      |  I1  /  |
 *        |       |      |        \    /   |
 *        I1      I2     I3         I2     I3
 */
TEST_F(ArithmeticSimpStageTest, RewriteInvStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* reciprocal = Reciprocal("Reciprocal", I1);
  auto* inv1 = Inv("Inv1", I2);
  auto* div = Div("Div", reciprocal, inv1);
  auto* inv2 = Inv("Inv2", I3);
  auto* mul = Mul("Mul", I2, inv2);
  auto* add = Add("Add", div, mul);
  auto* sigmoid = Sigmoid("Sigmoid", add);
  ASSERT_TRUE(graph.Compile({sigmoid}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteInvStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(7, item.node_size());

  std::string new_div1_name = ScopedName("Div");
  AssertTypeEQ(new_div1_name, typeid(DivNode));

  std::string new_div2_name = ScopedName("Mul");
  AssertTypeEQ(new_div2_name, typeid(DivNode));

  AssertNodesDeleted({reciprocal, inv1, inv2});

  AssertInputsEQ(sigmoid->name(), {add->name()});
  AssertInputsEQ(add->name(), {new_div1_name, new_div2_name});
  AssertInputsEQ(new_div1_name, {I2->name(), I1->name()});
  AssertInputsEQ(new_div2_name, {I2->name(), I3->name()});
}

/*
 *  Sigmoid  Reshape5      Sigmoid  Reshape5
 *     |        |             |       |
 *  Reshape3 Reshape4      Reshape3   |
 *     |        |             |       |
 *  Reshape2  Relu     ->     |     Relu
 *     |     /                |    /
 *  Reshape1                 Reshape1
 *     |                        |
 *     I                        I
 */
TEST_F(ArithmeticSimpStageTest, RewriteSuccessiveReshapeStage) {
  shape.resize(2, 3, 4);
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* reshape1 = Reshape2("Reshape1", I, Shape(3, 2, 4));
  auto* reshape2 = Reshape2("Reshape2", reshape1, Shape(-1, 6));
  auto* reshape3 = Reshape2("Reshape3", reshape2, Shape(4, 2, 3));
  auto* sigmoid = Sigmoid("Sigmoid", reshape3);
  auto* relu = Relu("Relu", reshape1);
  auto* reshape4 = Reshape2("Reshape4", relu, Shape(2, 4, 3));
  auto* reshape5 = Reshape2("Reshape5", reshape4, Shape(-1, 12));
  ASSERT_TRUE(graph.Compile({sigmoid, reshape5}, 1));
  item.FromGraph(graph);

  stage.reset(new RewriteSuccessiveReshapeStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(6, item.node_size());

  AssertNodesDeleted({reshape2, reshape4});

  AssertInputsEQ(reshape1->name(), {I->name()});
  AssertInputsEQ(reshape3->name(), {reshape1->name()});
  AssertInputsEQ(sigmoid->name(), {reshape3->name()});
  AssertInputsEQ(relu->name(), {reshape1->name()});
  AssertInputsEQ(reshape5->name(), {relu->name()});
}

/*
 *        ReduceMean1 ReduceMean2      ReduceMean1 ReduceMean2
 *            |           |                |           |
 *          Matmul1    Matmul2          *Matmul1   *Matmul2
 *          /   \      /  |        ->    /   \__  __/  |
 *  Transpose1 Transpose2 |             I1      I2     I3
 *      |          |      |
 *      I1         I2     I3
 */
TEST_F(ArithmeticSimpStageTest, FuseTransposeIntoMatmulOrGEMMStage_Matmul) {
  Shape shape1(2, 4, 3);
  Shape shape2(3, 4);
  Shape shape3(4);
  auto* I1 = new InstanceNode("I1", shape1, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape2, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape3, TENSOR_TYPE_TSR);
  auto* transpose1 = Transpose("Transpose1", I1, Shape(0, 2, 1));
  auto* transpose2 = Transpose("Transpose2", I2, Shape(1, 0));
  auto* matmul1 = Matmul("Matmul1", transpose1, transpose2);
  auto* matmul2 = Matmul2("Matmul2", transpose2, I3, 1, 0);
  auto* reduce_mean1 = ReduceMean("ReduceMean1", matmul1, 0, 1);
  auto* reduce_mean2 = ReduceMean("ReduceMean2", matmul2, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean1, reduce_mean2}, 1));

  stage.reset(new FuseTransposeIntoMatmulOrGEMMStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(7, item.node_size());

  std::string new_matmul1_name = ScopedName("Matmul1");
  AssertTypeEQ(new_matmul1_name, typeid(Matmul2Node));
  auto* new_matmul1 = (Matmul2Node*)item.find_node(new_matmul1_name);
  ASSERT_EQ(new_matmul1->transX(), 1);
  ASSERT_EQ(new_matmul1->transY(), 1);

  std::string new_matmul2_name = ScopedName("Matmul2");
  AssertTypeEQ(new_matmul2_name, typeid(Matmul2Node));
  auto* new_matmul2 = (Matmul2Node*)item.find_node(new_matmul2_name);
  ASSERT_EQ(new_matmul2->transX(), 0);
  ASSERT_EQ(new_matmul2->transY(), 0);

  AssertNodesDeleted({transpose1, transpose2});

  AssertInputsEQ(new_matmul1_name, {I1->name(), I2->name()});
  AssertInputsEQ(reduce_mean1->name(), {new_matmul1_name});
  AssertInputsEQ(new_matmul2_name, {I2->name(), I3->name()});
  AssertInputsEQ(reduce_mean2->name(), {new_matmul2_name});
}

/*
 *        ReduceMean1         ReduceMean2             ReduceMean1 ReduceMean2
 *            |                    |                       |           |
 *          GEMM                BatchGEMM               *GEMM      *BatchGEMM
 *          /   \              /         \       ->     /   \       /      \
 *  Transpose1 Transpose2 Transpose3 Transpose4        I1     I2   I3       I4
 *      |          |           |          |
 *      I1         I2          I3         I4
 */
TEST_F(ArithmeticSimpStageTest, FuseTransposeIntoMatmulOrGEMMStage_GEMM) {
  Shape shape1(4, 3);
  Shape shape2(3, 4);
  Shape shape3(2, 4, 3);
  Shape shape4(2, 4, 3);
  auto* I1 = new InstanceNode("I1", shape1, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape2, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape3, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", shape4, TENSOR_TYPE_TSR);
  auto* transpose1 = Transpose("Transpose1", I1, Shape(1, 0));
  auto* transpose2 = Transpose("Transpose2", I2, Shape(1, 0));
  auto* transpose3 = Transpose("Transpose3", I3, Shape(0, 2, 1));
  auto* transpose4 = Transpose("Transpose4", I4, Shape(0, 2, 1));
  auto* gemm = GEMM("GEMM", transpose1, transpose2, 0, 0);
  auto* batch_gemm = BatchGEMM("BatchGEMM", transpose3, transpose4, 1, 0);
  auto* reduce_mean1 = ReduceMean("ReduceMean1", gemm, 0, 1);
  auto* reduce_mean2 = ReduceMean("ReduceMean2", batch_gemm, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean1, reduce_mean2}, 1));

  stage.reset(new FuseTransposeIntoMatmulOrGEMMStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(8, item.node_size());

  std::string new_gemm_name = ScopedName("GEMM");
  AssertTypeEQ(new_gemm_name, typeid(GEMMNode));
  auto* new_gemm = (GEMMNode*)item.find_node(new_gemm_name);
  ASSERT_EQ(new_gemm->transX(), 1);
  ASSERT_EQ(new_gemm->transY(), 1);

  std::string new_batch_gemm_name = ScopedName("BatchGEMM");
  AssertTypeEQ(new_batch_gemm_name, typeid(BatchGEMMNode));
  auto* new_batch_gemm = (BatchGEMMNode*)item.find_node(new_batch_gemm_name);
  ASSERT_EQ(new_batch_gemm->transX(), 0);
  ASSERT_EQ(new_batch_gemm->transY(), 1);

  AssertNodesDeleted({transpose1, transpose2, transpose3, transpose4});

  AssertInputsEQ(new_gemm_name, {I1->name(), I2->name()});
  AssertInputsEQ(reduce_mean1->name(), {new_gemm_name});
  AssertInputsEQ(new_batch_gemm_name, {I3->name(), I4->name()});
  AssertInputsEQ(reduce_mean2->name(), {new_batch_gemm_name});
}

/*
 *  Relu   Sigmoid
 *   |       |                 Sigmoid
 *   |      Inv5                  |
 *  Inv2     |            Relu Reshape
 *   |     Reshape    ->    |     |
 *   |       |              | Transpose
 *  Inv1  Transpose         |    /
 *   |       |              |  Inv3
 *   |      Inv4            |  /
 *   |       |               I
 *   |      Inv3
 *    \    /
 *      I
 */
TEST_F(ArithmeticSimpStageTest, RemoveInvInvolutionStage) {
  shape.resize(2, 3, 4);
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* inv1 = Inv("Inv1", I);
  auto* inv2 = Inv("Inv2", inv1);
  auto* relu = Relu("Relu", inv2);
  auto* inv3 = Inv("Inv3", I);
  auto* inv4 = Inv("Inv4", inv3);
  auto* transpose = Transpose("Transpose", inv4, Shape(2, 1, 0));
  auto* reshape = Reshape2("Reshape", transpose, Shape(4, 6));
  auto* inv5 = Reciprocal("Inv5", reshape);
  auto* sigmoid = Sigmoid("Sigmoid", inv5);
  ASSERT_TRUE(graph.Compile({relu, sigmoid}, 1));

  stage.reset(new RemoveInvInvolutionStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(6, item.node_size());

  AssertNodesDeleted({inv1, inv2, inv4, inv5});

  AssertInputsEQ(relu->name(), {I->name()});
  AssertInputsEQ(inv3->name(), {I->name()});
  AssertInputsEQ(transpose->name(), {inv3->name()});
  AssertInputsEQ(reshape->name(), {transpose->name()});
  AssertInputsEQ(sigmoid->name(), {reshape->name()});
}

/*
 *   Relu     Sigmoid
 *    |         |          Relu  Sigmoid
 *  Negate2   Negate5  ->   |      |
 *    |         |           |   Transpose
 *  Negate1  Transpose      |     /
 *    |         |           |  Negate3
 *    |       Negate4       |  /
 *    |         |            I
 *    |       Negate3
 *     \     /
 *       I
 */
TEST_F(ArithmeticSimpStageTest, RemoveNegateInvolutionStage) {
  shape.resize(2, 3, 4);
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* negate1 = Negate("Negate1", I);
  auto* negate2 = Negate("Negate2", negate1);
  auto* relu = Relu("Relu", negate2);
  auto* negate3 = Negate("Negate3", I);
  auto* negate4 = Negate("Negate4", negate3);
  auto* transpose = Transpose("Transpose", negate4, Shape(2, 1, 0));
  auto* negate5 = Negate("Negate5", transpose);
  auto* sigmoid = Sigmoid("Sigmoid", negate5);
  ASSERT_TRUE(graph.Compile({relu, sigmoid}, 1));

  stage.reset(new RemoveNegateInvolutionStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(5, item.node_size());

  AssertNodesDeleted({negate1, negate2, negate4, negate5});

  AssertInputsEQ(relu->name(), {I->name()});
  AssertInputsEQ(negate3->name(), {I->name()});
  AssertInputsEQ(transpose->name(), {negate3->name()});
  AssertInputsEQ(sigmoid->name(), {transpose->name()});
}

/*
 *   Relu  Sigmoid      Relu  Sigmoid
 *    |      |            |     |
 *  Trans2 Trans4         |   Trans4
 *    |      |      ->    |     |
 *  Trans1 Trans3         |   Trans3
 *    |      |             \   /
 *     \    /                I
 *       I
 */
TEST_F(ArithmeticSimpStageTest, RemoveIneffectiveAdjacentTransposeStage) {
  shape.resize(1, 2, 3, 4);
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* trans1 = Transpose("Trans1", I, Shape(3, 2, 1, 0));
  auto* trans2 = Transpose("Trans2", trans1, Shape(3, 2, 1, 0));
  auto* relu = Relu("Relu", trans2);
  auto* trans3 = Transpose("Trans3", I, Shape(3, 1, 2, 0));
  auto* trans4 = Transpose("Trans4", trans3, Shape(3, 2, 1, 0));
  auto* sigmoid = Sigmoid("Sigmoid", trans4);
  ASSERT_TRUE(graph.Compile({relu, sigmoid}, 1));

  stage.reset(new RemoveIneffectiveAdjacentTransposeStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(5, item.node_size());

  AssertNodesDeleted({trans1, trans2});

  AssertInputsEQ(relu->name(), {I->name()});
  AssertInputsEQ(trans3->name(), {I->name()});
  AssertInputsEQ(trans4->name(), {trans3->name()});
  AssertInputsEQ(sigmoid->name(), {trans4->name()});
}

/*
 *   Relu     Sigmoid
 *    |         |
 *  StopGrad1 Identity3  Add
 *    |         |      /  |       Relu Sigmoid   Add
 *  Identity1 StopGrad2  /    ->      \   \    /  /
 *    |         |       /              \___\  /__/
 *    |       Identity2                     I
 *     \     /
 *        I
 */
TEST_F(ArithmeticSimpStageTest, RemoveIdempotentStage) {
  shape.resize(2, 3, 4);
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* identity1 = Identity("Identity1", I);
  auto* stopgrad1 = StopGrad("StopGrad1", identity1);
  auto* relu = Relu("Relu", stopgrad1);
  auto* identity2 = Identity("Identity2", I);
  auto* stop_grad2 = StopGrad("StopGrad2", identity2);
  auto* identity3 = Identity("Identity3", stop_grad2);
  auto* sigmoid = Sigmoid("Sigmoid", identity3);
  auto* add = Add("Add", stop_grad2, identity2);
  ASSERT_TRUE(graph.Compile({relu, sigmoid, add}, 1));

  stage.reset(new RemoveIdempotentStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(4, item.node_size());

  AssertNodesDeleted({identity1, stopgrad1, identity2, stop_grad2, identity3});

  AssertInputsEQ(relu->name(), {I->name()});
  AssertInputsEQ(sigmoid->name(), {I->name()});
  AssertInputsEQ(add->name(), {I->name(), I->name()});
}

/*
 *                                                     Sigmoid
 *   Relu           Sigmoid                   Relu        |
 *     |               |                       |        *Mul2
 *    Add             AddN                   *Mul1     /     \
 *    / \          /   |   \                /   |   *AddN     \
 *  Mul1 Square  Mul2 Mul3  Mul4    ->    *Add  |  /  |  \     |
 *   /  \___|   / |   | |    | \         /    \ | /   |   \    |
 *  I1      I2 I2 I3  I3I2   I3 I4      I1      I2    I2   I4  I3
 */
TEST_F(ArithmeticSimpStageTest, HoistCommonFactorOutOfAggregationStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", shape, TENSOR_TYPE_TSR);
  auto* mul1 = Mul("Mul1", I1, I2);
  auto* square = Square("Square", I2);
  auto* add = Add("Add", mul1, square);
  auto* relu = Relu("Relu", add);
  auto* mul2 = Mul("Mul2", I2, I3);
  auto* mul3 = Mul("Mul3", I3, I2);
  auto* mul4 = Mul("Mul4", I3, I4);
  auto* addn = AddN("AddN", {mul2, mul3, mul4});
  auto* sigmoid = Sigmoid("Sigmoid", addn);
  ASSERT_TRUE(graph.Compile({relu, sigmoid}, 1));

  stage.reset(new HoistCommonFactorOutOfAggregationStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(10, item.node_size());

  std::string new_mul1_name = ScopedName("Add_mul");
  AssertTypeEQ(new_mul1_name, typeid(MulNode));

  std::string new_add_name = ScopedName("Add_add");
  AssertTypeEQ(new_add_name, typeid(AddNode));

  std::string new_mul2_name = ScopedName("AddN_mul");
  AssertTypeEQ(new_mul2_name, typeid(MulNode));

  std::string new_addn_name = ScopedName("AddN_addn");
  AssertTypeEQ(new_addn_name, typeid(AddNNode));

  AssertNodesDeleted({mul1, square, add, mul2, mul3, mul4, addn});

  AssertInputsEQ(new_add_name, {I1->name(), I2->name()});
  AssertInputsEQ(new_mul1_name, {new_add_name, I2->name()});
  AssertInputsEQ(relu->name(), {new_mul1_name});
  AssertInputsEQ(new_addn_name, {I2->name(), I2->name(), I4->name()});
  AssertInputsEQ(new_mul2_name, {new_addn_name, I3->name()});
  AssertInputsEQ(sigmoid->name(), {new_mul2_name});
}

/*
 *                                                      Sigmoid
 *     Relu           Sigmoid                  Relu         |
 *       |               |                      |         *Div2
 *      Add             AddN                 *Div1      /     \
 *      / \          /   |   \              /   |    *AddN    |
 *    Div1 Div2  Div3  Div4  Div5  ->    *Add   |   /  |  \   |
 *    /  \  | |  /  \  | |  /  \        /    \  |  /   |   I4 |
 *   /    \ | | /    \_| | I4  |       /      \ | /    |      |
 *  I1      I2          I3 - - -      I1        I2    I3 - - -
 */
TEST_F(ArithmeticSimpStageTest, HoistCommonDenominatorOutOfAggregationStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", shape, TENSOR_TYPE_TSR);
  auto* div1 = Div("Div1", I1, I2);
  auto* div2 = Div("Div2", I2, I2);
  auto* add = Add("Add", div1, div2);
  auto* relu = Relu("Relu", add);
  auto* div3 = Div("Div3", I2, I3);
  auto* div4 = Div("Div4", I3, I3);
  auto* div5 = Div("Div5", I4, I3);
  auto* addn = AddN("AddN", {div3, div4, div5});
  auto* sigmoid = Sigmoid("Sigmoid", addn);
  ASSERT_TRUE(graph.Compile({relu, sigmoid}, 1));

  stage.reset(new HoistCommonDenominatorOutOfAggregationStage(simp_name, &ctx));
  item.FromGraph(graph);
  SimplifyTwice();
  ASSERT_EQ(10, item.node_size());

  std::string new_div1_name = ScopedName("Add_div");
  AssertTypeEQ(new_div1_name, typeid(DivNode));

  std::string new_add_name = ScopedName("Add_add");
  AssertTypeEQ(new_add_name, typeid(AddNode));

  std::string new_div2_name = ScopedName("AddN_div");
  AssertTypeEQ(new_div2_name, typeid(DivNode));

  std::string new_addn_name = ScopedName("AddN_addn");
  AssertTypeEQ(new_addn_name, typeid(AddNNode));

  AssertInputsEQ(new_add_name, {I1->name(), I2->name()});
  AssertInputsEQ(new_div1_name, {new_add_name, I2->name()});
  AssertInputsEQ(relu->name(), {new_div1_name});
  AssertInputsEQ(new_addn_name, {I2->name(), I3->name(), I4->name()});
  AssertInputsEQ(new_div2_name, {new_addn_name, I3->name()});
  AssertInputsEQ(sigmoid->name(), {new_div2_name});
}

}  // namespace deepx_core
