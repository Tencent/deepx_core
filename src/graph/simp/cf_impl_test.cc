// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "cf_impl.h"
#include <deepx_core/tensor/data_type.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include "simp_test.h"

namespace deepx_core {

class GraphFoldingTest : public SimpTestBase {
 protected:
  CFConfig config;
  Shape shape;
  std::unique_ptr<GraphFolding> folding;

 protected:
  void SetUp() override {
    simp_name = "constant_folding";
    config.use_static_shape = 1;
    shape.resize(2, 3);
    folding.reset(new GraphFolding(config));
  }

  void Simplify() override {
    bool simplified;
    do {
      simplified = folding->FoldGraph(&item);
      item.Prune();
    } while (simplified);
  }

  std::string ScopedName(const std::string& name, int rep = 1) const noexcept {
    std::string scope;
    while (rep-- > 0) {
      scope += simp_name + "/" + "fold_graph" + "/";
    }
    return scope + name;
  }
};

/*
 *       Add2                  Add2
 *      /    \                /    \
 *   Add1    Mul   ->  *Constant   Mul
 *   /  \   /  \                  /  \
 *  C1  C2 C3  C4                C3  C4
 */
TEST_F(GraphFoldingTest, GraphFolding) {
  auto* C1 = Constant("C1", shape, 1);
  auto* C2 = Constant("C2", shape, 2);
  auto* C3 = Constant("C3", shape, 3);
  auto* C4 = Constant("C4", shape, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  auto* add1 = Add("Add1", C1, C2);
  auto* mul = Mul("Mul", C3, C4);
  auto* add2 = Add("Add3", add1, mul);
  ASSERT_TRUE(graph.Compile({add2}, 1));
  item.FromGraph(graph);

  SimplifyTwice();
  ASSERT_EQ(5, item.node_size());

  std::string new_constant_name = ScopedName("Add1");
  AssertTypeEQ(new_constant_name, typeid(ConstantNode));

  AssertNodesDeleted({C1, C2, add1});

  AssertInputsEQ(add2->name(), {new_constant_name, mul->name()});
  AssertInputsEQ(mul->name(), {C3->name(), C4->name()});
}

/*
 *  ReduceMean      ReduceMean
 *     |               |
 *    Add       ->    Add
 *   /  \            /  \
 *  C1  C2          C1  C2
 */
TEST_F(GraphFoldingTest, GraphFolding_exceed_max_constant_size) {
  auto* C1 = Constant("C1", shape, 1);
  auto* C2 = Constant("C2", shape, 2);
  auto* add = Add("Add", C1, C2);
  auto* reduce_mean = ReduceMean("ReduceMean", add, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean}, 1));
  item.FromGraph(graph);

  config.use_static_shape = 1;
  config.max_constant_bytes =
      (int)sizeof(deepx_core::DataType::float_t) * shape.total_dim() - 1;
  folding.reset(new GraphFolding(config));

  SimplifyTwice();
  ASSERT_EQ(4, item.node_size());

  AssertInputsEQ(add->name(), {C1->name(), C2->name()});
  AssertInputsEQ(reduce_mean->name(), {add->name()});
}

class ConstantFoldingStageTest : public SimpStageTestBase {
 protected:
  CFConfig config;
  Shape shape;

 protected:
  void SetUp() override {
    simp_name = "constant_folding";
    config.use_static_shape = 1;
    shape.resize(2, 3);
  }
};

/*
 *            Add              Add
 *           /  \              | \
 *  Transpose1 Transpose2  ->  |  Transpose2
 *          \   /              |  /
 *            I                 I
 */
TEST_F(ConstantFoldingStageTest, RemoveIdenticalTransposeStage) {
  shape.reshape(1, 2, 3);
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* transpose1 = Transpose("Transpose1", I, Shape(0, 1, 2));
  auto* transpose2 = Transpose("Transpose2", I, Shape(0, 2, 1));
  auto* add = Add("Add", transpose1, transpose2);
  ASSERT_TRUE(graph.Compile({add}, 1));
  item.FromGraph(graph);

  stage.reset(new RemoveIdenticalTransposeStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(3, item.node_size());

  AssertNodesDeleted({transpose1});

  AssertInputsEQ(add->name(), {I->name(), transpose2->name()});
  AssertInputsEQ(transpose2->name(), {I->name()});
}

/*
 *      Add          Add
 *      /  \         | \
 *  Tile1 Tile2  ->  |  Tile2
 *      \  /         | /
 *        I           I
 */
TEST_F(ConstantFoldingStageTest, RemoveIdenticalTileStage) {
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* tile1 = Tile("Tile1", I, {1, 1});
  auto* tile2 = Tile("Tile2", I, {1, 2});
  auto* add = Add("Add", tile1, tile2);
  ASSERT_TRUE(graph.Compile({add}, 1));
  item.FromGraph(graph);

  stage.reset(new RemoveIdenticalTileStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(3, item.node_size());

  AssertNodesDeleted({tile1});

  AssertInputsEQ(add->name(), {I->name(), tile2->name()});
  AssertInputsEQ(tile2->name(), {I->name()});
}

/*
 *                Add                    Add
 *                /  \                   | \
 *  SubscriptRange1 SubscriptRange2  ->  |  SubscriptRange2
 *                \  /                   | /
 *                 I                      I
 */
TEST_F(ConstantFoldingStageTest, RemoveIdenticalSubscriptRangeStage) {
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* subscript_range1 = SubscriptRange("SubscriptRange1", I, 1, 0, 3);
  auto* subscript_range2 = SubscriptRange("SubscriptRange2", I, 1, 0, 2);
  auto* add = Add("Add", subscript_range1, subscript_range2);
  ASSERT_TRUE(graph.Compile({add}, 1));
  item.FromGraph(graph);

  stage.reset(new RemoveIdenticalSubscriptRangeStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(3, item.node_size());

  AssertNodesDeleted({subscript_range1});

  AssertInputsEQ(add->name(), {I->name(), subscript_range2->name()});
  AssertInputsEQ(subscript_range2->name(), {I->name()});
}

/*
 *  ReduceMean1 ReduceMean2      ReduceMean1 ReduceMean2
 *       |           |                |           |
 *   Reshape1    Reshape2    ->       |       Reshape2
 *           \  /                     | __________|
 *             I                       I
 */
TEST_F(ConstantFoldingStageTest, RemoveIdenticalReshapeStage) {
  auto* I = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* reshape1 = Reshape("Reshape1", I, Shape(-1, 3));
  auto* reshape2 = Reshape("Reshape2", I, Shape(-1, 2));
  auto* reduce_mean1 = ReduceMean("ReduceMean1", reshape1, 0, 1);
  auto* reduce_mean2 = ReduceMean("ReduceMean2", reshape2, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean1, reduce_mean2}, 1));
  item.FromGraph(graph);

  stage.reset(new RemoveIdenticalReshapeStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(4, item.node_size());

  AssertNodesDeleted({reshape1});

  AssertInputsEQ(reduce_mean1->name(), {I->name()});
  AssertInputsEQ(reshape2->name(), {I->name()});
}

/*    Add                      Add
 *   /    \                  /     \
 *  Div1  Div2          ->  *Mul1  Div2
 *  | \    | \              | \     | \
 *  I1 C1  I2 OnesLike      I1 *C  I2  OnesLike
 *               |                        |
 *               I3                       I3
 */
TEST_F(ConstantFoldingStageTest, DivToReciprocalMulFoldingStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* C1 = Constant("C1", shape, 1);
  auto* div1 = Div("Div1", I1, C1);
  auto* ones_like = OnesLike("OnesLike", I3);
  auto* div2 = Div("Div2", I2, ones_like);
  auto* add = Add("Add", div1, div2);
  ASSERT_TRUE(graph.Compile({add}, 1));
  item.FromGraph(graph);

  stage.reset(new DivToReciprocalMulFoldingStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(8, item.node_size());

  std::string new_constant_name = ScopedName("C1");
  AssertTypeEQ(new_constant_name, typeid(ConstantNode));

  std::string new_mul_name = ScopedName("Div1");
  AssertTypeEQ(new_mul_name, typeid(MulNode));

  AssertNodesDeleted({C1, div1});

  AssertInputsEQ(new_mul_name, {I1->name(), new_constant_name});
  AssertInputsEQ(ones_like->name(), {I3->name()});
  AssertInputsEQ(div2->name(), {I2->name(), ones_like->name()});
  AssertInputsEQ(add->name(), {new_mul_name, div2->name()});
}

/*
 *            _______Add_______                      Add
 *            |               |                   /       \
 *  ________AddN1_____      AddN2            *AddN         AddN2
 *  |  |   /    |     |    /  |  \   ->    /   |   \      /  |  \
 *  I1 C1 C2 OnesLike C3  I3  I4 C4      I1 OnesLike *C  I3  I4 C4
 *              |                              |
 *              I2                             I2
 */
TEST_F(ConstantFoldingStageTest, PartialAddNFoldingStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", shape, TENSOR_TYPE_TSR);
  auto* C1 = Constant("C1", shape, 1);
  auto* C2 = Constant("C2", shape, 2);
  auto* C3 = Constant("C3", shape, 3);
  auto* C4 = Constant("C4", shape, 4);
  auto* ones_like = OnesLike("OnesLike", I2);
  auto* addn1 = AddN("AddN1", {I1, C1, C2, ones_like, C3});
  auto* addn2 = AddN("AddN2", {I3, I4, C4});
  auto* add = Add("Add", addn1, addn2);
  ASSERT_TRUE(graph.Compile({add}, 1));
  item.FromGraph(graph);

  stage.reset(new PartialAddNFoldingStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(10, item.node_size());

  std::string new_constant_name = ScopedName("AddN1_new_constant");
  AssertTypeEQ(new_constant_name, typeid(ConstantNode));

  std::string new_addn_name = ScopedName("AddN1_new_addn");
  AssertTypeEQ(new_addn_name, typeid(AddNNode));

  AssertNodesDeleted({C1, C2, C3});

  AssertInputsEQ(ones_like->name(), {I2->name()});
  AssertInputsEQ(new_addn_name,
                 {I1->name(), ones_like->name(), new_constant_name});
  AssertInputsEQ(addn2->name(), {I3->name(), I4->name(), C4->name()});
}

/*
 *             ReduceMean                     ReduceMean
 *                |                              |
 *   ___________Concat___________      _______*Concat_______
 *   |  |  |  |  |  |  |  |  |  |  ->  |  |  |   |  |  |   |
 *   C1 I1 C2 C3 I2 C4 I3 C5 C6 C7     C1 I1 *C1 I2 C4 I3 *C2
 */
TEST_F(ConstantFoldingStageTest, PartialConcatFoldingStage) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* C1 = Constant("C1", shape, 1);
  auto* C2 = Constant("C2", shape, 2);
  auto* C3 = Constant("C3", shape, 3);
  auto* C4 = Constant("C4", shape, 4);
  auto* C5 = Constant("C5", shape, 5);
  auto* C6 = Constant("C6", shape, 6);
  auto* C7 = Constant("C7", shape, 7);
  auto* concat = Concat("Concat", {C1, I1, C2, C3, I2, C4, I3, C5, C6, C7}, 0);
  auto* reduce_mean = ReduceMean("ReduceMean", concat, 0, 1);
  ASSERT_TRUE(graph.Compile({reduce_mean}, 1));
  item.FromGraph(graph);

  stage.reset(new PartialConcatFoldingStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(9, item.node_size());

  std::string new_C1_name = ScopedName("Concat_new_constant_1");
  AssertTypeEQ(new_C1_name, typeid(ConstantNode));

  std::string new_C2_name = ScopedName("Concat_new_constant_2");
  AssertTypeEQ(new_C2_name, typeid(ConstantNode));

  std::string new_concat_name = ScopedName("Concat_new_concat");
  AssertTypeEQ(new_concat_name, typeid(ConcatNode));

  AssertNodesDeleted({C2, C3, C5, C6, C7});

  AssertInputsEQ(new_concat_name,
                 {C1->name(), I1->name(), new_C1_name, I2->name(), C4->name(),
                  I3->name(), new_C2_name});
  AssertInputsEQ(reduce_mean->name(), {new_concat_name});
}

/*
 *              ________AddN____                  ________AddN______
 *              |   |     |    |                  |   |     |      |
 *            Add2 Sub2 Mul2 Div2                 | *Negate |  *ZerosLike2
 *           /  | / |   / |  / |                  |   |     |      |
 *       C(0)---|---|-----|-   |                  |   |     |      |
 *            Add1 Sub1 Mul1 Div1  ->           Add1 Sub1   |     *Inv
 *           /  | / |   / |  / |               /  | / | *ZerosLike1|
 *   OnesLike---|---|-----|-   |       OnesLike---|-  |     |      |
 *      |       |___|_____|____|          |       |___|_____|______|
 *      I1      I2                        I1      I2
 */
TEST_F(ConstantFoldingStageTest, ArithmeticOperationsFoldingStage1) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* ones_like = OnesLike("OnesLike", I1);
  auto* add1 = Add("Add1", ones_like, I2);
  auto* sub1 = Sub("Sub1", ones_like, I2);
  auto* mul1 = Mul("Mul1", ones_like, I2);
  auto* div1 = Div("Div1", ones_like, I2);
  auto* C = Constant("C", shape, std::vector<double>(shape.total_dim(), 0));
  auto* add2 = Add("Add2", C, add1);
  auto* sub2 = Sub("Sub2", C, sub1);
  auto* mul2 = Mul("Mul2", C, mul1);
  auto* div2 = Div("Div2", C, div1);
  auto* add_n = AddN("AddN", {add2, sub2, mul2, div2});
  ASSERT_TRUE(graph.Compile({add_n}, 1));
  item.FromGraph(graph);

  stage.reset(new ArithmeticOperationsFoldingStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(10, item.node_size());

  std::string new_zeros_like1_name = ScopedName("Mul1_ZerosLike");
  AssertTypeEQ(new_zeros_like1_name, typeid(ZerosLikeNode));

  std::string new_inv_name = ScopedName("I2_Inv");
  AssertTypeEQ(new_inv_name, typeid(InvNode));

  std::string new_negate_name = ScopedName("Sub1_Negate");
  AssertTypeEQ(new_negate_name, typeid(NegateNode));

  std::string new_zeros_like2_name = ScopedName("Div1_ZerosLike");
  AssertTypeEQ(new_zeros_like2_name, typeid(ZerosLikeNode));

  AssertNodesDeleted({mul1, div1, C, add2, sub2, mul2, div2});

  AssertInputsEQ(ones_like->name(), {I1->name()});
  AssertInputsEQ(add1->name(), {ones_like->name(), I2->name()});
  AssertInputsEQ(sub1->name(), {ones_like->name(), I2->name()});
  AssertInputsEQ(new_zeros_like1_name, {I2->name()});
  AssertInputsEQ(new_inv_name, {I2->name()});
  AssertInputsEQ(new_negate_name, {sub1->name()});
  AssertInputsEQ(new_zeros_like2_name, {new_inv_name});
  AssertInputsEQ(add_n->name(), {add1->name(), new_negate_name,
                                 new_zeros_like1_name, new_zeros_like2_name});
}

/*
 *   _______AddN______                  _______AddN_______
 *   |    |     |    |                  |    |     |     |
 *   Add2 Sub2  Mul2 |                  |    |*ZerosLike |
 *   | \  | \   | \  |                  |    |     |     |
 *   |  --|-----|----|---ZerosLike      |    |     |     |
 *   Add1 Sub1 Mul1 Div1     |      ->  Add1 Sub1  |     |
 *   | \  | \   | \  | \     |          | \  | \   |     |
 *   |  --|-----|----|------C(1)        |  --|-----|-----|------C(1)
 *   |____|_____|____|                  |____|_____|_____|
 *   I1                                 I1
 */
TEST_F(ConstantFoldingStageTest, ArithmeticOperationsFoldingStage2) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* C = Constant("C", shape, 1);
  auto* add1 = Add("Add1", I1, C);
  auto* sub1 = Sub("Sub1", I1, C);
  auto* mul1 = Mul("Mul1", I1, C);
  auto* div1 = Div("Div1", I1, C);
  auto* zeros_like = ZerosLike("ZerosLike", C);
  auto* add2 = Add("Add2", add1, zeros_like);
  auto* sub2 = Sub("Sub2", sub1, zeros_like);
  auto* mul2 = Mul("Mul2", mul1, zeros_like);
  auto* add_n = AddN("AddN", {add2, sub2, mul2, div1});
  ASSERT_TRUE(graph.Compile({add_n}, 1));
  item.FromGraph(graph);

  stage.reset(new ArithmeticOperationsFoldingStage(simp_name, &ctx));
  SimplifyTwice();
  ASSERT_EQ(6, item.node_size());

  std::string new_zeros_like_name = ScopedName("Mul1_ZerosLike");
  AssertTypeEQ(new_zeros_like_name, typeid(ZerosLikeNode));

  AssertNodesDeleted({mul1, div1, add2, sub2, mul2});

  AssertInputsEQ(add1->name(), {I1->name(), C->name()});
  AssertInputsEQ(sub1->name(), {I1->name(), C->name()});
  AssertInputsEQ(new_zeros_like_name, {I1->name()});
  AssertInputsEQ(add_n->name(),
                 {add1->name(), sub1->name(), new_zeros_like_name, I1->name()});
}

/*
 *        Add3                  Add3
 *       /    \               /     \
 *   Add2      Mul2       Add2       Mul2
 *   / \       / \        / \        / |
 *  C1 Add1 Mul1 C2  ->  I1 Add1  Mul1 |
 *     / \  /  \            / \   /  \ |
 *    I1  I2   I3          C1  | C2  I3|
 *                             |________|
 *                             I2
 */
TEST_F(ConstantFoldingStageTest, ConstantPushDownStage1) {
  auto* C1 = Constant("C1", shape, 1);
  auto* C2 = Constant("C2", shape, 2);
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* add1 = Add("Add1", I1, I2);
  auto* add2 = Add("Add2", C1, add1);
  auto* mul1 = Mul("Mul1", I2, I3);
  auto* mul2 = Mul("Mul2", mul1, C2);
  auto* add3 = Add("Add3", add2, mul2);
  ASSERT_TRUE(graph.Compile({add3}, 1));
  item.FromGraph(graph);

  stage.reset(new ConstantPushDownStage(simp_name, &ctx));
  int old_node_size = item.node_size();
  SimplifyTwice();
  ASSERT_EQ(old_node_size, item.node_size());

  AssertInputsEQ(add1->name(), {C1->name(), I2->name()});
  AssertInputsEQ(add2->name(), {I1->name(), add1->name()});
  AssertInputsEQ(mul1->name(), {C2->name(), I3->name()});
  AssertInputsEQ(mul2->name(), {mul1->name(), I2->name()});
  AssertInputsEQ(add3->name(), {add2->name(), mul2->name()});
}

/*
 *                               AddN
 *    ____________________________|____________________________
 *    |       |       |       |       |       |       |       |
 *   Add1    Add2    Add4    Add6    Sub3    Sub4   Sub6    Sub8
 *   / \     / \     / \     / \     / \     / \     / \     / \
 *  C1 Sub1 C2 Sub2 C4 Add3 C5 Add5 C1 Add7 C2 Add8 C4 Sub5 C5 Sub7
 *     / |     / |     / |     / |     / |     / |     / |     / |
 *    I1 I2  C3  I3   I4 I5  C6  I6   I1 I2  C3  I3   I4 I5  C6  I6
 *  ->
 *                               AddN
 *     ___________________________|________________________________
 *     |          |     |      |          |         |       |     |
 *   *Add1      *Sub2  Add4    Add6     *Sub4     *Sub6   *Sub7  *Add10
 *   / \         / \   / \     / \        / \      / \     / \    / \
 *  I1 *Sub1 *Add2 I3 I4 Add3 I6 Add5 *Sub3 I1 *Sub5 I3 *Add9 I4 I6 *Sub8
 *     / |   / |        / |     / |   / |       / |     / |         / |
 *    C1 I2 C3 C2      C4 I5   C6 C5 C1 I2     C2 C3   C4 I5       C5 C6
 */
TEST_F(ConstantFoldingStageTest, ConstantPushDownStage2) {
  auto* C1 = Constant("C1", shape, 1);
  auto* C2 = Constant("C2", shape, 2);
  auto* C3 = Constant("C3", shape, 3);
  auto* C4 = Constant("C4", shape, 4);
  auto* C5 = Constant("C5", shape, 5);
  auto* C6 = Constant("C6", shape, 6);
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", shape, TENSOR_TYPE_TSR);
  auto* I5 = new InstanceNode("I5", shape, TENSOR_TYPE_TSR);
  auto* I6 = new InstanceNode("I6", shape, TENSOR_TYPE_TSR);
  auto* sub1 = Sub("Sub1", I1, I2);
  auto* add1 = Add("Add1", C1, sub1);
  auto* sub2 = Sub("Sub2", C3, I3);
  auto* add2 = Add("Add2", C2, sub2);
  auto* add3 = Add("Add3", I4, I5);
  auto* add4 = Add("Add4", C4, add3);
  auto* add5 = Add("Add5", C6, I6);
  auto* add6 = Add("Add6", C5, add5);
  auto* add7 = Add("Add7", I1, I2);
  auto* sub3 = Sub("Sub3", C1, add7);
  auto* add8 = Add("Add8", C3, I3);
  auto* sub4 = Sub("Sub4", C2, add8);
  auto* sub5 = Sub("Sub5", I4, I5);
  auto* sub6 = Sub("Sub6", C4, sub5);
  auto* sub7 = Sub("Sub7", C6, I6);
  auto* sub8 = Sub("Sub8", C5, sub7);
  auto* addn = AddN("AddN", {add1, add2, add4, add6, sub3, sub4, sub6, sub8});
  ASSERT_TRUE(graph.Compile({addn}, 1));
  item.FromGraph(graph);

  stage.reset(new ConstantPushDownStage(simp_name, &ctx));
  int old_node_size = item.node_size();
  SimplifyTwice();
  ASSERT_EQ(old_node_size, item.node_size());

  std::string new_add1_name = ScopedName("Add1");
  AssertTypeEQ(new_add1_name, typeid(AddNode));

  std::string new_sub1_name = ScopedName("Sub1");
  AssertTypeEQ(new_sub1_name, typeid(SubNode));

  std::string new_add2_name = ScopedName("Sub2");
  AssertTypeEQ(new_add2_name, typeid(AddNode));

  std::string new_sub2_name = ScopedName("Add2");
  AssertTypeEQ(new_sub2_name, typeid(SubNode));

  std::string new_sub3_name = ScopedName("Add7");
  AssertTypeEQ(new_sub3_name, typeid(SubNode));

  std::string new_sub4_name = ScopedName("Sub3");
  AssertTypeEQ(new_sub4_name, typeid(SubNode));

  std::string new_sub5_name = ScopedName("Add8");
  AssertTypeEQ(new_sub5_name, typeid(SubNode));

  std::string new_sub6_name = ScopedName("Sub4");
  AssertTypeEQ(new_sub6_name, typeid(SubNode));

  std::string new_add9_name = ScopedName("Sub5");
  AssertTypeEQ(new_add9_name, typeid(AddNode));

  std::string new_sub7_name = ScopedName("Sub6");
  AssertTypeEQ(new_sub7_name, typeid(SubNode));

  std::string new_sub8_name = ScopedName("Sub7");
  AssertTypeEQ(new_sub8_name, typeid(SubNode));

  std::string new_add10_name = ScopedName("Sub8");
  AssertTypeEQ(new_add10_name, typeid(AddNode));

  AssertNodesDeleted(
      {add1, sub1, add2, sub2, sub3, add7, sub4, add8, sub5, sub6, sub7, sub8});

  AssertInputsEQ(new_sub1_name, {C1->name(), I2->name()});
  AssertInputsEQ(new_add1_name, {I1->name(), new_sub1_name});
  AssertInputsEQ(new_add2_name, {C2->name(), C3->name()});
  AssertInputsEQ(new_sub2_name, {new_add2_name, I3->name()});
  AssertInputsEQ(add3->name(), {C4->name(), I5->name()});
  AssertInputsEQ(add4->name(), {I4->name(), add3->name()});
  AssertInputsEQ(add5->name(), {C6->name(), C5->name()});
  AssertInputsEQ(add6->name(), {I6->name(), add5->name()});
  AssertInputsEQ(new_sub3_name, {C1->name(), I2->name()});
  AssertInputsEQ(new_sub4_name, {new_sub3_name, I1->name()});
  AssertInputsEQ(new_sub5_name, {C2->name(), C3->name()});
  AssertInputsEQ(new_sub6_name, {new_sub5_name, I3->name()});
  AssertInputsEQ(new_add9_name, {C4->name(), I5->name()});
  AssertInputsEQ(new_sub7_name, {new_add9_name, I4->name()});
  AssertInputsEQ(new_sub8_name, {C5->name(), C6->name()});
  AssertInputsEQ(new_add10_name, {I6->name(), new_sub8_name});
  AssertInputsEQ(addn->name(),
                 {new_add1_name, new_sub2_name, add4->name(), add6->name(),
                  new_sub4_name, new_sub6_name, new_sub7_name, new_add10_name});
}

/*
 *                                AddN
 *     ____________________________|____________________________
 *     |       |       |       |       |       |       |       |
 *    Mul1    Mul2    Mul4    Mul6    Div3    Div4   Div6    Div8
 *    / \     / \     / \     / \     / \     / \     / \     / \
 *  Div1 C1 Div2 C2 Mul3 C4 Mul5 C5 Mul7 C1 Mul8 C2 Div5 C4 Div7 C5
 *  | \     | \     | \     | \     | \     | \     | \     | \
 *  I1 I2   C3 I3   I4 I5   C6 I6   I1 I2   C3 I3   I4 I5   C6 I6
 *  ->
 *                                AddN
 *     ____________________________|_____________________________________
 *     |         |       |       |       |       |        |             |
 *    *Mul1    *Div2    Mul4    Mul6    *Mul7   *Mul8    *Div9       *Div11
 *    / \       / \     / \     / \     / \     / \      / \           / \
 *  I1 *Div1 *Mul2 I3 Mul3 I4 Mul5 I6 I1 *Div7 I3 *Div8 I4 *Mul9  *Div10 I6
 *     | \    | \     | \     | \         | \      | \      | \    | \
 *     C1 I2  C3 C2   C4 I5   C6 C5       I2 C1    C3 C2    C4 I5  C6 C5
 */
TEST_F(ConstantFoldingStageTest, ConstantPushDownStage3) {
  auto* C1 = Constant("C1", shape, 1);
  auto* C2 = Constant("C2", shape, 2);
  auto* C3 = Constant("C3", shape, 3);
  auto* C4 = Constant("C4", shape, 4);
  auto* C5 = Constant("C5", shape, 5);
  auto* C6 = Constant("C6", shape, 6);
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I4", shape, TENSOR_TYPE_TSR);
  auto* I5 = new InstanceNode("I5", shape, TENSOR_TYPE_TSR);
  auto* I6 = new InstanceNode("I6", shape, TENSOR_TYPE_TSR);
  auto* div1 = Div("Div1", I1, I2);
  auto* mul1 = Mul("Mul1", C1, div1);
  auto* div2 = Div("Div2", C3, I3);
  auto* mul2 = Mul("Mul2", C2, div2);
  auto* mul3 = Mul("Mul3", I4, I5);
  auto* mul4 = Mul("Mul4", C4, mul3);
  auto* mul5 = Mul("Mul5", C6, I6);
  auto* mul6 = Mul("Mul6", C5, mul5);
  auto* mul7 = Mul("Mul7", I1, I2);
  auto* div3 = Div("Div3", mul7, C1);
  auto* mul8 = Mul("Mul8", C3, I3);
  auto* div4 = Div("Div4", mul8, C2);
  auto* div5 = Div("Div5", I4, I5);
  auto* div6 = Div("Div6", div5, C4);
  auto* div7 = Div("Div7", C6, I6);
  auto* div8 = Div("Div8", div7, C5);
  auto* addn = AddN("AddN", {mul1, mul2, mul4, mul6, div3, div4, div6, div8});
  ASSERT_TRUE(graph.Compile({addn}, 1));
  item.FromGraph(graph);

  stage.reset(new ConstantPushDownStage(simp_name, &ctx));
  int old_node_size = item.node_size();
  SimplifyTwice();
  ASSERT_EQ(old_node_size, item.node_size());

  std::string new_mul1_name = ScopedName("Mul1");
  AssertTypeEQ(new_mul1_name, typeid(MulNode));

  std::string new_div1_name = ScopedName("Div1");
  AssertTypeEQ(new_div1_name, typeid(DivNode));

  std::string new_mul2_name = ScopedName("Div2");
  AssertTypeEQ(new_mul2_name, typeid(MulNode));

  std::string new_div2_name = ScopedName("Mul2");
  AssertTypeEQ(new_div2_name, typeid(DivNode));

  std::string new_div7_name = ScopedName("Mul7");
  AssertTypeEQ(new_div7_name, typeid(DivNode));

  std::string new_mul7_name = ScopedName("Div3");
  AssertTypeEQ(new_mul7_name, typeid(MulNode));

  std::string new_div8_name = ScopedName("Mul8");
  AssertTypeEQ(new_div8_name, typeid(DivNode));

  std::string new_mul8_name = ScopedName("Div4");
  AssertTypeEQ(new_mul8_name, typeid(MulNode));

  std::string new_mul9_name = ScopedName("Div5");
  AssertTypeEQ(new_mul9_name, typeid(MulNode));

  std::string new_div9_name = ScopedName("Div6");
  AssertTypeEQ(new_div9_name, typeid(DivNode));

  std::string new_div10_name = ScopedName("Div7");
  AssertTypeEQ(new_div10_name, typeid(DivNode));

  std::string new_div11_name = ScopedName("Div8");
  AssertTypeEQ(new_div11_name, typeid(DivNode));

  AssertNodesDeleted(
      {div1, mul1, div2, mul2, div3, mul7, div4, mul8, div5, div6, div7, div8});

  AssertInputsEQ(new_div1_name, {C1->name(), I2->name()});
  AssertInputsEQ(new_mul1_name, {I1->name(), new_div1_name});
  AssertInputsEQ(new_mul2_name, {C2->name(), C3->name()});
  AssertInputsEQ(new_div2_name, {new_mul2_name, I3->name()});
  AssertInputsEQ(mul3->name(), {C4->name(), I5->name()});
  AssertInputsEQ(mul4->name(), {I4->name(), mul3->name()});
  AssertInputsEQ(mul5->name(), {C6->name(), C5->name()});
  AssertInputsEQ(mul6->name(), {I6->name(), mul5->name()});
  AssertInputsEQ(new_div7_name, {I2->name(), C1->name()});
  AssertInputsEQ(new_mul7_name, {I1->name(), new_div7_name});
  AssertInputsEQ(new_div8_name, {C3->name(), C2->name()});
  AssertInputsEQ(new_mul8_name, {I3->name(), new_div8_name});
  AssertInputsEQ(new_mul9_name, {C4->name(), I5->name()});
  AssertInputsEQ(new_div9_name, {I4->name(), new_mul9_name});
  AssertInputsEQ(new_div10_name, {C6->name(), C5->name()});
  AssertInputsEQ(new_div11_name, {new_div10_name, I6->name()});
  AssertInputsEQ(addn->name(),
                 {new_mul1_name, new_div2_name, mul4->name(), mul6->name(),
                  new_mul7_name, new_mul8_name, new_div9_name, new_div11_name});
}

}  // namespace deepx_core
