// Copyright 2020 the deepx authors.
// Author: Shuting Guo (tinkleguo@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include "cse.h"
#include <gtest/gtest.h>
#include <memory>
#include "simp_test.h"

namespace deepx_core {

class CSESimpTest : public SimpTestBase {
 protected:
  Shape shape;
  std::unique_ptr<CSESimp> common_subexpression_elimination;

 protected:
  void SetUp() override {
    shape.resize(3, 3);
    common_subexpression_elimination.reset(new CSESimp);
  }
  void Simplify() override {
    common_subexpression_elimination->Simplify(&item);
  }
};

/*
 *                       AddN2                                  AddN2
 *       _____________/   |    \                             /    |   \_____
 *       |                |     \                           /     |         |
 *  ConstantLike1 ConstantLike2 AddN1-----     ConstantLike1 ConstantLike2 AddN1
 *       |       /____________/    \      |           |______/____________/  |
 *       |      /            /      \     |           |     /            /   |
 *  ReduceMean1   ReduceMean2 ReduceMean3 C1      ReduceMean1 ReduceMean3    C1
 *       |           |           |            ->     |       /
 *    Sigmoid1    Sigmoid2    Sigmoid3             Sigmoid1
 *        \__________|__________/                     |
 *                   I1                               I1
 */
TEST_F(CSESimpTest, CSESimp) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* C1 = Constant("C1", Shape(3), 1);
  auto* sigmoid1 = Sigmoid("Sigmoid1", I1);
  auto* sigmoid2 = Sigmoid("Sigmoid2", I1);
  auto* sigmoid3 = Sigmoid("Sigmoid3", I1);
  auto* reduce_mean1 = ReduceMean("ReduceMean1", sigmoid1, 0, 0);
  auto* reduce_mean2 = ReduceMean("ReduceMean2", sigmoid2, 0, 0);
  auto* reduce_mean3 = ReduceMean("ReduceMean3", sigmoid3, 1, 0);
  auto* constant_like1 = ConstantLike(
      "ConstantLike1", reduce_mean1,
      TENSOR_INITIALIZER_TYPE ::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  auto* constant_like2 = ConstantLike(
      "ConstantLike2", reduce_mean1,
      TENSOR_INITIALIZER_TYPE ::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  auto* addn1 = AddN("AddN1", {reduce_mean1, reduce_mean2, reduce_mean3, C1});
  auto* addn2 = AddN("AddN2", {constant_like1, constant_like2, addn1});
  ASSERT_TRUE(graph.Compile({addn2}, 1));
  item.FromGraph(graph);

  SimplifyTwice();
  ASSERT_EQ(9, item.node_size());

  AssertNodesDeleted({sigmoid2, sigmoid3, reduce_mean2});

  AssertInputsEQ(sigmoid1->name(), {I1->name()});
  AssertInputsEQ(reduce_mean1->name(), {sigmoid1->name()});
  AssertInputsEQ(reduce_mean3->name(), {sigmoid1->name()});
  AssertInputsEQ(addn1->name(), {reduce_mean1->name(), reduce_mean1->name(),
                                 reduce_mean3->name(), C1->name()});
  AssertInputsEQ(constant_like1->name(), {reduce_mean1->name()});
  AssertInputsEQ(constant_like2->name(), {reduce_mean1->name()});
  AssertInputsEQ(addn2->name(), {constant_like1->name(), constant_like2->name(),
                                 addn1->name()});
}

/*
 *         AddN                    AddN
 *     /   |  |   \              / / |  \
 *    /    /  \    \            / /  |   \
 *  Add1 Add2 GEMM1 GEMM2  ->  Add1 GEMM1 GEMM2
 *  | \  / |   |  \  / |       | |   |  \  / |
 *  |  / \ |   |  /  \ |       | |   |  /  \ |
 *  I1    I2   I3     I4       I1 I2  I3     I4
 */
TEST_F(CSESimpTest, CommonSubExpressionElimination_inputs_commutability) {
  auto* I1 = new InstanceNode("I1", shape, TENSOR_TYPE_TSR);
  auto* I2 = new InstanceNode("I2", shape, TENSOR_TYPE_TSR);
  auto* I3 = new InstanceNode("I3", shape, TENSOR_TYPE_TSR);
  auto* I4 = new InstanceNode("I", shape, TENSOR_TYPE_TSR);
  auto* add1 = Add("Add1", I2, I1);
  auto* add2 = Add("Add2", I1, I2);
  auto* gemm1 = GEMM("GEMM1", I3, I4, 0, 0);
  auto* gemm2 = GEMM("GEMM2", I4, I3, 0, 0);
  auto* addn = AddN("AddN", {add1, add2, gemm1, gemm2});
  ASSERT_TRUE(graph.Compile({addn}, 1));
  item.FromGraph(graph);

  SimplifyTwice();
  ASSERT_EQ(8, item.node_size());

  AssertNodesDeleted({add2});

  AssertInputsEQ(addn->name(),
                 {add1->name(), add1->name(), gemm1->name(), gemm2->name()});
  AssertInputsEQ(add1->name(), {I2->name(), I1->name()});
  AssertInputsEQ(gemm1->name(), {I3->name(), I4->name()});
  AssertInputsEQ(gemm2->name(), {I4->name(), I3->name()});
}

}  // namespace deepx_core
