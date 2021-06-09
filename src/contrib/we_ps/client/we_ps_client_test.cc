// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/contrib/we_ps/client/we_ps_client.h>
#include <deepx_core/dx_gtest.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

namespace deepx_core {

#define EXPECT_TSR_DELTA(W, delta_W, new_W)               \
  do {                                                    \
    tsr_t expected_new_W = W;                             \
    ll_sparse_tensor_t::add_to(delta_W, &expected_new_W); \
    EXPECT_TSR_NEAR(new_W, expected_new_W);               \
  } while (0)

#define EXPECT_SRM_DELTA(W, delta_W, new_W)               \
  do {                                                    \
    srm_t expected_new_W = W;                             \
    ll_sparse_tensor_t::add_to(delta_W, &expected_new_W); \
    EXPECT_SRM_NEAR(new_W, expected_new_W);               \
  } while (0)

class WePSClientTest : public testing::Test, public DataType {
 protected:
  std::default_random_engine engine;
  std::unique_ptr<WePSClient> client;

 protected:
  void TestSetTSR_GetTSR_1() {
    tsr_t W0;
    W0.resize(2, 3).randn(engine);
    ASSERT_TRUE(client->SetTSR("TSR_W0", W0));

    tsr_t new_W0;
    new_W0.resize(W0.shape());
    ASSERT_TRUE(client->GetTSR("TSR_W0", &new_W0));
    EXPECT_TSR_NEAR(new_W0, W0);
  }

  void TestSetTSR_GetTSR_1_EmptyTSR() {
    tsr_t W0;
    ASSERT_TRUE(client->SetTSR("TSR_W0", W0));

    tsr_t new_W0;
    ASSERT_TRUE(client->GetTSR("TSR_W0", &new_W0));
    EXPECT_TSR_NEAR(new_W0, W0);
  }

  void TestUpdateTSR_1() {
    tsr_t W0;
    W0.resize(2, 3).randn(engine);
    ASSERT_TRUE(client->SetTSR("TSR_W0", W0));

    tsr_t delta_W0, new_W0;
    delta_W0.resize(W0.shape()).randn(engine);
    new_W0.resize(W0.shape());
    ASSERT_TRUE(client->UpdateTSR("TSR_W0", delta_W0, &new_W0));
    EXPECT_TSR_DELTA(W0, delta_W0, new_W0);
  }

  void TestUpdateTSR_1_EmptyTSR() {
    tsr_t W0;
    ASSERT_TRUE(client->SetTSR("TSR_W0", W0));

    tsr_t delta_W0, new_W0;
    ASSERT_TRUE(client->UpdateTSR("TSR_W0", delta_W0, &new_W0));
    EXPECT_TSR_DELTA(W0, delta_W0, new_W0);
  }

  void TestSetTSR_GetTSR_2() {
    TensorMap param;
    auto& W0 = param.insert<tsr_t>("TSR_W0").resize(2, 3).randn(engine);
    auto& W1 = param.insert<tsr_t>("TSR_W1").resize(3, 4).randn(engine);
    ASSERT_TRUE(client->SetTSR(param));

    TensorMap new_param;
    auto& new_W0 = new_param.insert<tsr_t>("TSR_W0").resize(W0.shape());
    auto& new_W1 = new_param.insert<tsr_t>("TSR_W1").resize(W1.shape());
    ASSERT_TRUE(client->GetTSR(&new_param));
    EXPECT_TSR_NEAR(new_W0, W0);
    EXPECT_TSR_NEAR(new_W1, W1);
  }

  void TestSetTSR_2_NoTSR() {
    TensorMap param;
    ASSERT_TRUE(client->SetTSR(param));
  }

  void TestGetTSR_2_NoTSR() {
    TensorMap param;
    ASSERT_TRUE(client->GetTSR(&param));
    ASSERT_TRUE(param.empty());
  }

  void TestUpdateTSR_2() {
    TensorMap param;
    auto& W0 = param.insert<tsr_t>("TSR_W0").resize(2, 3).randn(engine);
    auto& W1 = param.insert<tsr_t>("TSR_W1").resize(3, 4).randn(engine);
    ASSERT_TRUE(client->SetTSR(param));

    TensorMap delta_param, new_param;
    auto& delta_W0 =
        delta_param.insert<tsr_t>("TSR_W0").resize(W0.shape()).randn(engine);
    auto& delta_W1 =
        delta_param.insert<tsr_t>("TSR_W1").resize(W1.shape()).randn(engine);
    auto& new_W0 = new_param.insert<tsr_t>("TSR_W0").resize(W0.shape());
    auto& new_W1 = new_param.insert<tsr_t>("TSR_W1").resize(W1.shape());
    ASSERT_TRUE(client->UpdateTSR(delta_param, &new_param));
    EXPECT_TSR_DELTA(W0, delta_W0, new_W0);
    EXPECT_TSR_DELTA(W1, delta_W1, new_W1);
  }

  void TestUpdateTSR_2_NoTSR() {
    TensorMap delta_param, new_param;
    ASSERT_TRUE(client->UpdateTSR(delta_param, &new_param));
  }

  void TestSetSRM_GetSRM_1() {
    srm_t W0;
    W0.set_col(8);
    W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W0.get_row(engine, 1);
    W0.get_row(engine, 2);
    ASSERT_TRUE(client->SetSRM("SRM_W0", W0));

    id_set_t id_set{1, 2, 3, 4};
    srm_t new_W0;
    new_W0.set_col(8);
    ASSERT_TRUE(client->GetSRM("SRM_W0", id_set, &new_W0));
    EXPECT_SRM_NEAR(new_W0, W0);
  }

  void TestSetSRM_1_EmptySRM() {
    srm_t W0;
    W0.set_col(8);
    W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ASSERT_TRUE(client->SetSRM("SRM_W0", W0));
  }

  void TestGetSRM_1_EmptyIdSet() {
    id_set_t id_set;
    srm_t new_W0;
    new_W0.set_col(8);
    ASSERT_TRUE(client->GetSRM("SRM_W0", id_set, &new_W0));
    ASSERT_TRUE(new_W0.empty());
  }

  void TestUpdateSRM_1() {
    srm_t W0;
    W0.set_col(8);
    W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W0.get_row(engine, 1);
    W0.get_row(engine, 2);
    ASSERT_TRUE(client->SetSRM("SRM_W0", W0));

    srm_t delta_W0;
    delta_W0.set_col(8);
    delta_W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    delta_W0.get_row(engine, 1);
    delta_W0.get_row(engine, 2);
    ASSERT_TRUE(client->UpdateSRM("SRM_W0", delta_W0));

    id_set_t id_set{1, 2, 3, 4};
    srm_t new_W0;
    new_W0.set_col(8);
    ASSERT_TRUE(client->GetSRM("SRM_W0", id_set, &new_W0));
    EXPECT_SRM_DELTA(W0, delta_W0, new_W0);
  }

  void TestUpdateSRM_1_EmptySRM() {
    srm_t W0;
    W0.set_col(8);
    W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W0.get_row(engine, 1);
    W0.get_row(engine, 2);
    ASSERT_TRUE(client->SetSRM("SRM_W0", W0));

    srm_t delta_W0;
    delta_W0.set_col(8);
    delta_W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    ASSERT_TRUE(client->UpdateSRM("SRM_W0", delta_W0));

    id_set_t id_set{1, 2, 3, 4};
    srm_t new_W0;
    new_W0.set_col(8);
    ASSERT_TRUE(client->GetSRM("SRM_W0", id_set, &new_W0));
    EXPECT_SRM_DELTA(W0, delta_W0, new_W0);
  }

  void TestSetSRM_GetSRM_2() {
    TensorMap param;
    auto& W0 = param.insert<srm_t>("SRM_W0");
    W0.set_col(8);
    W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W0.get_row(engine, 1);
    W0.get_row(engine, 2);
    auto& W1 = param.insert<srm_t>("SRM_W1");
    W1.set_col(8);
    W1.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W1.get_row(engine, 3);
    W1.get_row(engine, 4);
    ASSERT_TRUE(client->SetSRM(param));

    std::unordered_map<std::string, id_set_t> id_set_map{{"SRM_W0", {1, 2}},
                                                         {"SRM_W1", {3, 4}}};
    TensorMap new_param;
    auto& new_W0 = new_param.insert<srm_t>("SRM_W0");
    new_W0.set_col(8);
    auto& new_W1 = new_param.insert<srm_t>("SRM_W1");
    new_W1.set_col(8);
    ASSERT_TRUE(client->GetSRM(id_set_map, &new_param));
    EXPECT_SRM_NEAR(new_W0, W0);
    EXPECT_SRM_NEAR(new_W1, W1);
  }

  void TestUpdateSRM_2() {
    TensorMap param;
    auto& W0 = param.insert<srm_t>("SRM_W0");
    W0.set_col(8);
    W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W0.get_row(engine, 1);
    W0.get_row(engine, 2);
    auto& W1 = param.insert<srm_t>("SRM_W1");
    W1.set_col(8);
    W1.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W1.get_row(engine, 3);
    W1.get_row(engine, 4);
    ASSERT_TRUE(client->SetSRM(param));

    TensorMap delta_param;
    auto& delta_W0 = delta_param.insert<srm_t>("SRM_W0");
    delta_W0.set_col(8);
    delta_W0.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    delta_W0.get_row(engine, 1);
    delta_W0.get_row(engine, 2);
    auto& delta_W1 = delta_param.insert<srm_t>("SRM_W1");
    delta_W1.set_col(8);
    delta_W1.set_initializer(TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    delta_W1.get_row(engine, 3);
    delta_W1.get_row(engine, 4);
    ASSERT_TRUE(client->UpdateSRM(delta_param));

    std::unordered_map<std::string, id_set_t> id_set_map{{"SRM_W0", {1, 2}},
                                                         {"SRM_W1", {3, 4}}};
    TensorMap new_param;
    auto& new_W0 = new_param.insert<srm_t>("SRM_W0");
    new_W0.set_col(8);
    auto& new_W1 = new_param.insert<srm_t>("SRM_W1");
    new_W1.set_col(8);
    ASSERT_TRUE(client->GetSRM(id_set_map, &new_param));
    EXPECT_SRM_DELTA(W0, delta_W0, new_W0);
    EXPECT_SRM_DELTA(W1, delta_W1, new_W1);
  }

  void TestSetGraph_GetGraph() {
    Graph graph;
    VariableNode X1("X1", Shape(2, 3));
    ReduceMeanNode X2("X2", &X1, 1, 1);
    ASSERT_TRUE(graph.Compile({&X2}, 0));

    Graph read_graph;
    int exist = 0;
    ASSERT_TRUE(client->GetGraph(&read_graph, &exist));
    ASSERT_TRUE(exist == 0);

    ASSERT_TRUE(client->SetGraph(graph));
    ASSERT_TRUE(client->GetGraph(&read_graph, &exist));
    ASSERT_TRUE(exist == 1);
    ASSERT_TRUE(read_graph.compiled());

    ASSERT_EQ(graph.target_size(), read_graph.target_size());
    for (int i = 0; i < graph.target_size(); ++i) {
      EXPECT_EQ(graph.target(i).name(), read_graph.target(i).name());
      EXPECT_EQ(graph.target(i).forward_name(),
                read_graph.target(i).forward_name());
    }
  }
};

class WePSProxyClientTest : public WePSClientTest {
 protected:
  void SetUp() override {
    client = NewWePSClient("proxy");
    ASSERT_TRUE(client);
    StringMap config;
    config["model_id"] = "7000";
    config["client_uuid"] = "11";
    ASSERT_TRUE(client->InitConfig(config));
  }
};

#if 0
TEST_F(WePSProxyClientTest, SetTSR_GetTSR_1) { TestSetTSR_GetTSR_1(); }
TEST_F(WePSProxyClientTest, SetTSR_GetTSR_1_EmptyTSR) {
  TestSetTSR_GetTSR_1_EmptyTSR();
}
TEST_F(WePSProxyClientTest, UpdateTSR_1) { TestUpdateTSR_1(); }
TEST_F(WePSProxyClientTest, UpdateTSR_1_EmptyTSR) {
  TestUpdateTSR_1_EmptyTSR();
}
TEST_F(WePSProxyClientTest, SetTSR_GetTSR_2) { TestSetTSR_GetTSR_2(); }
TEST_F(WePSProxyClientTest, SetTSR_2_NoTSR) { TestSetTSR_2_NoTSR(); }
TEST_F(WePSProxyClientTest, GetTSR_2_NoTSR) { TestGetTSR_2_NoTSR(); }
TEST_F(WePSProxyClientTest, UpdateTSR_2) { TestUpdateTSR_2(); }
TEST_F(WePSProxyClientTest, UpdateTSR_2_NoTSR) { TestUpdateTSR_2_NoTSR(); }
TEST_F(WePSProxyClientTest, SetSRM_GetSRM_1) { TestSetSRM_GetSRM_1(); }
TEST_F(WePSProxyClientTest, SetSRM_1_EmptySRM) { TestSetSRM_1_EmptySRM(); }
TEST_F(WePSProxyClientTest, GetSRM_1_EmptyIdSet) { TestGetSRM_1_EmptyIdSet(); }
TEST_F(WePSProxyClientTest, UpdateSRM_1) { TestUpdateSRM_1(); }
TEST_F(WePSProxyClientTest, UpdateSRM_1_EmptySRM) {
  TestUpdateSRM_1_EmptySRM();
}
TEST_F(WePSProxyClientTest, SetSRM_GetSRM_2) { TestSetSRM_GetSRM_2(); }
TEST_F(WePSProxyClientTest, UpdateSRM_2) { TestUpdateSRM_2(); }
TEST_F(WePSProxyClientTest, SetGraph_GetGraph) { TestSetGraph_GetGraph(); }
#endif

class WePSMockClientTest : public WePSClientTest {
 protected:
  void SetUp() override {
    client = NewWePSClient("mock");
    ASSERT_TRUE(client);
    StringMap config;
    ASSERT_TRUE(client->InitConfig(config));
  }
};

TEST_F(WePSMockClientTest, SetTSR_GetTSR_1) { TestSetTSR_GetTSR_1(); }
TEST_F(WePSMockClientTest, SetTSR_GetTSR_1_EmptyTSR) {
  TestSetTSR_GetTSR_1_EmptyTSR();
}
TEST_F(WePSMockClientTest, UpdateTSR_1) { TestUpdateTSR_1(); }
TEST_F(WePSMockClientTest, UpdateTSR_1_EmptyTSR) { TestUpdateTSR_1_EmptyTSR(); }
TEST_F(WePSMockClientTest, SetTSR_GetTSR_2) { TestSetTSR_GetTSR_2(); }
TEST_F(WePSMockClientTest, SetTSR_2_NoTSR) { TestSetTSR_2_NoTSR(); }
TEST_F(WePSMockClientTest, GetTSR_2_NoTSR) { TestGetTSR_2_NoTSR(); }
TEST_F(WePSMockClientTest, UpdateTSR_2) { TestUpdateTSR_2(); }
TEST_F(WePSMockClientTest, UpdateTSR_2_NoTSR) { TestUpdateTSR_2_NoTSR(); }
TEST_F(WePSMockClientTest, SetSRM_GetSRM_1) { TestSetSRM_GetSRM_1(); }
TEST_F(WePSMockClientTest, SetSRM_1_EmptySRM) { TestSetSRM_1_EmptySRM(); }
TEST_F(WePSMockClientTest, GetSRM_1_EmptyIdSet) { TestGetSRM_1_EmptyIdSet(); }
TEST_F(WePSMockClientTest, UpdateSRM_1) { TestUpdateSRM_1(); }
TEST_F(WePSMockClientTest, UpdateSRM_1_EmptySRM) { TestUpdateSRM_1_EmptySRM(); }
TEST_F(WePSMockClientTest, SetSRM_GetSRM_2) { TestSetSRM_GetSRM_2(); }
TEST_F(WePSMockClientTest, UpdateSRM_2) { TestUpdateSRM_2(); }
TEST_F(WePSMockClientTest, SetGraph_GetGraph) { TestSetGraph_GetGraph(); }

}  // namespace deepx_core
