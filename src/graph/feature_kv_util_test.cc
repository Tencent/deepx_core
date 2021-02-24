// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_gtest.h>
#include <deepx_core/graph/feature_kv_util.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/tensor_map.h>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace deepx_core {

class FeatureKVUtilTest : public testing::Test, public FeatureKVUtil {
 protected:
  std::default_random_engine engine;
  VariableNode* W1node = nullptr;
  VariableNode* W2node = nullptr;
  VariableNode* W3node = nullptr;
  VariableNode* W4node = nullptr;
  VariableNode* W5node = nullptr;
  VariableNode* W6node = nullptr;
  VariableNode* W7node = nullptr;
  Graph graph;
  TensorMap param;
  std::string key, value;
  std::vector<std::string> keys, values;
  std::vector<int16_t> codes;
  ParamParserStat stat;

 protected:
  void SetUp() override {
    InitGraph();
    InitParam(&param);
  }

  void InitGraph() {
    W1node = new VariableNode("W1", Shape(2, 3), TENSOR_TYPE_TSR,
                              TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W2node = new VariableNode("W2", Shape(3, 4), TENSOR_TYPE_TSR,
                              TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W3node = new VariableNode("W3", Shape(0, 2), TENSOR_TYPE_SRM,
                              TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W4node = new VariableNode("W4", Shape(0, 4), TENSOR_TYPE_SRM,
                              TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W5node = new VariableNode("W5", Shape(0, 8), TENSOR_TYPE_SRM,
                              TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
    W6node = new VariableNode("W6", Shape(0, 16), TENSOR_TYPE_SRM,
                              TENSOR_INITIALIZER_TYPE_ZEROS, 0, 1);
    W7node = new VariableNode("W7", Shape(0, 1), TENSOR_TYPE_SRM,
                              TENSOR_INITIALIZER_TYPE_ZEROS, 0, 1);
    ASSERT_TRUE(graph.Compile(
        {W1node, W2node, W3node, W4node, W5node, W6node, W7node}, 1));
  }

  void InitParamPlaceholder(TensorMap* _param) {
    auto& W1 = _param->insert<tsr_t>(W1node->name());
    auto& W2 = _param->insert<tsr_t>(W2node->name());
    auto& W3 = _param->insert<srm_t>(W3node->name());
    auto& W4 = _param->insert<srm_t>(W4node->name());
    auto& W5 = _param->insert<srm_t>(W5node->name());
    auto& W6 = _param->insert<srm_t>(W6node->name());
    auto& W7 = _param->insert<srm_t>(W7node->name());
    W1.resize(W1node->shape());
    W2.resize(W2node->shape());
    W3.set_col(W3node->shape()[1]);
    W3.set_initializer(W3node->initializer_type(),
                       (float_t)W3node->initializer_param1(),
                       (float_t)W3node->initializer_param2());
    W4.set_col(W4node->shape()[1]);
    W4.set_initializer(W4node->initializer_type(),
                       (float_t)W4node->initializer_param1(),
                       (float_t)W4node->initializer_param2());
    W5.set_col(W5node->shape()[1]);
    W5.set_initializer(W5node->initializer_type(),
                       (float_t)W5node->initializer_param1(),
                       (float_t)W5node->initializer_param2());
    W6.set_col(W6node->shape()[1]);
    W6.set_initializer(W6node->initializer_type(),
                       (float_t)W6node->initializer_param1(),
                       (float_t)W6node->initializer_param2());
    W7.set_col(W7node->shape()[1]);
    W7.set_initializer(W7node->initializer_type(),
                       (float_t)W7node->initializer_param1(),
                       (float_t)W7node->initializer_param2());
  }

  void InitParam(TensorMap* _param) {
    InitParamPlaceholder(_param);
    auto& W1 = _param->get<tsr_t>(W1node->name());
    auto& W2 = _param->get<tsr_t>(W2node->name());
    auto& W3 = _param->get<srm_t>(W3node->name());
    auto& W4 = _param->get<srm_t>(W4node->name());
    auto& W5 = _param->get<srm_t>(W5node->name());
    auto& W6 = _param->get<srm_t>(W6node->name());
    auto& W7 = _param->get<srm_t>(W7node->name());
    W1.randn(engine);
    W2.randn(engine);
    // W3: 0, 1, 2, 3, 4
    // W4: 0, 2, 4, 6, 8
    // W5: 0, 1, 2, 3, 4
    // W6: 0, 2, 4, 6, 8
    // W7: 0, 3, 6, 9
    for (int i = 0; i < 5; ++i) {
      W3.get_row(engine, i);
      W5.get_row(engine, i);
    }
    for (int i = 0; i < 10; i += 2) {
      W4.get_row(engine, i);
      W6.get_row(engine, i);
    }
    for (int i = 0; i < 10; i += 3) {
      W7.get_row(engine, i);
    }
  }

  void TestWriteSparseParam_1_SparseParamParser(int version) {
    CheckVersion(version);

    OutputStringStream os;
    ASSERT_TRUE(WriteSparseParam(os, graph, param, version));
    ASSERT_TRUE(GetKeyValues(os.GetString(), &keys, &values));
    codes.assign(keys.size(), 0);

    TensorMap parsed_param;
    InitParamPlaceholder(&parsed_param);

    SparseParamParser parser;
    parser.Init(&graph, &parsed_param, version);
    parser.Parse(keys, values, codes, &stat);

    if (version == 2) {
      EXPECT_EQ(param.get<srm_t>(W3node->name()),
                parsed_param.get<srm_t>(W3node->name()));
      EXPECT_EQ(param.get<srm_t>(W4node->name()),
                parsed_param.get<srm_t>(W4node->name()));
      EXPECT_EQ(param.get<srm_t>(W5node->name()),
                parsed_param.get<srm_t>(W5node->name()));
      EXPECT_EQ(param.get<srm_t>(W6node->name()),
                parsed_param.get<srm_t>(W6node->name()));
    } else if (version == 3) {
#if HAVE_SAGE2 == 1
      EXPECT_SRM_NEAR(param.get<srm_t>(W3node->name()),
                      parsed_param.get<srm_t>(W3node->name()));
      EXPECT_SRM_NEAR(param.get<srm_t>(W4node->name()),
                      parsed_param.get<srm_t>(W4node->name()));
      EXPECT_SRM_NEAR(param.get<srm_t>(W5node->name()),
                      parsed_param.get<srm_t>(W5node->name()));
      EXPECT_SRM_NEAR(param.get<srm_t>(W6node->name()),
                      parsed_param.get<srm_t>(W6node->name()));
      EXPECT_SRM_NEAR(param.get<srm_t>(W7node->name()),
                      parsed_param.get<srm_t>(W7node->name()));
#endif
    }

    // W3: 0, 1, 2, 3, 4
    EXPECT_EQ(parsed_param.get<srm_t>(W3node->name()).size(), 5u);
    // W4: 0, 2, 4, 6, 8
    EXPECT_EQ(parsed_param.get<srm_t>(W4node->name()).size(), 5u);
    // W5: 0, 1, 2, 3, 4
    EXPECT_EQ(parsed_param.get<srm_t>(W5node->name()).size(), 5u);
    // W6: 0, 2, 4, 6, 8
    EXPECT_EQ(parsed_param.get<srm_t>(W6node->name()).size(), 5u);
    if (version == 2) {
      EXPECT_TRUE(parsed_param.get<srm_t>(W7node->name()).empty());
    } else if (version == 3) {
#if HAVE_SAGE2 == 1
      // W7: 0, 3, 6, 9
      EXPECT_EQ(parsed_param.get<srm_t>(W7node->name()).size(), 4u);
#endif
    }

    // 0, 1, 2, 3, 4, 6, 8, 9
    EXPECT_EQ(stat.key_exist, 8);
    EXPECT_EQ(stat.key_not_exist, 0);
    EXPECT_EQ(stat.key_bad, 0);
    EXPECT_EQ(stat.value_bad, 0);
    EXPECT_EQ(stat.feature_kv_client_error, 0);
  }

  void TestWriteSparseParam_2_SparseParamParser(int version) {
    OutputStringStream os;
    id_set_t id_set = {0, 2, 4, 6, 8};
    ASSERT_TRUE(WriteSparseParam(os, graph, param, id_set, version));
    ASSERT_TRUE(GetKeyValues(os.GetString(), &keys, &values));
    codes.assign(keys.size(), 0);

    TensorMap parsed_param;
    InitParamPlaceholder(&parsed_param);

    SparseParamParser parser;
    parser.Init(&graph, &parsed_param, version);
    parser.Parse(keys, values, codes, &stat);

    // W3: 0, 2, 4
    EXPECT_EQ(parsed_param.get<srm_t>(W3node->name()).size(), 3u);
    // W4: 0, 2, 4, 6, 8
    EXPECT_EQ(parsed_param.get<srm_t>(W4node->name()).size(), 5u);
    // W5: 0, 2, 4
    EXPECT_EQ(parsed_param.get<srm_t>(W5node->name()).size(), 3u);
    // W6: 0, 2, 4, 6, 8
    EXPECT_EQ(parsed_param.get<srm_t>(W6node->name()).size(), 5u);
    if (version == 2) {
      EXPECT_TRUE(parsed_param.get<srm_t>(W7node->name()).empty());
    } else if (version == 3) {
#if HAVE_SAGE2 == 1
      // W7: 0, 6
      EXPECT_EQ(parsed_param.get<srm_t>(W7node->name()).size(), 2u);
#endif
    }

    // 0, 2, 4, 6, 8
    EXPECT_EQ(stat.key_exist, 5);
    EXPECT_EQ(stat.key_not_exist, 0);
    EXPECT_EQ(stat.key_bad, 0);
    EXPECT_EQ(stat.value_bad, 0);
    EXPECT_EQ(stat.feature_kv_client_error, 0);
  }
};

TEST_F(FeatureKVUtilTest, WriteVersion) {
  OutputStringStream os;
  ASSERT_TRUE(WriteVersion(os, 2));
  ASSERT_TRUE(GetKeyValue(os.GetString(), &key, &value));

  EXPECT_EQ(key, GetVersionKey());
  int parsed_version;
  ASSERT_TRUE(GetVersion(value, &parsed_version));
  EXPECT_EQ(parsed_version, 2);
}

TEST_F(FeatureKVUtilTest, WriteGraph) {
  OutputStringStream os;
  ASSERT_TRUE(WriteGraph(os, graph));
  ASSERT_TRUE(GetKeyValue(os.GetString(), &key, &value));

  EXPECT_EQ(key, GetGraphKey());
  Graph parsed_graph;
  ASSERT_TRUE(GetGraph(value, &parsed_graph));
  EXPECT_EQ(graph.Dot(), parsed_graph.Dot());
}

TEST_F(FeatureKVUtilTest, WriteDenseParam_DenseParamParser) {
  OutputStringStream os;
  ASSERT_TRUE(WriteDenseParam(os, param));
  ASSERT_TRUE(GetKeyValues(os.GetString(), &keys, &values));
  codes.assign(keys.size(), 0);

  TensorMap parsed_param;
  InitParamPlaceholder(&parsed_param);

  DenseParamParser parser;
  parser.Init(&graph, &parsed_param);
  parser.Parse(keys, values, codes, &stat);

  EXPECT_EQ(param.get<tsr_t>(W1node->name()),
            parsed_param.get<tsr_t>(W1node->name()));
  EXPECT_EQ(param.get<tsr_t>(W2node->name()),
            parsed_param.get<tsr_t>(W2node->name()));

  EXPECT_EQ(stat.key_exist, 2);
  EXPECT_EQ(stat.key_not_exist, 0);
  EXPECT_EQ(stat.key_bad, 0);
  EXPECT_EQ(stat.value_bad, 0);
  EXPECT_EQ(stat.feature_kv_client_error, 0);
}

TEST_F(FeatureKVUtilTest, WriteSparseParam_1_SparseParamParser_version2) {
  TestWriteSparseParam_1_SparseParamParser(2);
}

TEST_F(FeatureKVUtilTest, WriteSparseParam_2_SparseParamParser_version2) {
  TestWriteSparseParam_2_SparseParamParser(2);
}

TEST_F(FeatureKVUtilTest, WriteSparseParam_1_SparseParamParser_version3) {
#if HAVE_SAGE2 == 1
  TestWriteSparseParam_1_SparseParamParser(3);
#endif
}

TEST_F(FeatureKVUtilTest, WriteSparseParam_2_SparseParamParser_version3) {
#if HAVE_SAGE2 == 1
  TestWriteSparseParam_2_SparseParamParser(3);
#endif
}

}  // namespace deepx_core
