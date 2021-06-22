// Copyright 2021 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/contrib/we_ps/graph/we_ps_util.h>
#include <deepx_core/dx_gtest.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/tensor.h>
#include <string>
#include <vector>

namespace deepx_core {

class WePSUtilTest : public testing::Test, public WePSUtil {};

TEST_F(WePSUtilTest, DenseParamParser_Parse) {
  std::string name = "W";
  std::vector<std::string> keys{name};
  std::vector<std::vector<float_t>> values{{1, 2, 3, 4, 5, 6}};

  TensorMap param;
  param.insert<tsr_t>(name) = tsr_t(values[0]).resize(2, 3);

  TensorMap parsed_param;
  auto& W = parsed_param.insert<tsr_t>(name);
  W.resize(2, 3);

  DenseParamParser parser;
  ParamParserStat stat;
  parser.Init(&parsed_param);
  parser.Parse(keys, values, &stat);

  EXPECT_EQ(param.get<tsr_t>(name), parsed_param.get<tsr_t>(name));
  EXPECT_EQ(stat.key_exist, 1);
  EXPECT_EQ(stat.key_not_exist, 0);
  EXPECT_EQ(stat.key_bad, 0);
  EXPECT_EQ(stat.value_bad, 0);
  EXPECT_EQ(stat.we_ps_client_error, 0);
}

TEST_F(WePSUtilTest, SparseParamParser_Parse) {
  std::string name = "W";
  int_t id = 1;
  std::vector<float_t> v{1, 1};
  std::string key;
  std::string value;
  WePSUtil::GetSparseParamKey(name, id, &key);
  WePSUtil::GetSparseParamValue(v, &value);
  std::vector<std::string> keys{key};
  std::vector<std::string> values{value};
  std::vector<int16_t> codes(keys.size());

  TensorMap param;
  param.insert<srm_t>(name) = srm_t{{id}, {{v[0], v[1]}}};

  TensorMap parsed_param;
  auto& W = parsed_param.insert<srm_t>(name);
  W.set_col(2);

  SparseParamParser parser;
  ParamParserStat stat;
  parser.set_view(0);
  parser.Init(&parsed_param);
  parser.Parse(keys, values, codes, &stat);

  EXPECT_EQ(param.get<srm_t>(name), parsed_param.get<srm_t>(name));
  EXPECT_EQ(parsed_param.get<srm_t>(name).size(), 1u);
  EXPECT_EQ(stat.key_exist, 1);
  EXPECT_EQ(stat.key_not_exist, 0);
  EXPECT_EQ(stat.key_bad, 0);
  EXPECT_EQ(stat.value_bad, 0);
  EXPECT_EQ(stat.we_ps_client_error, 0);
}

}  // namespace deepx_core
