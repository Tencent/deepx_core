// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/str_util.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/feature_kv_util.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/instance_reader.h>
#include <deepx_core/graph/model.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/graph/tensor_map.h>
#include <gflags/gflags.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#if HAVE_WXG_FEATURE_KV_CLIENT == 0
/************************************************************************/
/* MockFeatureKVClientConfig */
/************************************************************************/
class MockFeatureKVClientConfig {};

/************************************************************************/
/* MockFeatureKVClient */
/************************************************************************/
class MockFeatureKVClient {
 private:
  std::unordered_map<std::string, std::string> kv_map_;

 public:
  void InitMock(const std::vector<std::string>& keys,
                const std::vector<std::string>& values) {
    kv_map_.clear();
    for (size_t i = 0; i < keys.size(); ++i) {
      kv_map_.emplace(keys[i], values[i]);
    }
  }

 public:
  int Get(uint32_t /*table_id*/, const std::string& key, std::string* value,
          uint32_t* /*version*/ = nullptr, uint32_t /*route_key*/ = 0,
          uint32_t /*cli_version*/ = 0) {
    auto it = kv_map_.find(key);
    if (it != kv_map_.end()) {
      *value = it->second;
      return 0;
    }
    return 1;
  }

  int BatchGet(uint32_t /*table_id*/, const std::vector<std::string>& keys,
               std::vector<int16_t>* codes, std::vector<std::string>* values,
               uint32_t* /*version*/ = nullptr, uint32_t /*route_key*/ = 0,
               uint32_t /*cli_version*/ = 0) {
    codes->assign(keys.size(), 1);
    values->assign(keys.size(), "");
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = kv_map_.find(keys[i]);
      if (it != kv_map_.end()) {
        (*codes)[i] = 0;
        (*values)[i] = it->second;
      }
    }
    return 0;
  }
};

using FeaturekvClientConfig = MockFeatureKVClientConfig;
using FeaturekvClient = MockFeatureKVClient;
#else
#include <featurekvclient.h>
#endif

namespace deepx_core {
namespace {

using int_t = FeatureKVUtil::int_t;
using tsr_t = FeatureKVUtil::tsr_t;
using srm_t = FeatureKVUtil::srm_t;
using csr_t = FeatureKVUtil::csr_t;

#if HAVE_WXG_FEATURE_KV_CLIENT == 0
/************************************************************************/
/* MockModelContext */
/************************************************************************/
class MockModelContext {
 private:
  Graph graph_;
  Model model_;
  int version_ = 2;

 public:
  void InitGraph() {
    DXINFO("Initializing mock graph...");
    auto* X = new InstanceNode(X_NAME, Shape(64, 0), TENSOR_TYPE_CSR);
    auto* Y = new InstanceNode(Y_NAME, Shape(64, 1), TENSOR_TYPE_TSR);
    auto* W1 = new VariableNode("W1", Shape(0, 8), TENSOR_TYPE_SRM,
                                TENSOR_INITIALIZER_TYPE_RANDN, 0, 1e-3);
    auto* Z1 = new EmbeddingLookupNode("Z1", X, W1);
    auto* W2 = new VariableNode("W2", Shape(8, 1), TENSOR_TYPE_TSR,
                                TENSOR_INITIALIZER_TYPE_RAND_XAVIER, 0, 0);
    auto* Z2 = new FullyConnectNode("Z2", Z1, W2);
    auto* L = new SigmoidBCELossNode("L", Z2, Y);
    auto* M = new ReduceMeanNode("M", L);
    auto* P = new SigmoidNode("P", Z2);
    DXCHECK_THROW(graph_.Compile({M, P}, 1));
    DXINFO("Done.");
  }

  void InitModel(const std::vector<int_t>& ids) {
    DXINFO("Initializing mock model...");
    std::default_random_engine engine;
    model_.Init(&graph_);
    DXCHECK_THROW(model_.InitParam(engine));
    for (auto& entry : *model_.mutable_param()) {
      Any& Wany = entry.second;
      if (Wany.is<srm_t>()) {
        auto& W = Wany.unsafe_to_ref<srm_t>();
        for (int_t id : ids) {
          W.get_row(engine, id);
        }
      }
    }
    DXINFO("Done.");
  }

  void GetKeyValues(std::vector<std::string>* keys,
                    std::vector<std::string>* values) const {
    OutputStringStream os;
    DXCHECK_THROW(
        FeatureKVUtil::WriteModel(os, graph_, model_.param(), version_));
    DXCHECK_THROW(FeatureKVUtil::GetKeyValues(os.GetString(), keys, values));
  }
};
#endif

/************************************************************************/
/* FeatureKVClientContext */
/************************************************************************/
class FeatureKVClientContext {
 private:
  int default_table_id_ = 0;
  std::unique_ptr<::FeaturekvClientConfig> config_;
  ::FeaturekvClient client_;

 public:
  void set_default_table_id(int default_table_id) noexcept {
    default_table_id_ = default_table_id;
  }

 public:
#if HAVE_WXG_FEATURE_KV_CLIENT == 0
  void Init(const std::vector<int_t>& ids) {
    MockModelContext mock;
    mock.InitGraph();
    mock.InitModel(ids);
    std::vector<std::string> keys, values;
    mock.GetKeyValues(&keys, &values);

    DXINFO("Initializing mock feature kv client...");
    client_.InitMock(keys, values);
    DXINFO("Done.");
  }
#else
  void InitFromConfig(const std::string& config, int timeout = 0) {
    DXINFO("Initializing feature kv config...");
    config_.reset(new ::FeaturekvClientConfig);
    DXCHECK_THROW(config_->SetFileIfUnset(config.c_str()) == 0);
    DXINFO("Done.");

    DXINFO("Initializing feature kv client from config...");
    client_.SetConfig(config_.get(), timeout);
    DXINFO("Done.");
  }

  void InitFromKey(const std::string& key, int timeout = 0) {
    DXINFO("Initializing feature kv client from key...");
    client_ = ::FeaturekvClient::GetClient(key, timeout);
    DXINFO("Done.");
  }
#endif

  bool Get(const std::string& key, std::string* value) {
    return client_.Get((uint32_t)default_table_id_, key, value) == 0;
  }

  bool BatchGet(const std::vector<std::string>& keys,
                std::vector<int16_t>* codes, std::vector<std::string>* values) {
    return client_.BatchGet((uint32_t)default_table_id_, keys, codes, values) ==
           0;
  }

  bool Get(int table_id, const std::string& key, std::string* value) {
    return client_.Get((uint32_t)table_id, key, value) == 0;
  }

  bool BatchGet(int table_id, const std::vector<std::string>& keys,
                std::vector<int16_t>* codes, std::vector<std::string>* values) {
    return client_.BatchGet((uint32_t)table_id, keys, codes, values) == 0;
  }
};

/************************************************************************/
/* ModelContext */
/************************************************************************/
class ModelContext {
 private:
  Graph graph_;
  Model model_;
  int version_ = 0;
  std::string value_;
  std::vector<std::string> keys_, values_;
  std::vector<int16_t> codes_;

 public:
  const Graph& graph() const noexcept { return graph_; }
  const TensorMap& param() const noexcept { return model_.param(); }
  int version() const noexcept { return version_; }

 public:
  void GetVersion(FeatureKVClientContext* client, int default_version) {
    DXINFO("Getting version...");
    if (!client->Get(FeatureKVUtil::GetVersionKey(), &value_) ||
        !FeatureKVUtil::GetVersion(value_, &version_)) {
      DXERROR("Failed to get version, use default version.");
      version_ = default_version;
    }
    std::cout << "version=" << std::endl;
    std::cout << version_ << std::endl;
  }

  void GetGraph(FeatureKVClientContext* client) {
    DXINFO("Getting graph...");
    DXCHECK_THROW(client->Get(FeatureKVUtil::GetGraphKey(), &value_));
    DXCHECK_THROW(FeatureKVUtil::GetGraph(value_, &graph_));
    std::cout << "graph=" << std::endl;
    std::cout << graph_.Dot() << std::endl;
  }

  void GetDenseParam(FeatureKVClientContext* client) {
    model_.Init(&graph_);
    DXCHECK_THROW(model_.InitParamPlaceholder());

    DXINFO("Getting dense param...");
    FeatureKVUtil::GetDenseParamKeys(model_.param(), &keys_);
    DXCHECK_THROW(client->BatchGet(keys_, &codes_, &values_));
    FeatureKVUtil::DenseParamParser parser;
    FeatureKVUtil::ParamParserStat stat;
    parser.Init(&graph_, model_.mutable_param());
    parser.Parse(keys_, values_, codes_, &stat);
    DXINFO("key_exist=%d", stat.key_exist);
    DXINFO("key_not_exist=%d", stat.key_not_exist);
    DXINFO("key_bad=%d", stat.key_bad);
    DXINFO("value_bad=%d", stat.value_bad);
    DXINFO("feature_kv_client_error=%d", stat.feature_kv_client_error);
  }
};

/************************************************************************/
/* LocalModelContext */
/************************************************************************/
class LocalModelContext {
 private:
  const Graph* graph_ = nullptr;
  const TensorMap* dense_param_ = nullptr;
  int version_ = 0;
  Model local_model_;
  TensorMap* local_param_ = nullptr;
  std::string target_name_;
  OpContext op_context_;
  PullRequest pull_request_;
  std::vector<std::string> keys_, values_;
  std::vector<int16_t> codes_;

 public:
  void Init(const ModelContext& context) {
    graph_ = &context.graph();
    dense_param_ = &context.param();
    version_ = context.version();

    local_model_.Init(graph_);
    DXCHECK_THROW(local_model_.InitParamPlaceholder());
    local_param_ = local_model_.mutable_param();

    // Check out graph target conventions.
    if (graph_->target_size() >= 3) {
      target_name_ = graph_->target(2).name();
    } else {
      target_name_ = graph_->target(1).name();
    }

    op_context_.Init(graph_, local_param_);
    DXCHECK_THROW(op_context_.InitOp({target_name_}, -1));
  }

  void AddInstance(const std::vector<int_t>& ids) {
    DXINFO("Adding instances...");
    Instance* inst = op_context_.mutable_inst();
    auto& X = inst->insert<csr_t>(X_NAME);
    // Add one instance with 'ids'.
    for (int_t id : ids) {
      X.emplace(id, 1);
    }
    X.add_row();
    // Add some instances without 'ids'.
    X.emplace(1, 1);
    X.emplace(2, 1);
    X.emplace(3, 1);
    X.add_row();
    X.emplace(4, 1);
    X.emplace(5, 1);
    X.emplace(6, 1);
    X.add_row();
    inst->set_batch(X.row());
    std::cout << "instance=" << std::endl;
    std::cout << *inst << std::endl;
  }

  void GetSparseParam(FeatureKVClientContext* client) {
    op_context_.InitPredict();
    op_context_.GetPullRequest(&pull_request_);

    DXINFO("Getting sparse param...");
    FeatureKVUtil::GetSparseParamKeys(pull_request_, &keys_);
    DXCHECK_THROW(client->BatchGet(keys_, &codes_, &values_));
    FeatureKVUtil::SparseParamParser parser;
    FeatureKVUtil::ParamParserStat stat;
    parser.set_view(1);
    parser.Init(graph_, local_param_, version_);
    parser.Parse(keys_, values_, codes_, &stat);
    DXINFO("key_exist=%d", stat.key_exist);
    DXINFO("key_not_exist=%d", stat.key_not_exist);
    DXINFO("key_bad=%d", stat.key_bad);
    DXINFO("value_bad=%d", stat.value_bad);
    DXINFO("feature_kv_client_error=%d", stat.feature_kv_client_error);

    DXINFO("Merging dense and sparse param...");
    for (const auto& entry : *dense_param_) {
      const std::string& name = entry.first;
      const Any& Wany = entry.second;
      if (Wany.is<tsr_t>()) {
        const auto& W = Wany.unsafe_to_ref<tsr_t>();
        // view, zero-copy
        local_param_->get<tsr_t>(name) = W.get_view();
      }
    }
    std::cout << "param=" << std::endl;
    std::cout << *local_param_ << std::endl;
  }

  void Predict() {
    DXINFO("Predicting...");
    op_context_.Predict();
    const auto* target = op_context_.ptr().get<tsr_t*>(target_name_);
    std::cout << "target=" << std::endl;
    std::cout << *target << std::endl;
  }
};

#if HAVE_WXG_FEATURE_KV_CLIENT == 1
DEFINE_string(config, "", "feature kv client config file");
DEFINE_string(key, "", "feature kv client key");
DEFINE_int32(timeout, 0, "feature kv client timeout in ms");
DEFINE_int32(table_id, 0, "feature kv table id");
#endif
DEFINE_string(ids, "10000,10001,10002,10003,10004,10005",
              "feature ids(separated by comma)");
DEFINE_int32(default_version, 2, "default feature kv protocol version");
std::vector<int_t> ids;

void CheckFlags() {
#if HAVE_WXG_FEATURE_KV_CLIENT == 1
  DXCHECK_THROW(!FLAGS_config.empty() || !FLAGS_key.empty());
  DXCHECK_THROW(FLAGS_table_id > 0);
#endif
  if (!FLAGS_ids.empty()) {
    DXCHECK_THROW(Split(FLAGS_ids, ",", &ids));
  }
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  FeatureKVClientContext client;
#if HAVE_WXG_FEATURE_KV_CLIENT == 0
  client.Init(ids);
#else
  client.set_default_table_id(FLAGS_table_id);
  if (!FLAGS_config.empty()) {
    client.InitFromConfig(FLAGS_config, FLAGS_timeout);
  } else {
    client.InitFromKey(FLAGS_key, FLAGS_timeout);
  }
#endif

  ModelContext model_context;
  model_context.GetVersion(&client, FLAGS_default_version);
  model_context.GetGraph(&client);
  model_context.GetDenseParam(&client);

  LocalModelContext local_model_context;
  local_model_context.Init(model_context);
  local_model_context.AddInstance(ids);
  local_model_context.GetSparseParam(&client);
  local_model_context.Predict();

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
