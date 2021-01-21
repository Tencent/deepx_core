// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/dist_proto.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/ps/coord_server.h>
#include <deepx_core/ps/param_server.h>
#include <deepx_core/tensor/data_type.h>
#include <memory>
#include "dist_flags.h"
#include "model_zoo.h"

namespace deepx_core {
namespace {

/************************************************************************/
/* RankCoordServer */
/************************************************************************/
class RankCoordServer : public CoordServer {};

/************************************************************************/
/* RankParamServer */
/************************************************************************/
class RankParamServer : public ParamServer {
 private:
  struct SessionData {
    PullRequest pull_request;
    TensorMap param;
    TensorMap grad;
    TensorMap overwritten_param;
  };

 private:
  Graph graph_;
  ModelShard model_shard_;

 public:
  void Init();

 protected:
  void OnAccept(conn_t conn) override;
  void OnPullRequest(conn_t conn) override;
  void OnPushNotify(conn_t conn) override;
  void OnModelSaveRequest(conn_t conn) override;
  void OnTerminationNotify(conn_t conn) override;
};

void RankParamServer::Init() {
  if (FLAGS_is_train) {
    if (FLAGS_in_model.empty()) {
      std::unique_ptr<ModelZoo> model_zoo(NewModelZoo(FLAGS_model));
      DXCHECK_THROW(model_zoo);
      StringMap config;
      DXCHECK_THROW(ParseConfig(FLAGS_model_config, &config));
      DXCHECK_THROW(model_zoo->InitConfig(config));
      DXCHECK_THROW(model_zoo->InitGraph(&graph_));
    } else {
      DXCHECK_THROW(LoadGraph(FLAGS_in_model, &graph_));
    }

    model_shard_.seed(FLAGS_seed + FLAGS_ps_id * 10099);  // magic number
    model_shard_.InitShard(&FLAGS_shard, FLAGS_ps_id);
    model_shard_.InitGraph(&graph_);
    if (FLAGS_in_model.empty()) {
      DXCHECK_THROW(model_shard_.InitModel());
      DXCHECK_THROW(
          model_shard_.InitOptimizer(FLAGS_optimizer, FLAGS_optimizer_config));
      if (FLAGS_ts_enable) {
        DXCHECK_THROW(model_shard_.InitTSStore(
            (DataType::ts_t)FLAGS_ts_now,
            (DataType::ts_t)FLAGS_ts_expire_threshold));
      }
      if (FLAGS_freq_filter_threshold > 0) {
        DXCHECK_THROW(model_shard_.InitFreqStore(
            (DataType::freq_t)FLAGS_freq_filter_threshold));
      }
    } else {
      DXCHECK_THROW(model_shard_.LoadModel(FLAGS_in_model));
      DXCHECK_THROW(
          model_shard_.LoadOptimizer(FLAGS_in_model, FLAGS_optimizer_config));
      if (FLAGS_ts_enable) {
        if (!model_shard_.LoadTSStore(FLAGS_in_model, FLAGS_ts_now,
                                      FLAGS_ts_expire_threshold)) {
          DXCHECK_THROW(model_shard_.InitTSStore(
              (DataType::ts_t)FLAGS_ts_now,
              (DataType::ts_t)FLAGS_ts_expire_threshold));
        }
      }
      if (FLAGS_freq_filter_threshold > 0) {
        if (!model_shard_.LoadFreqStore(
                FLAGS_in_model,
                (DataType::freq_t)FLAGS_freq_filter_threshold)) {
          DXCHECK_THROW(model_shard_.InitFreqStore(
              (DataType::freq_t)FLAGS_freq_filter_threshold));
        }
      }
    }
    if (!FLAGS_warmup_model.empty()) {
      DXCHECK_THROW(model_shard_.WarmupModel(FLAGS_warmup_model));
      DXCHECK_THROW(model_shard_.WarmupOptimizer(FLAGS_warmup_model));
      if (FLAGS_ts_enable) {
        DXCHECK_THROW(model_shard_.WarmupTSStore(FLAGS_warmup_model));
      }
      if (FLAGS_freq_filter_threshold > 0) {
        DXCHECK_THROW(model_shard_.WarmupFreqStore(FLAGS_warmup_model));
      }
    }
  } else {
    DXCHECK_THROW(LoadGraph(FLAGS_in_model, &graph_));
    model_shard_.InitShard(&FLAGS_shard, FLAGS_ps_id);
    model_shard_.InitGraph(&graph_);
    DXCHECK_THROW(model_shard_.LoadModel(FLAGS_in_model));
  }

  DXCHECK_THROW(model_shard_.model().HasSRM());

  if (FLAGS_is_train && config_.thread > 1) {
    DXCHECK_THROW(model_shard_.InitLock());
  }
}

void RankParamServer::OnAccept(conn_t conn) {
  conn->mutable_user_data()->emplace(SessionData());
  ParamServer::OnAccept(conn);
}

void RankParamServer::OnPullRequest(conn_t conn) {
  auto& session_data = conn->mutable_user_data()->unsafe_to_ref<SessionData>();

  {
    const const_string_view& buf = conn->in_message().pull_request().buf;
    InputStringStream is;
    is.SetView(buf.data(), buf.size());
    is >> session_data.pull_request;
    DXCHECK_THROW(is);
  }

  model_shard_.Pull(&session_data.pull_request, &session_data.param);

  {
    std::string& buf =
        conn->mutable_out_message()->mutable_pull_response()->buf;
    OutputStringStream os;
    buf.clear();
    os.SetView(&buf);
    os << session_data.param;
    DXCHECK_THROW(os);
  }
}

void RankParamServer::OnPushNotify(conn_t conn) {
  auto& session_data = conn->mutable_user_data()->unsafe_to_ref<SessionData>();

  const const_string_view& buf = conn->in_message().push_notify().buf;
  InputStringStream is;
  is.SetView(buf.data(), buf.size());
  // view, zero-copy
  ReadView(is, session_data.grad);
  ReadView(is, session_data.overwritten_param);
  DXCHECK_THROW(is);

  model_shard_.Push(&session_data.grad, &session_data.overwritten_param);
}

void RankParamServer::OnModelSaveRequest(conn_t /*conn*/) {
  if (FLAGS_ps_id == 0) {
    DXCHECK_THROW(SaveGraph(FLAGS_out_model, graph_));
    DXCHECK_THROW(SaveShard(FLAGS_out_model, FLAGS_shard));
  }
  if (FLAGS_out_model_remove_zeros) {
    model_shard_.mutable_model()->RemoveZerosSRM();
  }
  if (FLAGS_ts_enable && FLAGS_ts_expire_threshold > 0) {
    model_shard_.ExpireTSStore();
  }
  DXCHECK_THROW(model_shard_.SaveModel(FLAGS_out_model));
  if (!FLAGS_out_text_model.empty()) {
    DXCHECK_THROW(model_shard_.SaveTextModel(FLAGS_out_text_model));
  }
  if (!FLAGS_out_feature_kv_model.empty()) {
    DXCHECK_THROW(model_shard_.SaveFeatureKVModel(
        FLAGS_out_feature_kv_model, FLAGS_out_feature_kv_protocol_version));
  }
  DXCHECK_THROW(model_shard_.SaveOptimizer(FLAGS_out_model));
  if (FLAGS_ts_enable) {
    DXCHECK_THROW(model_shard_.SaveTSStore(FLAGS_out_model));
  }
  if (FLAGS_freq_filter_threshold > 0) {
    DXCHECK_THROW(model_shard_.SaveFreqStore(FLAGS_out_model));
  }
}

void RankParamServer::OnTerminationNotify(conn_t /*conn*/) {}

}  // namespace

void RunCoordServer() {
  CoordServerConfig config;
  config.listen_endpoint = FLAGS_cs_endpoint;
  config.ps_endpoints = FLAGS_ps_endpoints;
  config.epoch = FLAGS_is_train ? FLAGS_epoch : 1;
  DXCHECK_THROW(AutoFileSystem::ListRecursive(FLAGS_in, true,
                                              &config.file_dispatcher_files));
  DXCHECK_THROW(!config.file_dispatcher_files.empty());
  DXINFO("Got %zu files.", config.file_dispatcher_files.size());
  for (const std::string& file : config.file_dispatcher_files) {
    DXINFO("  %s", file.c_str());
  }
  if (FLAGS_is_train) {
    config.file_dispatcher_reverse = FLAGS_reverse_in;
    config.file_dispatcher_shuffle = FLAGS_shuffle_in;
  } else {
    config.file_dispatcher_reverse = 0;
    config.file_dispatcher_shuffle = 0;
  }
  config.file_dispatcher_timeout = 0;
  if (FLAGS_is_train) {
    config.dump_model = 1;
  } else {
    config.dump_model = 0;
  }

  RankCoordServer server;
  server.set_config(config);
  server.Run();
}

void RunParamServer() {
  TcpServerConfig config;
  config.listen_endpoint = FLAGS_ps_endpoints[FLAGS_ps_id];
  config.thread = FLAGS_ps_thread;

  RankParamServer server;
  server.set_config(config);
  server.Init();
  server.Run();
}

}  // namespace deepx_core
