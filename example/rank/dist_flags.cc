// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include "dist_flags.h"
#include <deepx_core/common/any_map.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/feature_kv_util.h>
#include <deepx_core/tensor/data_type.h>
#include <limits>  // std::numeric_limits
#include <string>

DEFINE_string(sub_command, "train", "train or predict");
DEFINE_string(role, "ps", "ps or wk");
DEFINE_string(cs_addr, "127.0.0.1:61000", "coord server address");
DEFINE_string(ps_addrs, "127.0.0.1:60000", "param server addresses");
DEFINE_int32(ps_id, 0, "param server id");
DEFINE_int32(ps_thread, 1, "# of param server working threads");

DEFINE_string(instance_reader, "libsvm", "instance reader name");
DEFINE_string(instance_reader_config, "", "instance reader config");
DEFINE_string(model, "lr", "model name");
DEFINE_string(model_config, "", "model config");
DEFINE_string(optimizer, "adagrad", "optimizer name");
DEFINE_string(optimizer_config, "", "optimizer config");
DEFINE_int32(epoch, 1, "# of epochs");
DEFINE_int32(batch, 32, "batch size");
DEFINE_string(in, "", "input dir/file of training/testing data");
DEFINE_int32(reverse_in, 0, "reverse input files");
DEFINE_int32(shuffle_in, 1, "shuffle input files for each epoch");
DEFINE_string(in_model, "", "input model dir");
DEFINE_string(warmup_model, "", "warmup model dir");
DEFINE_int32(out_model_remove_zeros, 0, "remove zeros from output model");
DEFINE_string(out_model, "", "output model dir");
DEFINE_string(out_text_model, "", "output text model dir(optional)");
DEFINE_string(out_feature_kv_model, "",
              "output feature kv model dir(optional)");
DEFINE_int32(out_feature_kv_protocol_version, 2,
             "output feature kv protocol version");
DEFINE_string(out_predict, "", "output predict dir(optional)");
DEFINE_int32(verbose, 1, "verbose level: 0-10");
DEFINE_int32(seed, 9527, "seed of random engine");
DEFINE_int32(ts_enable, 0, "enable timestamp");
DEFINE_uint64(ts_now, 0, "timestamp of now");
DEFINE_uint64(ts_expire_threshold, 0, "timestamp expiration threshold");
DEFINE_uint64(freq_filter_threshold, 0,
              "feature frequency filtering threshold");

namespace deepx_core {

int FLAGS_is_train = 0;
int FLAGS_is_ps = 0;
TcpEndpoint FLAGS_cs_endpoint;
std::vector<TcpEndpoint> FLAGS_ps_endpoints;
int FLAGS_ps_size = 0;
Shard FLAGS_shard;

void CheckFlags() {
  AutoFileSystem fs;

  DXCHECK_THROW(FLAGS_sub_command == "train" || FLAGS_sub_command == "predict");
  if (FLAGS_sub_command == "train") {
    FLAGS_is_train = 1;
  } else {
    FLAGS_is_train = 0;
  }

  DXCHECK_THROW(FLAGS_role == "ps" || FLAGS_role == "wk");
  if (FLAGS_role == "ps") {
    FLAGS_is_ps = 1;
  } else {
    FLAGS_is_ps = 0;
  }

  FLAGS_cs_endpoint = MakeTcpEndpoint(FLAGS_cs_addr);
  FLAGS_ps_endpoints = MakeTcpEndpoints(FLAGS_ps_addrs);
  FLAGS_ps_size = (int)FLAGS_ps_endpoints.size();
  DXCHECK_THROW(0 <= FLAGS_ps_id && FLAGS_ps_id < FLAGS_ps_size);
  DXCHECK_THROW(FLAGS_ps_thread > 0);

  DXCHECK_THROW(!FLAGS_instance_reader.empty());
  StringMap config;
  DXCHECK_THROW(ParseConfig(FLAGS_instance_reader_config, &config));
  if (config.count("batch") > 0) {
    int batch = std::stoi(config.at("batch"));
    if (batch != FLAGS_batch) {
      DXINFO(
          "Batch size from --instance_reader_config and --batch are "
          "inconsistent, use %d.",
          batch);
      FLAGS_batch = batch;
    }
  }

  if (FLAGS_is_train) {
    DXCHECK_THROW(FLAGS_epoch > 0);
  }

  DXCHECK_THROW(FLAGS_batch > 0);

  CanonicalizePath(&FLAGS_in);
  DXCHECK_THROW(!FLAGS_in.empty());
  DXCHECK_THROW(fs.Open(FLAGS_in));
  DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_in));

  if (FLAGS_is_train) {
    CanonicalizePath(&FLAGS_in_model);
    if (FLAGS_in_model.empty()) {
      DXCHECK_THROW(!FLAGS_model.empty());
      DXCHECK_THROW(!FLAGS_optimizer.empty());
    } else {
      DXCHECK_THROW(fs.Open(FLAGS_in_model));
      DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_in_model));
    }
    CanonicalizePath(&FLAGS_warmup_model);
    if (!FLAGS_warmup_model.empty()) {
      DXCHECK_THROW(fs.Open(FLAGS_warmup_model));
      DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_warmup_model));
    }
  } else {
    CanonicalizePath(&FLAGS_in_model);
    DXCHECK_THROW(!FLAGS_in_model.empty());
    DXCHECK_THROW(fs.Open(FLAGS_in_model));
    DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_in_model));
  }

  if (FLAGS_is_train) {
    CanonicalizePath(&FLAGS_out_model);
    DXCHECK_THROW(!FLAGS_out_model.empty());
    DXCHECK_THROW(fs.Open(FLAGS_out_model));
    DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_out_model));
    (void)AutoFileSystem::MakeDir(FLAGS_out_model);

    CanonicalizePath(&FLAGS_out_text_model);
    if (!FLAGS_out_text_model.empty()) {
      DXCHECK_THROW(fs.Open(FLAGS_out_text_model));
      DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_out_text_model));
      (void)AutoFileSystem::MakeDir(FLAGS_out_text_model);
    }

    CanonicalizePath(&FLAGS_out_feature_kv_model);
    if (!FLAGS_out_feature_kv_model.empty()) {
      DXCHECK_THROW(fs.Open(FLAGS_out_feature_kv_model));
      DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_out_feature_kv_model));
      (void)AutoFileSystem::MakeDir(FLAGS_out_feature_kv_model);
      FeatureKVUtil::CheckVersion(FLAGS_out_feature_kv_protocol_version);
    }
  } else {
    CanonicalizePath(&FLAGS_out_predict);
    if (FLAGS_out_predict.empty()) {
      FLAGS_out_predict = FLAGS_in + ".predict";
      DXINFO("Didn't specify --out_predict, output to: %s.",
             FLAGS_out_predict.c_str());
    }
    DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_out_predict));
    (void)AutoFileSystem::MakeDir(FLAGS_out_predict);
  }

  DXCHECK_THROW(FLAGS_verbose >= 0);

  if (FLAGS_is_train) {
    if (FLAGS_ts_enable) {
      DXCHECK_THROW(FLAGS_ts_now <=
                    (google::uint64)std::numeric_limits<DataType::ts_t>::max());
      DXCHECK_THROW(FLAGS_ts_expire_threshold <=
                    (google::uint64)std::numeric_limits<DataType::ts_t>::max());
    }

    DXCHECK_THROW(FLAGS_freq_filter_threshold <=
                  (google::uint64)std::numeric_limits<DataType::freq_t>::max());
  }

  FLAGS_shard.InitShard(FLAGS_ps_size, "default");
}

}  // namespace deepx_core
