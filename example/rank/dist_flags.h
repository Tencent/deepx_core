// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#pragma once
#include <deepx_core/graph/shard.h>
#include <deepx_core/ps/tcp_connection.h>
#include <gflags/gflags.h>
#include <vector>

DECLARE_string(sub_command);
DECLARE_string(role);
DECLARE_string(cs_addr);
DECLARE_string(ps_addrs);
DECLARE_int32(ps_id);
DECLARE_int32(ps_thread);

DECLARE_string(instance_reader);
DECLARE_string(instance_reader_config);
DECLARE_string(model);
DECLARE_string(model_config);
DECLARE_string(optimizer);
DECLARE_string(optimizer_config);
DECLARE_int32(epoch);
DECLARE_int32(batch);
DECLARE_string(in);
DECLARE_int32(reverse_in);
DECLARE_int32(shuffle_in);
DECLARE_string(in_model);
DECLARE_string(warmup_model);
DECLARE_int32(out_model_remove_zeros);
DECLARE_string(out_model);
DECLARE_string(out_text_model);
DECLARE_string(out_feature_kv_model);
DECLARE_int32(out_feature_kv_protocol_version);
DECLARE_string(out_predict);
DECLARE_int32(verbose);
DECLARE_int32(seed);
DECLARE_int32(ts_enable);
DECLARE_uint64(ts_now);
DECLARE_uint64(ts_expire_threshold);
DECLARE_uint64(freq_filter_threshold);

namespace deepx_core {

extern int FLAGS_is_train;
extern int FLAGS_is_ps;
extern TcpEndpoint FLAGS_cs_endpoint;
extern std::vector<TcpEndpoint> FLAGS_ps_endpoints;
extern int FLAGS_ps_size;
extern Shard FLAGS_shard;

void CheckFlags();

}  // namespace deepx_core
