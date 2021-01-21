// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/shard.h>
#include <gflags/gflags.h>
#include <memory>
#include <string>
#include <vector>

DEFINE_string(in_model, "", "input model dir");
DEFINE_string(out_model, "", "output model file(optional)");

namespace deepx_core {
namespace {

Shard FLAGS_shard;

void CheckFlags() {
  AutoFileSystem fs;

  CanonicalizePath(&FLAGS_in_model);
  DXCHECK_THROW(!FLAGS_in_model.empty());
  DXCHECK_THROW(fs.Open(FLAGS_in_model));
  DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_in_model));

  CanonicalizePath(&FLAGS_out_model);
  if (FLAGS_out_model.empty()) {
    FLAGS_out_model = FLAGS_in_model + ".merge";
    DXINFO("Didn't specify --out_model, output to: %s.",
           FLAGS_out_model.c_str());
  }
  DXCHECK_THROW(fs.Open(FLAGS_out_model));
  DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_out_model));
  DXCHECK_THROW(LoadShard(FLAGS_in_model, &FLAGS_shard));
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

  int shard_size = FLAGS_shard.shard_size();
  DXINFO("shard_size=%d", shard_size);
  if (shard_size == 1) {
    DXINFO("Nothing to merge.");
    google::ShutDownCommandLineFlags();
    return 0;
  }

  Graph graph;
  DXCHECK_THROW(LoadGraph(FLAGS_in_model, &graph));

  std::vector<std::unique_ptr<ModelShard>> model_shards(shard_size);
  for (int i = 0; i < shard_size; ++i) {
    model_shards[i].reset(new ModelShard);
    model_shards[i]->InitShard(&FLAGS_shard, i);
    model_shards[i]->InitGraph(&graph);
    DXCHECK_THROW(model_shards[i]->LoadModel(FLAGS_in_model));
  }

  Shard shard;
  shard.InitNonShard();
  std::unique_ptr<ModelShard> merged;
  merged.reset(new ModelShard);
  merged->InitShard(&shard, 0);
  merged->InitGraph(&graph);
  DXCHECK_THROW(merged->InitModelPlaceholder());
  for (int i = 0; i < shard_size; ++i) {
    merged->mutable_model()->Merge(model_shards[i]->mutable_model());
  }

  std::string new_path;
  if (AutoFileSystem::BackupIfExists(FLAGS_out_model, &new_path)) {
    DXINFO("Backed up %s to %s.", FLAGS_out_model.c_str(), new_path.c_str());
  }
  DXCHECK_THROW(merged->model().Save(FLAGS_out_model));

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
