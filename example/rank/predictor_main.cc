// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/misc.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/shard.h>
#include <deepx_core/ps/file_dispatcher.h>
#include <gflags/gflags.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "trainer_context.h"

DEFINE_string(instance_reader, "libsvm", "instance reader name");
DEFINE_string(instance_reader_config, "", "instance reader config");
DEFINE_int32(batch, 32, "batch size");
DEFINE_int32(thread, 1, "# of threads");
DEFINE_string(in, "", "input dir/file of testing data");
DEFINE_string(in_model, "", "input model dir");
DEFINE_string(out_predict, "", "output predict dir(optional)");
DEFINE_int32(verbose, 1, "verbose level: 0-10");

namespace deepx_core {
namespace {

Shard FLAGS_shard;

/************************************************************************/
/* Predictor */
/************************************************************************/
class Predictor {
 protected:
  Graph graph_;

  std::vector<std::string> files_;
  FileDispatcher file_dispatcher_;

  std::vector<std::unique_ptr<TrainerContext>> contexts_tls_;

 public:
  virtual ~Predictor() = default;
  virtual void Init();
  virtual void Predict();
  void PredictEntry(int thread_id);
  virtual void PredictFile(int thread_id, const std::string& file,
                           const std::string& out_file);
};

void Predictor::Init() {
  DXCHECK_THROW(LoadGraph(FLAGS_in_model, &graph_));

  DXCHECK_THROW(AutoFileSystem::ListRecursive(FLAGS_in, true, &files_));

  file_dispatcher_.set_reverse(0);
  file_dispatcher_.set_shuffle(0);
  file_dispatcher_.set_timeout(0);

  if (!IsStdinStdoutPath(FLAGS_out_predict)) {
    std::string new_path;
    if (AutoFileSystem::BackupIfExists(FLAGS_out_predict, &new_path)) {
      DXINFO("Backed up %s to %s.", FLAGS_out_predict.c_str(),
             new_path.c_str());
    }

    if (!AutoFileSystem::Exists(FLAGS_out_predict)) {
      DXCHECK_THROW(AutoFileSystem::MakeDir(FLAGS_out_predict));
    }
  }
}

void Predictor::Predict() {
  file_dispatcher_.PreTrain(files_);
  file_dispatcher_.PreEpoch();
  std::vector<std::thread> threads;
  for (int j = 0; j < FLAGS_thread; ++j) {
    threads.emplace_back(&Predictor::PredictEntry, this, j);
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
}

void Predictor::PredictEntry(int thread_id) {
  for (;;) {
    size_t file_size = 0;
    std::string file;
    if (!file_dispatcher_.WorkerDispatchFile(&file)) {
      DXINFO("[%d] [%3.1f%%] Predicting completed. ", thread_id, 100.0);
      break;
    }

    DXINFO("[%d] [%3.1f%%] Predicting %s...", thread_id,
           (100.0 - 100.0 * file_size / files_.size()), file.c_str());
    PredictFile(thread_id, file, GetOutputPredictFile(FLAGS_out_predict, file));
    (void)file_dispatcher_.WorkerFinishFile(file);
  }
}

void Predictor::PredictFile(int thread_id, const std::string& file,
                            const std::string& out_file) {
  TrainerContext* context = contexts_tls_[thread_id].get();
  context->PredictFile(thread_id, file, out_file);
}

/************************************************************************/
/* PredictorNonShard */
/************************************************************************/
class PredictorNonShard : public Predictor {
 private:
  ModelShard model_shard_;

 public:
  void Init() override;
};

void PredictorNonShard::Init() {
  Predictor::Init();

  model_shard_.InitShard(&FLAGS_shard, 0);
  model_shard_.InitGraph(&graph_);
  DXCHECK_THROW(model_shard_.LoadModel(FLAGS_in_model));

  contexts_tls_.resize(FLAGS_thread);
  for (int i = 0; i < FLAGS_thread; ++i) {
    std::unique_ptr<TrainerContextNonShard> context(new TrainerContextNonShard);
    context->set_instance_reader(FLAGS_instance_reader);
    context->set_instance_reader_config(FLAGS_instance_reader_config);
    context->set_batch(FLAGS_batch);
    context->set_verbose(FLAGS_verbose);
    // Check out graph target conventions.
    context->set_target_name(graph_.target(1).name());
    context->Init(&model_shard_);
    contexts_tls_[i] = std::move(context);
  }
}

/************************************************************************/
/* PredictorShard */
/************************************************************************/
class PredictorShard : public Predictor {
 private:
  int shard_size_ = 0;
  std::vector<ModelShard> model_shards_;
  std::vector<ModelShard> model_shards_tls_;

 public:
  void Init() override;
  void Predict() override;
};

void PredictorShard::Init() {
  Predictor::Init();

  shard_size_ = FLAGS_shard.shard_size();
  model_shards_.resize(shard_size_);
  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i].InitShard(&FLAGS_shard, i);
    model_shards_[i].InitGraph(&graph_);
    DXCHECK_THROW(model_shards_[i].LoadModel(FLAGS_in_model));
  }

  contexts_tls_.resize(FLAGS_thread);
  model_shards_tls_.resize(FLAGS_thread);
  for (int i = 0; i < FLAGS_thread; ++i) {
    model_shards_tls_[i].InitShard(&FLAGS_shard, 0);
    model_shards_tls_[i].InitGraph(&graph_);
    DXCHECK_THROW(model_shards_tls_[i].InitModelPlaceholder());
    std::unique_ptr<TrainerContextShard> context(new TrainerContextShard);
    context->set_instance_reader(FLAGS_instance_reader);
    context->set_instance_reader_config(FLAGS_instance_reader_config);
    context->set_batch(FLAGS_batch);
    context->set_verbose(FLAGS_verbose);
    // Check out graph target conventions.
    context->set_target_name(graph_.target(1).name());
    context->Init(&model_shards_, &model_shards_tls_[i]);
    contexts_tls_[i] = std::move(context);
  }
}

void PredictorShard::Predict() {
  for (int i = 0; i < shard_size_; ++i) {
    DXCHECK_THROW(model_shards_[i].InitThreadPool());
    model_shards_[i].StartThreadPool();
  }

  Predictor::Predict();

  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i].StopThreadPool();
  }
}

/************************************************************************/
/* main */
/************************************************************************/
void CheckFlags() {
  AutoFileSystem fs;

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
  DXCHECK_THROW(FLAGS_batch > 0);
  DXCHECK_THROW(FLAGS_thread > 0);

  CanonicalizePath(&FLAGS_in);
  DXCHECK_THROW(!FLAGS_in.empty());
  DXCHECK_THROW(fs.Open(FLAGS_in));
  if (IsStdinStdoutPath(FLAGS_in)) {
    if (FLAGS_thread != 1) {
      DXINFO("--thread will be set to 1.");
      FLAGS_thread = 1;
    }
  }

  CanonicalizePath(&FLAGS_in_model);
  DXCHECK_THROW(!FLAGS_in_model.empty());
  DXCHECK_THROW(fs.Open(FLAGS_in_model));
  DXCHECK_THROW(!IsStdinStdoutPath(FLAGS_in_model));

  CanonicalizePath(&FLAGS_out_predict);
  if (FLAGS_out_predict.empty()) {
    if (IsStdinStdoutPath(FLAGS_in)) {
      FLAGS_out_predict = "stdin.predict";
    } else {
      FLAGS_out_predict = FLAGS_in + ".predict";
    }
    DXINFO("Didn't specify --out_predict, output to: %s.",
           FLAGS_out_predict.c_str());
  }
  DXCHECK_THROW(fs.Open(FLAGS_out_predict));

  DXCHECK_THROW(FLAGS_verbose >= 0);

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

  std::unique_ptr<Predictor> predictor;
  if (FLAGS_shard.shard_mode() == 0) {
    predictor.reset(new PredictorNonShard);
  } else {
    predictor.reset(new PredictorShard);
  }

  predictor->Init();
  predictor->Predict();

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
