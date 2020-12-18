// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <gflags/gflags.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "model_server.h"

DEFINE_string(in, "", "input file");
DEFINE_string(in_graph, "", "input graph file");
DEFINE_string(in_model, "", "input model param file");

namespace deepx_core {
namespace {

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  ModelServer model_server;
  std::string line;
  std::istringstream is;
  uint64_t feature_id;
  float feature_value;
  std::vector<features_t> batch_features;
  std::vector<std::vector<float>> batch_probs;

  if (!FLAGS_in.empty()) {
    DXCHECK_THROW(model_server.Load(FLAGS_in));
  } else {
    DXCHECK_THROW(!FLAGS_in_graph.empty());
    DXCHECK_THROW(!FLAGS_in_model.empty());
    DXCHECK_THROW(model_server.LoadGraph(FLAGS_in_graph));
    DXCHECK_THROW(model_server.LoadModel(FLAGS_in_model));
  }

  auto op_context = model_server.NewOpContext();
  DXCHECK_THROW(op_context);

  while (std::getline(std::cin, line)) {
    is.clear();
    is.str(line);
    batch_features.resize(1);
    auto& features = batch_features[0];
    features.clear();
    while (is >> feature_id >> feature_value) {
      features.emplace_back(feature_id, feature_value);
    }
    DXCHECK_THROW(model_server.BatchPredict(op_context.get(), batch_features,
                                            &batch_probs));
    auto& probs = batch_probs[0];
    for (size_t i = 0; i < probs.size(); ++i) {
      std::cout << probs[i];
      if (i != probs.size() - 1) {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
