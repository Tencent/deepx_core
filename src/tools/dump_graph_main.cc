// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/graph.h>
#include <gflags/gflags.h>
#include <iostream>

DEFINE_string(in, "", "input model dir/graph file");

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
  DXCHECK_THROW(!FLAGS_in.empty());

  Graph graph;
  if (LoadGraph(FLAGS_in, &graph) || graph.Load(FLAGS_in)) {
    std::cout << graph.Dot() << std::endl;
  }

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
