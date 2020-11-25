// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <gflags/gflags.h>
#include <thread>
#include "dist_flags.h"

namespace deepx_core {

void RunCoordServer();
void RunParamServer();
void RunWorker();

namespace {

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  if (FLAGS_is_ps) {
    if (FLAGS_ps_id == 0) {
      std::thread cs_thread(RunCoordServer);
      std::thread ps_thread(RunParamServer);
      ps_thread.join();
      cs_thread.join();
    } else {
      RunParamServer();
    }
    DXINFO("Param server %d normally exits.", FLAGS_ps_id);
  } else {
    RunWorker();
    DXINFO("Worker normally exits.");
  }

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
