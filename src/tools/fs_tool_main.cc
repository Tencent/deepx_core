// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <gflags/gflags.h>
#include <iostream>
#include <string>

namespace deepx_core {
namespace {

void ShowHelp() {
  std::cout << "Usage:" << std::endl;
  std::cout << "  text [file]" << std::endl;
  std::cout << "  testwr [file]" << std::endl;
  std::cout << "  ls [path]" << std::endl;
  std::cout << "  lsr [path]" << std::endl;
}

int real_main(int argc, char** argv) {
  if (argc < 3) {
    ShowHelp();
    return 1;
  }

  std::string action = argv[1];
  std::string path = argv[2];
  if (action == "text") {
    AutoInputFileStream is;
    if (!is.Open(path)) {
      DXERROR("Failed to open: %s.", path.c_str());
      return 1;
    }

    std::string line;
    while (GetLine(is, line)) {
      std::cout << line << std::endl;
    }
  } else if (action == "testwr") {
    AutoOutputFileStream os;
    if (!os.Open(path)) {
      DXERROR("Failed to open: %s.", path.c_str());
      return 1;
    }
    for (int i = 0; i < 100; ++i) {
      std::string s = "This is the string " + std::to_string(i) + ".";
      os << s;
    }
    os.Close();

    AutoInputFileStream is;
    if (!is.Open(path)) {
      DXERROR("Failed to open: %s.", path.c_str());
      return 1;
    }
    for (;;) {
      std::string s;
      is >> s;
      if (!is) {
        break;
      }
      std::cout << s << std::endl;
    }
    is.Close();
  } else if (action == "ls" || action == "lsr") {
    AutoFileSystem fs;
    if (!fs.Open(path)) {
      DXERROR("Failed to open: %s.", path.c_str());
      return 1;
    }

    std::vector<std::pair<FilePath, FileStat>> children;
    if (action == "ls") {
      if (!fs.List(path, false, &children)) {
        DXERROR("Failed to list: %s.", path.c_str());
        return 1;
      }
    } else {
      if (!fs.ListRecursive(path, false, &children)) {
        DXERROR("Failed to list: %s.", path.c_str());
        return 1;
      }
    }

    for (const auto& entry : children) {
      const FileStat& stat = entry.second;
      if (stat.IsDir()) {
        std::cout << "dir ";
      } else if (stat.IsRegFile()) {
        std::cout << "reg ";
      } else if (stat.IsSymLink()) {
        std::cout << "sym ";
      } else if (stat.IsOther()) {
        std::cout << "??? ";
      }
      std::cout.width(12);
      std::cout << stat.GetFileSize();
      std::cout << " ";
      std::cout << entry.first.str();
      std::cout << std::endl;
    }
  } else {
    ShowHelp();
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);
  int ret = real_main(argc, argv);
  google::ShutDownCommandLineFlags();
  return ret;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
