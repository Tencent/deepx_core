// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/misc.h>
#include <deepx_core/common/str_util.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>

namespace deepx_core {

size_t GetVerboseBatch(int verbose) noexcept {
  switch (verbose) {
    case 1:
      return 1024;
    case 2:
      return 512;
    case 3:
      return 256;
    case 4:
      return 128;
    case 5:
      return 64;
    case 6:
      return 32;
    case 7:
      return 16;
    case 8:
      return 8;
    case 9:
      return 4;
    case 10:
    default:
      return 1;
  }
}

std::string GetOutputPredictFile(const std::string& dir,
                                 const std::string& file) {
  if (IsStdinStdoutPath(dir)) {
    return StdinStdoutPath();
  }

  if (IsStdinStdoutPath(file)) {
    return dir + "/stdin";
  }

  std::string out_file = file;
  if (IsHDFSPath(out_file)) {
    // trim leading hdfs://
    out_file.erase(0, 7);
  }
  if (IsGzipFile(out_file)) {
    // trim trailing .gz
    out_file.resize(out_file.size() - 3);
  }
  for (char& c : out_file) {
    if (IsDirSeparator(c) || c == '.' || c == ':') {
      c = '_';
    }
  }
  return dir + "/" + out_file;
}

bool ParseDeepDims(const std::string& v, std::vector<int>* deep_dims,
                   const char* gflag) {
  if (v.empty()) {
    DXERROR("Please specify %s.", gflag);
    return false;
  }

  if (!Split(v, ",", deep_dims)) {
    DXERROR("Invalid %s: %s.", gflag, v.c_str());
    return false;
  }

  for (int dim : *deep_dims) {
    if (dim <= 0) {
      DXERROR("Invalid %s: %s.", gflag, v.c_str());
      return false;
    }
  }
  return true;
}

bool ParseDeepDimsAppendOne(const std::string& v, std::vector<int>* deep_dims,
                            const char* gflag) {
  if (!ParseDeepDims(v, deep_dims, gflag)) {
    return false;
  }

  if (deep_dims->back() != 1) {
    deep_dims->emplace_back(1);
  }
  return true;
}

}  // namespace deepx_core
