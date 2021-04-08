// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/tensor/data_type.h>
#include <gflags/gflags.h>
#include <algorithm>  // std::sort
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

DEFINE_string(in, "", "input dir/file of <label, probability> data");

namespace deepx_core {
namespace {

using float_t = DataType::float_t;
using ll_math_t = DataType::ll_math_t;

struct EvalItem {
  float_t label = 0;
  float_t prob = 0;
  EvalItem() = default;
  EvalItem(float_t _label, float_t _prob) : label(_label), prob(_prob) {}
};

void EvalAUC(const std::vector<EvalItem>& items) {
  if (items.empty()) {
    return;
  }

  double auc;
  size_t pos = 0, acc_pos = 0;
  for (const EvalItem& item : items) {
    if (item.label > 0) {
      pos += 1;
    } else {
      acc_pos += pos;
    }
  }
  if (pos == 0 || pos == items.size()) {
    auc = 1;
  } else {
    auc = 1.0 * acc_pos / pos / (items.size() - pos);
  }
  std::cout << "auc=" << auc << std::endl;

  double total_loss = 0, total_prob = 0;
  for (const EvalItem& item : items) {
    if (item.label > 0) {
      total_loss -= ll_math_t::safe_log(item.prob);
    } else {
      total_loss -= ll_math_t::safe_log(1 - item.prob);
    }
    total_prob += item.prob;
  }
  std::cout << "loss=" << total_loss / items.size() << std::endl;
  std::cout << "predictive_ctr=" << total_prob / items.size() << std::endl;
  std::cout << "statistical_ctr=" << 1.0 * pos / items.size() << std::endl;
}

void EvalAUC(const std::vector<std::string>& files) {
  std::vector<EvalItem> items;
  std::string line;
  std::istringstream iss;
  float_t label, prob;

  items.reserve(1000000);  // magic number
  for (const std::string& file : files) {
    DXINFO("Loading from %s...", file.c_str());
    AutoInputFileStream is;
    DXCHECK_THROW(is.Open(file));
    while (GetLine(is, line)) {
      iss.clear();
      iss.str(line);
      if (!(iss >> label >> prob)) {
        DXERROR("Invalid line: %s.", line.c_str());
        continue;
      }
      items.emplace_back(label, prob);
    }
  }

  DXINFO("Sorting...");
  std::sort(
      items.begin(), items.end(),
      [](const EvalItem& a, const EvalItem& b) { return a.prob > b.prob; });
  DXINFO("Done.");

  DXINFO("Evaluating AUC...");
  EvalAUC(items);
  DXINFO("Done.");
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);
  DXCHECK_THROW(!FLAGS_in.empty());

  std::vector<std::string> files;
  DXCHECK_THROW(AutoFileSystem::ListRecursive(FLAGS_in, true, &files));

  EvalAUC(files);
  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace deepx_core

int main(int argc, char** argv) { return deepx_core::main(argc, argv); }
